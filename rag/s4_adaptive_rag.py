"""S4 Adaptive RAG pipeline implementation.

Query is gated by S3 disjunction classifier query check.
Retrieved/injected documents are scanned via SentenceWindowScanner, scored
via RetrievalRiskScorer, and adaptively quarantined or sanitized.
"""
from __future__ import annotations

import os
import time
import config
from rag.baseline_rag import BaselineRAG
from rag.defended_rag import BaseHelpers, BLOCKED_RESPONSE, NO_CLEAN_CONTEXT
from rag.llm_client import SAFETY_BLOCKED
from security.rule_classifier import RuleClassifier
from security.simple_ml_detector import SimpleMLDetector, DEFAULT_MODEL_PATH
from security.s4_retrieval_risk import RetrievalRiskScorer, S4RiskConfig, cosine_similarity
from security.s4_sentence_scanner import SentenceWindowScanner


class RuleOrMLClassifier:
    """S3 classifier composition: RuleClassifier OR persisted S2 TF-IDF/LR."""

    NAME = "Rule_OR_TFIDF_LR"

    def __init__(self, rule_clf=None, ml_clf=None):
        self.rule_clf = rule_clf or RuleClassifier()
        self.ml_clf = ml_clf or SimpleMLDetector(model_path=DEFAULT_MODEL_PATH, threshold=0.40)

    def classify(self, text: str) -> dict:
        if not text or not str(text).strip():
            return {
                "blocked": False,
                "score": 0.0,
                "rule_score": None,
                "rule_blocked": None,
                "ml_score": None,
                "ml_blocked": None,
                "triggered_groups": [],
                "decided_by": None,
            }

        r = self.rule_clf.classify(text)
        m = self.ml_clf.classify(text)
        blocked = bool(r.get("blocked") or m.get("blocked"))

        if r.get("blocked") and m.get("blocked"):
            decided_by = "both"
        elif r.get("blocked"):
            decided_by = "rule"
        elif m.get("blocked"):
            decided_by = "ml"
        else:
            decided_by = None

        return {
            "classifier": self.NAME,
            "blocked": blocked,
            "score": max(float(r.get("score") or 0.0), float(m.get("score") or 0.0)),
            "rule_score": r.get("score"),
            "rule_blocked": r.get("blocked"),
            "ml_score": m.get("score"),
            "ml_blocked": m.get("blocked"),
            "triggered_groups": r.get("triggered_groups") or [],
            "decided_by": decided_by,
        }


class S4AdaptiveRAG(BaselineRAG, BaseHelpers):
    """S4 Adaptive RAG system with retrieval-aware risk scoring and sentence-level sanitization."""

    SYSTEM_NAME = "S4"
    CLASSIFIER_NAME = RuleOrMLClassifier.NAME

    def __init__(self, db_path=config.CLEAN_DB_PATH, config_risk=None):
        super().__init__(db_path=db_path)
        self.rule_clf = RuleClassifier()
        self.ml_clf = SimpleMLDetector(model_path=DEFAULT_MODEL_PATH, threshold=0.40)
        self.classifier = RuleOrMLClassifier(self.rule_clf, self.ml_clf)
        self.scanner = SentenceWindowScanner(self.classifier, max_window_chars=420)
        self.risk_scorer = RetrievalRiskScorer(config_risk or S4RiskConfig())
        self.fast_path_count = 0
        self.total_queries = 0

    def _init_chroma(self):
        """Bind local dense retriever if on-disk HNSW is broken/absent."""
        return BaseHelpers._init_chroma(self)

    def retrieve_chunks(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """Retrieve chunks with metadata, including distance and doc/domain tags."""
        if top_k <= 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        ids = (results.get("ids") or [[]])[0] or []
        docs = (results.get("documents") or [[]])[0] or []
        metas = (results.get("metadatas") or [[]])[0] or []
        dists = (results.get("distances") or [[]])[0] or []

        chunks = []
        for i, text in enumerate(docs):
            chunk_id = ids[i] if i < len(ids) else f"retrieved_{i}"
            dist = float(dists[i]) if i < len(dists) and dists[i] is not None else None
            meta = metas[i] if i < len(metas) else {}
            doc_id = (meta or {}).get("doc_id")
            domain = (meta or {}).get("domain")
            chunks.append({
                "chunk_id": chunk_id,
                "text": text or "",
                "distance": dist,
                "doc_id": doc_id,
                "domain": domain,
                "injected": False,
            })
        return chunks

    def _audit_and_sanitize_chunks(self, query: str, chunks: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """Scan, score, and sanitize/quarantine chunks according to retrieval-aware risk."""
        allowed_chunks = []
        doc_clf_results = []
        s4_audit_log = []

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]

            # 1. Whole document classification using S3 composition
            doc_decision = self.classifier.classify(chunk_text)
            document_score = doc_decision["score"]

            # 2. Fine-grained sentence window scanning
            scan_res = self.scanner.scan(chunk_text)
            max_sentence_score = scan_res["max_sentence_score"]
            suspicious_window_proportion = scan_res["suspicious_window_proportion"]

            # 3. Query-document alignment calculation
            dist = chunk.get("distance")
            query_document_alignment = self.risk_scorer.alignment(
                query, chunk_text, retrieval_distance=dist
            )
            if query_document_alignment is None:
                # Attempt to extract embedding function from retriever collection
                ef = None
                if hasattr(self.collection, "embedding_function") and self.collection.embedding_function is not None:
                    ef = self.collection.embedding_function
                elif hasattr(self.collection, "_ef") and self.collection._ef is not None:
                    ef = self.collection._ef
                if ef is not None:
                    try:
                        v = ef([query, chunk_text])
                        query_document_alignment = cosine_similarity(v[0], v[1])
                    except Exception:
                        pass

            # 4. Retrieval-aware risk scoring
            risk_res = self.risk_scorer.score(
                document_score=document_score,
                max_sentence_score=max_sentence_score,
                suspicious_window_proportion=suspicious_window_proportion,
                query_document_alignment=query_document_alignment
            )
            risk_score = risk_res["risk_score"]
            risk_level = risk_res["risk_level"]  # SAFE, SUSPICIOUS, MALICIOUS

            # 5. Quarantine/sanitize logic.
            # Implemented behavior, which the October 2026 logs follow:
            # MALICIOUS quarantines the whole chunk. Any other chunk that
            # has at least one classifier-blocked window is sanitized, even
            # when risk_level is SAFE. The 0.45 suspicious threshold is
            # recorded in `policy` but does not gate sanitization.
            # On the frozen logs the alignment penalty was 0 for every
            # chunk, and the MALICIOUS set matched document_score >= 0.80.
            if risk_level == "MALICIOUS":
                quarantine_state = "quarantined"
                final_text = ""
                removed_windows = [w["window_text"] for w in scan_res["windows"]]
                flagged_text_reached_context = False
            else:
                if scan_res["suspicious_window_count"] > 0:
                    quarantine_state = "sanitized"
                    final_text = scan_res["clean_text"]
                    removed_windows = scan_res["flagged_text"]
                    flagged_text_reached_context = False
                else:
                    quarantine_state = "allowed"
                    final_text = chunk_text
                    removed_windows = []
                    flagged_text_reached_context = False

            audit_entry = {
                "rank": idx + 1,
                "distance": dist,
                "alignment": query_document_alignment,
                "document_score": document_score,
                "window_scores": [w["score"] for w in scan_res["windows"]],
                "window_counts": {
                    "total": scan_res["total_windows"],
                    "suspicious": scan_res["suspicious_window_count"]
                },
                "risk": risk_score,
                "policy": risk_level,
                "quarantine_state": quarantine_state,
                "removed_windows": removed_windows,
                "flagged_text_reached_context": flagged_text_reached_context,
                "chunk_id": chunk["chunk_id"],
                "injected": bool(chunk.get("injected")),
            }
            s4_audit_log.append(audit_entry)

            doc_clf_results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id"),
                "distance": dist,
                "injected": bool(chunk.get("injected")),
                "classifier": self.CLASSIFIER_NAME,
                "rule_score": doc_decision.get("rule_score"),
                "ml_score": doc_decision.get("ml_score"),
                "score": document_score,
                "blocked": quarantine_state == "quarantined",
                "allowed": quarantine_state != "quarantined",
                "reason": doc_decision.get("reason") or f"risk_{risk_level.lower()}",
                "layer": "document"
            })

            if quarantine_state != "quarantined" and final_text.strip():
                allowed_chunk = dict(chunk)
                allowed_chunk["text"] = final_text
                allowed_chunks.append(allowed_chunk)

        return allowed_chunks, doc_clf_results, s4_audit_log

    def query(self, question: str, top_k: int = config.TOP_K) -> dict:
        """Gated RAG query path with adaptive document auditing."""
        t_start = time.perf_counter()
        self.total_queries += 1
        original_query = question

        query_res = self.classifier.classify(question)
        if query_res["blocked"]:
            self.fast_path_count += 1
            res = self._blocked_response(
                t_start, "query", query_res, [], original_query,
                retrieval_called=False,
            )
            res["s4_audit"] = []
            return res

        chunks = self.retrieve_chunks(question, top_k=top_k)
        allowed, doc_clf_res, s4_audit_log = self._audit_and_sanitize_chunks(question, chunks)
        n_blocked = sum(1 for e in doc_clf_res if e.get("blocked"))

        if chunks and not allowed:
            res = self._no_context_response(
                t_start, "document", query_res, doc_clf_res, original_query,
                n_chunks_retrieved=len(chunks),
                n_chunks_blocked=n_blocked,
            )
            res["s4_audit"] = s4_audit_log
            return res

        res = self._generate_from_chunks(
            question, allowed, t_start, query_res, doc_clf_res, original_query,
        )
        res["s4_audit"] = s4_audit_log
        return res

    def query_with_injected_doc(self, trigger_query: str, poisoned_doc_text: str, top_k: int = config.TOP_K) -> dict:
        """Gated query path with injected document scanning and risk-aware sanitization."""
        t_start = time.perf_counter()
        original_query = trigger_query

        query_res = self.classifier.classify(trigger_query)
        if query_res["blocked"]:
            res = self._blocked_response(
                t_start, "query", query_res, [], original_query,
                retrieval_called=False,
                poisoned_doc_injected=True,
            )
            res["s4_audit"] = []
            return res

        clean_chunks = self.retrieve_chunks(trigger_query, top_k=top_k - 1)
        injected = {
            "chunk_id": "injected_poisoned_doc",
            "text": poisoned_doc_text,
            "distance": None,
            "doc_id": "injected_poisoned_doc",
            "domain": None,
            "injected": True,
        }
        all_chunks = [injected] + clean_chunks
        allowed, doc_clf_res, s4_audit_log = self._audit_and_sanitize_chunks(trigger_query, all_chunks)
        n_blocked = sum(1 for e in doc_clf_res if e.get("blocked"))

        if all_chunks and not allowed:
            res = self._no_context_response(
                t_start, "document", query_res, doc_clf_res, original_query,
                poisoned_doc_injected=True,
                n_chunks_retrieved=len(all_chunks),
                n_chunks_blocked=n_blocked,
            )
            res["s4_audit"] = s4_audit_log
            return res

        res = self._generate_from_chunks(
            trigger_query, allowed, t_start, query_res, doc_clf_res,
            original_query, poisoned_doc_injected=True,
        )
        res["s4_audit"] = s4_audit_log
        return res

    def query_multi_turn(self, turns: list[dict], attack_turn_index: int) -> dict:
        """Multi-turn defended path with first-turn audit propagation and second-turn gate."""
        t_start = time.perf_counter()

        t1_content = next(
            (t["content"] for t in turns if t["role"] == "user"), None
        )
        if t1_content is None:
            res = self._result(
                t_start,
                error="No user turn found",
                status="error",
                original_query=None,
            )
            res["s4_audit"] = []
            return res

        t1_result = self.query(t1_content)
        t1_response = t1_result.get("response", "")
        t1_prompt = t1_result.get("prompt") or t1_content
        t1_llm_called = bool(t1_result.get("llm_called"))
        t1_docs = t1_result.get("context_docs") or []
        t1_scan = t1_result.get("doc_clf_results") or []
        t1_query_res = t1_result.get("query_clf_result")
        t1_audit = t1_result.get("s4_audit") or []

        if t1_result.get("status") in ("BLOCKED", "NO_CLEAN_CONTEXT"):
            return t1_result

        attack_content = turns[attack_turn_index]["content"]
        followup_res = self.classifier.classify(attack_content)

        if followup_res["blocked"]:
            res = self._blocked_response(
                t_start, "followup", t1_query_res, t1_scan,
                original_query=attack_content,
                followup_inspected=True,
                followup_clf_result=followup_res,
                t1_response=t1_response,
                t1_llm_called=t1_llm_called,
            )
            res["s4_audit"] = t1_audit
            return res

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "user", "content": t1_prompt},
                    {"role": "assistant", "content": t1_response},
                    {"role": "user", "content": attack_content},
                ],
                system_prompt=config.RAG_SYSTEM_PROMPT,
            )
            error = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except Exception as e:
            response = ""
            error = str(e)
            safety_blocked = False

        res = self._result(
            t_start,
            response=response,
            context_docs=t1_docs,
            prompt=t1_prompt,
            error=error,
            safety_blocked=safety_blocked,
            query_clf_result=t1_query_res,
            doc_clf_results=t1_scan,
            original_query=attack_content,
            retrieval_called=t1_result.get("retrieval_called", True),
            llm_called=True,
            followup_inspected=True,
            followup_clf_result=followup_res,
            t1_response=t1_response,
            t1_llm_called=t1_llm_called,
            n_chunks_retrieved=t1_result.get("n_chunks_retrieved", 0),
            n_chunks_blocked=t1_result.get("n_chunks_blocked", 0),
            n_chunks_allowed=t1_result.get("n_chunks_allowed", 0),
        )
        res["s4_audit"] = t1_audit
        return res

    @property
    def fast_path_rate(self):
        if self.total_queries == 0:
            return 0.0
        return self.fast_path_count / self.total_queries
