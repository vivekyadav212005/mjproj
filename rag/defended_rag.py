"""
S1 / S2 / S3 defended RAG systems.

LLM used for generation: Ollama `llama3:latest` (via rag.llm_client).
These systems do NOT use DeBERTa. S2 and S3 use SimpleMLDetector
(TF-IDF + logistic regression). S1 uses RuleClassifier only.

Inheritance note
----------------
Each defended class is declared as ``(BaselineRAG, BaseHelpers)``.
Python MRO therefore prefers BaselineRAG for any method defined on
both bases. BaselineRAG.query_with_injected_doc and
BaselineRAG.query_multi_turn are the *undefended* S0 implementations.

S0 is intentionally left unchanged. To keep I1/I2/I3 and D5 on the
defended path, each subclass *explicitly* routes those methods to
BaseHelpers rather than relying on mixin order.

Pipeline (S1, S2, S3 — identical structure, different classifier)
-----------------------------------------------------------------
USER QUERY
  -> query security check
  -> if blocked: structured BLOCK, no retrieval, no LLM
  -> retrieval (top-k)
  -> scan each retrieved chunk with the system classifier
  -> quarantine blocked chunks (never added to the prompt)
  -> if no clean chunks remain: NO_CLEAN_CONTEXT, no LLM
  -> otherwise llama3:latest answers from remaining chunks only
"""

import os
import time

import config
from rag.baseline_rag import BaselineRAG
from rag.llm_client import LLMError, SAFETY_BLOCKED
from security.rule_classifier import RuleClassifier
from security.simple_ml_detector import SimpleMLDetector

BLOCKED_RESPONSE = (
    "I'm sorry, I cannot process this request as it appears to contain "
    "instructions that may attempt to override my operational guidelines."
)

NO_CLEAN_CONTEXT = (
    "I was unable to retrieve relevant information to answer your question."
)

# S3 hybrid weights / thresholds. Query block uses 0.75.
# Document scan uses a lower threshold (0.40) so poisoned chunks are
# filtered more aggressively than user queries. S3_HARD_BLOCK_THRESHOLD
# is retained for documentation only; runtime query threshold is 0.75.
S3_HARD_BLOCK_THRESHOLD = 0.85
S3_QUERY_BLOCK_THRESHOLD = 0.75
S3_DOC_BLOCK_THRESHOLD = 0.40
S3_RULE_EARLY_EXIT = 0.95
S3_RULE_WEIGHT = 0.35
S3_ML_WEIGHT = 0.65

INJECTED_CHUNK_ID = "injected_poisoned_doc"


def _empty_decision(classifier_name):
    return {
        "classifier": classifier_name,
        "blocked": False,
        "reason": "empty_text",
        "score": 0.0,
        "rule_score": None,
        "ml_score": None,
        "triggered_groups": [],
        "layer": None,
    }


class BaseHelpers:
    """
    Shared defended execution path for S1/S2/S3.

    Subclasses must implement ``_security_check(text, layer=...)`` and
    set ``CLASSIFIER_NAME``. Do not call these methods via MRO fallback;
    subclasses must override query / query_with_injected_doc /
    query_multi_turn and route here explicitly.
    """

    CLASSIFIER_NAME = "unspecified"

    def _security_check(self, text, layer="query"):
        raise NotImplementedError

    def _init_chroma(self):
        """
        Defended systems only.

        S0's BaselineRAG._init_chroma uses SentenceTransformer + Chroma
        HNSW. That path is left unchanged. On this machine the HNSW
        vector file is missing, so defended classes explicitly bind
        this method instead of inheriting S0's client.

        Retrieval uses the same sqlite document store plus ONNX
        all-MiniLM-L6-v2 (see rag/local_retriever.py).
        """
        from rag.local_retriever import LocalDenseRetriever
        self._local_retriever = LocalDenseRetriever(self.db_path)
        self.collection = self._local_retriever

    def retrieve_chunks(self, query, top_k=config.TOP_K):
        """
        Retrieve top-k chunks with IDs.

        This is a defended-path helper. It does not modify
        BaselineRAG.retrieve() (the S0 method).
        """
        if top_k <= 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents"],
        )
        ids = (results.get("ids") or [[]])[0] or []
        docs = (results.get("documents") or [[]])[0] or []
        chunks = []
        for i, text in enumerate(docs):
            chunk_id = ids[i] if i < len(ids) else f"retrieved_{i}"
            chunks.append({"chunk_id": chunk_id, "text": text})
        return chunks

    def _scan_chunks(self, chunks):
        """Classify every chunk. Blocked chunks are quarantined."""
        allowed = []
        scan_log = []
        for chunk in chunks:
            decision = self._security_check(chunk["text"], layer="document")
            entry = {
                "chunk_id": chunk["chunk_id"],
                "classifier": decision.get("classifier", self.CLASSIFIER_NAME),
                "rule_score": decision.get("rule_score"),
                "ml_score": decision.get("ml_score"),
                "score": decision.get("score"),
                "blocked": bool(decision.get("blocked")),
                "allowed": not bool(decision.get("blocked")),
                "reason": decision.get("reason"),
                "layer": "document",
            }
            scan_log.append(entry)
            if not entry["blocked"]:
                allowed.append(chunk)
        return allowed, scan_log

    def _elapsed_ms(self, t_start):
        return round((time.perf_counter() - t_start) * 1000, 2)

    def _result(
        self,
        t_start,
        *,
        response="",
        context_docs=None,
        prompt="",
        error=None,
        safety_blocked=False,
        blocked_by=None,
        blocked_at_layer=None,
        status="ok",
        query_clf_result=None,
        doc_clf_results=None,
        original_query=None,
        retrieval_called=False,
        llm_called=False,
        poisoned_doc_injected=False,
        followup_inspected=False,
        followup_clf_result=None,
        t1_response="",
        t1_llm_called=False,
        n_chunks_retrieved=0,
        n_chunks_blocked=0,
        n_chunks_allowed=0,
    ):
        return {
            "response": response,
            "context_docs": context_docs or [],
            "prompt": prompt,
            "latency_ms": self._elapsed_ms(t_start),
            "error": error,
            "safety_blocked": safety_blocked,
            "blocked_by": blocked_by,
            "blocked_at_layer": blocked_at_layer,
            "status": status,
            "query_clf_result": query_clf_result,
            "doc_clf_results": doc_clf_results or [],
            "original_query": original_query,
            "substituted_query": None,
            "retrieval_called": retrieval_called,
            "llm_called": llm_called,
            "poisoned_doc_injected": poisoned_doc_injected,
            "followup_inspected": followup_inspected,
            "followup_clf_result": followup_clf_result,
            "t1_response": t1_response,
            "t1_llm_called": t1_llm_called,
            "classifier": self.CLASSIFIER_NAME,
            "n_chunks_retrieved": n_chunks_retrieved,
            "n_chunks_blocked": n_chunks_blocked,
            "n_chunks_allowed": n_chunks_allowed,
        }

    def _blocked_response(self, t_start, blocked_by, query_res, doc_res,
                          original_query, *, followup_inspected=False,
                          followup_clf_result=None, t1_response="",
                          t1_llm_called=False, poisoned_doc_injected=False,
                          retrieval_called=False, n_chunks_retrieved=0,
                          n_chunks_blocked=0, n_chunks_allowed=0):
        return self._result(
            t_start,
            response=BLOCKED_RESPONSE,
            status="BLOCKED",
            blocked_by=blocked_by,
            blocked_at_layer=blocked_by,
            query_clf_result=query_res,
            doc_clf_results=doc_res,
            original_query=original_query,
            retrieval_called=retrieval_called,
            llm_called=False,
            followup_inspected=followup_inspected,
            followup_clf_result=followup_clf_result,
            t1_response=t1_response,
            t1_llm_called=t1_llm_called,
            poisoned_doc_injected=poisoned_doc_injected,
            n_chunks_retrieved=n_chunks_retrieved,
            n_chunks_blocked=n_chunks_blocked,
            n_chunks_allowed=n_chunks_allowed,
        )

    def _no_context_response(self, t_start, blocked_by, query_res, doc_res,
                             original_query, *, poisoned_doc_injected=False,
                             n_chunks_retrieved=0, n_chunks_blocked=0):
        return self._result(
            t_start,
            response=NO_CLEAN_CONTEXT,
            status="NO_CLEAN_CONTEXT",
            blocked_by=blocked_by,
            blocked_at_layer=blocked_by,
            query_clf_result=query_res,
            doc_clf_results=doc_res,
            original_query=original_query,
            retrieval_called=True,
            llm_called=False,
            poisoned_doc_injected=poisoned_doc_injected,
            n_chunks_retrieved=n_chunks_retrieved,
            n_chunks_blocked=n_chunks_blocked,
            n_chunks_allowed=0,
        )

    def _generate_from_chunks(self, question, allowed_chunks, t_start,
                              query_res, doc_res, original_query, *,
                              poisoned_doc_injected=False):
        """
        Build context from *allowed* chunks only and call llama3:latest.

        Uses the same S0 prompt template so S1/S2/S3 differ by classifier
        filtering, not by a second hidden prompt-hardening layer.
        """
        context_docs = [c["text"] for c in allowed_chunks]
        prompt = self.build_prompt(question, context_docs)
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=config.RAG_SYSTEM_PROMPT,
            )
            error = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response = ""
            error = str(e)
            safety_blocked = False

        n_blocked = sum(1 for e in doc_res if e.get("blocked"))
        return self._result(
            t_start,
            response=response,
            context_docs=context_docs,
            prompt=prompt,
            error=error,
            safety_blocked=safety_blocked,
            query_clf_result=query_res,
            doc_clf_results=doc_res,
            original_query=original_query,
            retrieval_called=True,
            llm_called=True,
            poisoned_doc_injected=poisoned_doc_injected,
            n_chunks_retrieved=len(doc_res),
            n_chunks_blocked=n_blocked,
            n_chunks_allowed=len(allowed_chunks),
        )

    def defended_query(self, question, top_k=config.TOP_K):
        """Direct query path with classifier gate + document scan."""
        t_start = time.perf_counter()
        original_query = question

        query_res = self._security_check(question, layer="query")
        if query_res["blocked"]:
            return self._blocked_response(
                t_start, "query", query_res, [], original_query,
                retrieval_called=False,
            )

        chunks = self.retrieve_chunks(question, top_k=top_k)
        allowed, scan_log = self._scan_chunks(chunks)
        n_blocked = sum(1 for e in scan_log if e.get("blocked"))

        if scan_log and not allowed:
            return self._no_context_response(
                t_start, "document", query_res, scan_log, original_query,
                n_chunks_retrieved=len(scan_log),
                n_chunks_blocked=n_blocked,
            )

        return self._generate_from_chunks(
            question, allowed, t_start, query_res, scan_log, original_query,
        )

    def query_with_injected_doc(self, trigger_query, poisoned_doc_text,
                                top_k=config.TOP_K):
        """
        I1/I2/I3 defended path.

        Trigger query is classified first. Retrieved clean chunks and the
        injected poisoned document are all scanned. A blocked chunk is
        never included in the LLM prompt.
        """
        t_start = time.perf_counter()
        original_query = trigger_query

        query_res = self._security_check(trigger_query, layer="query")
        if query_res["blocked"]:
            return self._blocked_response(
                t_start, "query", query_res, [], original_query,
                retrieval_called=False,
                poisoned_doc_injected=True,
            )

        clean_chunks = self.retrieve_chunks(trigger_query, top_k=top_k - 1)
        injected = {"chunk_id": INJECTED_CHUNK_ID, "text": poisoned_doc_text}
        all_chunks = [injected] + clean_chunks
        allowed, scan_log = self._scan_chunks(all_chunks)
        n_blocked = sum(1 for e in scan_log if e.get("blocked"))

        if scan_log and not allowed:
            return self._no_context_response(
                t_start, "document", query_res, scan_log, original_query,
                poisoned_doc_injected=True,
                n_chunks_retrieved=len(scan_log),
                n_chunks_blocked=n_blocked,
            )

        return self._generate_from_chunks(
            trigger_query, allowed, t_start, query_res, scan_log,
            original_query, poisoned_doc_injected=True,
        )

    def query_multi_turn(self, turns, attack_turn_index):
        """
        D5 defended path.

        Turn 1 (benign anchor) goes through the defended query path.
        The attack follow-up is classified with the system classifier
        *before* it is sent to the LLM. A blocked follow-up never
        reaches llama3. Conversation history is preserved; the follow-up
        text is never substituted.
        """
        t_start = time.perf_counter()

        t1_content = next(
            (t["content"] for t in turns if t["role"] == "user"), None
        )
        if t1_content is None:
            return self._result(
                t_start,
                error="No user turn found",
                status="error",
                original_query=None,
            )

        t1_result = self.query(t1_content)
        t1_response = t1_result.get("response", "")
        t1_prompt = t1_result.get("prompt") or t1_content
        t1_llm_called = bool(t1_result.get("llm_called"))
        t1_docs = t1_result.get("context_docs") or []
        t1_scan = t1_result.get("doc_clf_results") or []
        t1_query_res = t1_result.get("query_clf_result")

        attack_content = turns[attack_turn_index]["content"]
        followup_res = self._security_check(attack_content, layer="followup")

        if followup_res["blocked"]:
            return self._blocked_response(
                t_start, "followup", t1_query_res, t1_scan,
                original_query=attack_content,
                followup_inspected=True,
                followup_clf_result=followup_res,
                t1_response=t1_response,
                t1_llm_called=t1_llm_called,
                retrieval_called=bool(t1_result.get("retrieval_called")),
                n_chunks_retrieved=t1_result.get("n_chunks_retrieved", 0),
                n_chunks_blocked=t1_result.get("n_chunks_blocked", 0),
                n_chunks_allowed=t1_result.get("n_chunks_allowed", 0),
            )

        final_messages = [
            {"role": "user", "content": t1_prompt},
            {"role": "assistant", "content": t1_response},
            {"role": "user", "content": attack_content},
        ]
        try:
            t2_response = self.llm.chat(
                messages=final_messages,
                system_prompt=config.RAG_SYSTEM_PROMPT,
            )
            error = None
            safety_blocked = (t2_response == SAFETY_BLOCKED)
        except LLMError as e:
            t2_response = ""
            error = str(e)
            safety_blocked = False

        return self._result(
            t_start,
            response=t2_response,
            context_docs=t1_docs,
            prompt=t1_prompt,
            error=error,
            safety_blocked=safety_blocked,
            query_clf_result=t1_query_res,
            doc_clf_results=t1_scan,
            original_query=attack_content,
            retrieval_called=bool(t1_result.get("retrieval_called")),
            llm_called=True,
            followup_inspected=True,
            followup_clf_result=followup_res,
            t1_response=t1_response,
            t1_llm_called=t1_llm_called,
            n_chunks_retrieved=t1_result.get("n_chunks_retrieved", 0),
            n_chunks_blocked=t1_result.get("n_chunks_blocked", 0),
            n_chunks_allowed=t1_result.get("n_chunks_allowed", 0),
        )


# =========================
# S1 — RuleClassifier only
# =========================
class RuleDefendedRAG(BaselineRAG, BaseHelpers):
    """
    S1 — rule-based defence.

    Classifier: ``security.rule_classifier.RuleClassifier`` (G1–G5).
    No SimpleMLDetector. No DeBERTa.

    Direct queries, retrieved/poisoned documents, and D5 follow-ups are
    all scored by RuleClassifier. Block if score >= 0.5.
    """

    SYSTEM_NAME = "S1"
    CLASSIFIER_NAME = "RuleClassifier"

    def __init__(self, db_path=config.CLEAN_DB_PATH):
        super().__init__(db_path=db_path)
        self.rule_clf = RuleClassifier()

    def _init_chroma(self):
        """Explicit route: do not use S0 SentenceTransformer/HNSW client."""
        return BaseHelpers._init_chroma(self)

    def _security_check(self, text, layer="query"):
        if not text or not str(text).strip():
            decision = _empty_decision(self.CLASSIFIER_NAME)
            decision["layer"] = layer
            return decision
        raw = self.rule_clf.classify(text)
        groups = raw.get("triggered_groups") or []
        blocked = bool(raw.get("blocked"))
        if blocked:
            reason = (
                f"RuleClassifier blocked at {layer}: "
                f"groups={groups} score={raw.get('score')}"
            )
        else:
            reason = (
                f"RuleClassifier allowed at {layer}: "
                f"score={raw.get('score')}"
            )
        return {
            "classifier": self.CLASSIFIER_NAME,
            "blocked": blocked,
            "reason": reason,
            "score": raw.get("score", 0.0),
            "rule_score": raw.get("score", 0.0),
            "ml_score": None,
            "triggered_groups": groups,
            "layer": layer,
        }

    def query(self, question, top_k=config.TOP_K):
        """Direct D1–D4 path — RuleClassifier, not S0.query."""
        return BaseHelpers.defended_query(self, question, top_k)

    def query_with_injected_doc(self, trigger_query, poisoned_doc_text,
                                top_k=config.TOP_K):
        """I1/I2/I3 — explicit BaseHelpers route (BaselineRAG wins MRO)."""
        return BaseHelpers.query_with_injected_doc(
            self, trigger_query, poisoned_doc_text, top_k
        )

    def query_multi_turn(self, turns, attack_turn_index):
        """D5 — explicit BaseHelpers route (BaselineRAG wins MRO)."""
        return BaseHelpers.query_multi_turn(self, turns, attack_turn_index)


# =========================
# S2 — SimpleMLDetector only
# =========================
class MLDefendedRAG(BaselineRAG, BaseHelpers):
    """
    S2 — TF-IDF + logistic regression defence.

    Classifier: ``security.simple_ml_detector.SimpleMLDetector``.
    This is NOT DeBERTa and NOT RuleClassifier.

    The detector is a persisted TF-IDF + logistic regression model
    trained on query- and document-shaped prompt-injection data
    (``security/training/s2_tfidf_lr.joblib``). It is NOT trained on
    S0 attack responses and is NOT DeBERTa or RuleClassifier.

    Direct queries, retrieved / poisoned documents, and D5 follow-ups
    are scored by SimpleMLDetector. Block if probability >= the
    validation-calibrated threshold stored in the model bundle.
    """

    SYSTEM_NAME = "S2"
    CLASSIFIER_NAME = "SimpleMLDetector"
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "security", "training", "s2_tfidf_lr.joblib"
    )

    def __init__(self, db_path=config.CLEAN_DB_PATH):
        super().__init__(db_path=db_path)
        model_path = os.path.abspath(self.MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"S2 model not found at {model_path}. "
                "Run security/training/prepare_s2_training_data.py then "
                "security/training/train_simple_ml.py"
            )
        self.ml_clf = SimpleMLDetector(model_path=model_path)

    def _init_chroma(self):
        """Explicit route: do not use S0 SentenceTransformer/HNSW client."""
        return BaseHelpers._init_chroma(self)

    def _security_check(self, text, layer="query"):
        if not text or not str(text).strip():
            decision = _empty_decision(self.CLASSIFIER_NAME)
            decision["layer"] = layer
            return decision
        raw = self.ml_clf.classify(text)
        blocked = bool(raw.get("blocked"))
        score = raw.get("score", 0.0)
        threshold = getattr(self.ml_clf, "threshold", None)
        if threshold is None:
            threshold = 0.4
        if blocked:
            reason = (
                f"SimpleMLDetector blocked at {layer}: "
                f"ml_score={score} threshold={threshold}"
            )
        else:
            reason = (
                f"SimpleMLDetector allowed at {layer}: "
                f"ml_score={score}"
            )
        return {
            "classifier": self.CLASSIFIER_NAME,
            "blocked": blocked,
            "reason": reason,
            "score": score,
            "rule_score": None,
            "ml_score": score,
            "triggered_groups": [],
            "layer": layer,
        }

    def query(self, question, top_k=config.TOP_K):
        """Direct D1–D4 path — SimpleMLDetector, not S0.query."""
        return BaseHelpers.defended_query(self, question, top_k)

    def query_with_injected_doc(self, trigger_query, poisoned_doc_text,
                                top_k=config.TOP_K):
        """I1/I2/I3 — explicit BaseHelpers route (BaselineRAG wins MRO)."""
        return BaseHelpers.query_with_injected_doc(
            self, trigger_query, poisoned_doc_text, top_k
        )

    def query_multi_turn(self, turns, attack_turn_index):
        """D5 — explicit BaseHelpers route (BaselineRAG wins MRO)."""
        return BaseHelpers.query_multi_turn(self, turns, attack_turn_index)


# =========================
# S3 — Rule + SimpleMLDetector hybrid
# =========================
class HybridDefendedRAG(BaselineRAG, BaseHelpers):
    """
    S3 — hybrid RuleClassifier + SimpleMLDetector.

    This is NOT DeBERTa. Scoring:

        if rule_score >= 0.95: block (rule fast path)
        else: 0.35 * rule_score + 0.65 * ml_score
        query / follow-up block if combined >= 0.75
        document block if combined >= 0.40

    A blocked query returns immediately: no retrieval, no LLM, and the
    original query is preserved in logs. There is no query substitution.
    """

    SYSTEM_NAME = "S3"
    CLASSIFIER_NAME = "hybrid_Rule_SimpleML"

    def __init__(self, db_path=config.CLEAN_DB_PATH):
        super().__init__(db_path=db_path)
        self.rule_clf = RuleClassifier()
        self.ml_clf = SimpleMLDetector()
        self.ml_clf.train("results/s0/S0_attack_results.jsonl")
        self.fast_path_count = 0
        self.total_queries = 0

    def _init_chroma(self):
        """Explicit route: do not use S0 SentenceTransformer/HNSW client."""
        return BaseHelpers._init_chroma(self)

    def _hybrid_classify(self, text, layer="query"):
        """Weighted hybrid used by S3 for queries, documents, and follow-ups."""
        r = self.rule_clf.classify(text)
        r_score = r.get("score", 0.0)
        groups = r.get("triggered_groups") or []

        if r_score >= S3_RULE_EARLY_EXIT:
            self.fast_path_count += 1
            return {
                "blocked": True,
                "score": r_score,
                "type": "rule",
                "rule_score": r_score,
                "ml_score": None,
                "triggered_groups": groups,
                "reason": (
                    f"S3 rule fast-path at {layer}: groups={groups} "
                    f"rule_score={r_score} >= {S3_RULE_EARLY_EXIT}"
                ),
            }

        m = self.ml_clf.classify(text)
        m_score = m.get("score", 0.0)
        combined = S3_RULE_WEIGHT * r_score + S3_ML_WEIGHT * m_score
        threshold = (
            S3_DOC_BLOCK_THRESHOLD if layer == "document"
            else S3_QUERY_BLOCK_THRESHOLD
        )
        blocked = combined >= threshold
        if blocked:
            reason = (
                f"S3 hybrid blocked at {layer}: combined={round(combined, 4)} "
                f">= {threshold} (rule={r_score}, ml={m_score})"
            )
        else:
            reason = (
                f"S3 hybrid allowed at {layer}: combined={round(combined, 4)} "
                f"< {threshold} (rule={r_score}, ml={m_score})"
            )
        return {
            "blocked": blocked,
            "score": round(combined, 4),
            "type": "hybrid",
            "rule_score": r_score,
            "ml_score": m_score,
            "triggered_groups": groups,
            "reason": reason,
        }

    def _security_check(self, text, layer="query"):
        if layer == "query":
            self.total_queries += 1
        if not text or not str(text).strip():
            decision = _empty_decision(self.CLASSIFIER_NAME)
            decision["layer"] = layer
            return decision
        raw = self._hybrid_classify(text, layer=layer)
        return {
            "classifier": self.CLASSIFIER_NAME,
            "blocked": bool(raw.get("blocked")),
            "reason": raw.get("reason"),
            "score": raw.get("score", 0.0),
            "rule_score": raw.get("rule_score"),
            "ml_score": raw.get("ml_score"),
            "triggered_groups": raw.get("triggered_groups") or [],
            "hybrid_type": raw.get("type"),
            "layer": layer,
        }

    def query(self, question, top_k=config.TOP_K):
        """
        Direct D1–D4 path.

        If the query is blocked the original string is logged and
        retrieval/LLM are skipped. The query is never replaced with
        another string.
        """
        return BaseHelpers.defended_query(self, question, top_k)

    def query_with_injected_doc(self, trigger_query, poisoned_doc_text,
                                top_k=config.TOP_K):
        """I1/I2/I3 — explicit BaseHelpers route (BaselineRAG wins MRO)."""
        return BaseHelpers.query_with_injected_doc(
            self, trigger_query, poisoned_doc_text, top_k
        )

    def query_multi_turn(self, turns, attack_turn_index):
        """D5 — explicit BaseHelpers route (BaselineRAG wins MRO)."""
        return BaseHelpers.query_multi_turn(self, turns, attack_turn_index)

    @property
    def fast_path_rate(self):
        if self.total_queries == 0:
            return 0.0
        return self.fast_path_count / self.total_queries
