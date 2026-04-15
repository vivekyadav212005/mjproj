# rag/defended_rag.py
"""
Defended RAG systems: S1, S2, S3.

All three inherit from BaselineRAG and override the query pipeline
to add classification-based defence. The base RAG architecture
(ChromaDB, embedding model, LLM, prompt template) is identical
across S0–S3, ensuring that observed differences in attack success
and answer quality are attributable solely to the defence strategy.

Defence architecture overview:

  S1 (RuleDefendedRAG):
    - Apply RuleClassifier to user query
    - Apply RuleClassifier to each retrieved document chunk
    - Block if either triggers above threshold
    - Latency overhead: ~0ms (compiled regex)

  S2 (MLDefendedRAG):
    - Apply MLClassifier to user query
    - Apply MLClassifier to each retrieved document chunk
    - Block if either triggers above threshold
    - Latency overhead: ~50-200ms (transformer inference)

  S3 (HybridDefendedRAG):
    - Sequential gate: if rule_score >= HARD_BLOCK_THRESHOLD (0.85),
      block immediately without invoking ML (fast path)
    - Otherwise: fuse scores → combined = 0.35*rule + 0.65*ml
    - Block if combined >= 0.5
    - Latency overhead: between S1 and S2 (ML skipped on fast-path)

Blocked response format:
    All three systems return the same BLOCKED_RESPONSE string when
    an injection is detected. This is intentional — it prevents the
    attacker from learning which defence triggered.

Document scanning:
    All three systems apply their classifier to each retrieved chunk
    individually before context assembly. Chunks that score above
    threshold are EXCLUDED from the context. If all chunks are excluded,
    a fallback message is returned. This document-level scanning is the
    key design decision that enables indirect injection detection —
    input-only classifiers applied only to the user query would miss
    poisoned documents entirely.
"""

import time
from rag.baseline_rag import BaselineRAG
from security.rule_classifier import RuleClassifier
from security.ml_classifier import MLClassifier
import config
from rag.llm_client import LLMError, SAFETY_BLOCKED

# Returned when an injection is detected in the user query
BLOCKED_RESPONSE = (
    "I'm sorry, I cannot process this request as it appears to contain "
    "instructions that may attempt to override my operational guidelines."
)

# Returned when all retrieved documents were flagged as injections
NO_CLEAN_CONTEXT = (
    "I was unable to retrieve relevant information to answer your question."
)

# Hard-block threshold for S3 fast path (rule score >= this → block without ML)
S3_HARD_BLOCK_THRESHOLD = 0.85

# Fusion weights for S3
S3_RULE_WEIGHT = 0.35
S3_ML_WEIGHT   = 0.65


class RuleDefendedRAG(BaselineRAG):
    """
    S1: Rule-based defence.

    The rule classifier is applied at two layers:
      Layer 1 — Input layer: user query classified before retrieval.
               If blocked, returns BLOCKED_RESPONSE immediately.
      Layer 2 — Document layer: each retrieved chunk classified.
               Chunks scoring >= threshold are excluded from context.

    This two-layer application is the key S1 contribution for the paper:
    it extends rule-based detection beyond the input layer to the
    retrieved document content, partially addressing indirect injection.
    """

    SYSTEM_NAME = "S1"

    def __init__(self, db_path: str = config.CLEAN_DB_PATH):
        super().__init__(db_path=db_path)
        self.rule_clf = RuleClassifier()

    def query(self, question: str,
              top_k: int = config.TOP_K) -> dict:
        t_start = time.perf_counter()

        # Layer 1: classify user query
        query_result = self.rule_clf.classify(question)
        if query_result["blocked"]:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response":         BLOCKED_RESPONSE,
                "context_docs":     [],
                "prompt":           "",
                "latency_ms":       round(latency_ms, 2),
                "error":            None,
                "safety_blocked":   False,
                "blocked_by":       "rule_query",
                "query_clf_result": query_result,
                "doc_clf_results":  [],
            }

        # Retrieve documents
        raw_docs = self.retrieve(question, top_k=top_k)

        # Layer 2: classify each retrieved document chunk
        clean_docs   = []
        doc_results  = []
        for doc in raw_docs:
            doc_result = self.rule_clf.classify(doc)
            doc_results.append(doc_result)
            if not doc_result["blocked"]:
                clean_docs.append(doc)

        # Build prompt from clean docs only
        if not clean_docs:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response":         NO_CLEAN_CONTEXT,
                "context_docs":     [],
                "prompt":           "",
                "latency_ms":       round(latency_ms, 2),
                "error":            None,
                "safety_blocked":   False,
                "blocked_by":       "rule_all_docs",
                "query_clf_result": query_result,
                "doc_clf_results":  doc_results,
            }

        prompt = self.build_prompt(question, clean_docs)
        try:
            response = self.llm.chat(
                messages      = [{"role": "user", "content": prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error          = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response       = ""
            error          = str(e)
            safety_blocked = False

        latency_ms = (time.perf_counter() - t_start) * 1000
        return {
            "response":         response,
            "context_docs":     clean_docs,
            "prompt":           prompt,
            "latency_ms":       round(latency_ms, 2),
            "error":            error,
            "safety_blocked":   safety_blocked,
            "blocked_by":       None,
            "query_clf_result": query_result,
            "doc_clf_results":  doc_results,
        }

    def query_with_injected_doc(self,
                                trigger_query: str,
                                poisoned_doc_text: str,
                                top_k: int = config.TOP_K) -> dict:
        """Indirect attack: classify injected doc before it enters context."""
        t_start = time.perf_counter()

        query_result = self.rule_clf.classify(trigger_query)
        if query_result["blocked"]:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": BLOCKED_RESPONSE, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "rule_query",
                "poisoned_doc_injected": True,
            }

        # Classify the poisoned doc
        poison_result = self.rule_clf.classify(poisoned_doc_text)
        clean_docs    = self.retrieve(trigger_query, top_k=top_k - 1)

        if poison_result["blocked"]:
            # Poisoned doc caught — build context from clean docs only
            context_docs = clean_docs
        else:
            context_docs = [poisoned_doc_text] + clean_docs

        if not context_docs:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": NO_CLEAN_CONTEXT, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "rule_all_docs",
                "poisoned_doc_injected": True,
            }

        prompt = self.build_prompt(trigger_query, context_docs)
        try:
            response       = self.llm.chat(
                messages      = [{"role": "user", "content": prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error          = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response       = ""
            error          = str(e)
            safety_blocked = False

        latency_ms = (time.perf_counter() - t_start) * 1000
        return {
            "response":              response,
            "context_docs":          context_docs,
            "prompt":                prompt,
            "latency_ms":            round(latency_ms, 2),
            "error":                 error,
            "safety_blocked":        safety_blocked,
            "blocked_by":            None,
            "poisoned_doc_blocked":  poison_result["blocked"],
            "poisoned_doc_injected": True,
        }

    def query_multi_turn(self, turns, attack_turn_index):
        """Multi-turn: classify each user turn before processing."""
        for turn in turns:
            if turn["role"] == "user" and turn["content"] != "__PLACEHOLDER__":
                result = self.rule_clf.classify(turn["content"])
                if result["blocked"]:
                    return {
                        "response": BLOCKED_RESPONSE, "context_docs": [],
                        "prompt": "", "latency_ms": 0,
                        "error": None, "safety_blocked": False,
                        "blocked_by": "rule_query", "t1_response": "",
                    }
        return super().query_multi_turn(turns, attack_turn_index)


class MLDefendedRAG(BaselineRAG):
    """
    S2: ML-based defence (DeBERTa).

    Identical two-layer architecture to S1 but uses the ML classifier.
    Applied to user query and each retrieved document chunk.
    """

    SYSTEM_NAME = "S2"

    def __init__(self, db_path: str = config.CLEAN_DB_PATH):
        super().__init__(db_path=db_path)
        self.ml_clf = MLClassifier()

    def query(self, question: str,
              top_k: int = config.TOP_K) -> dict:
        t_start = time.perf_counter()

        query_result = self.ml_clf.classify(question)
        if query_result["blocked"]:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": BLOCKED_RESPONSE, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "ml_query",
                "query_clf_result": query_result, "doc_clf_results": [],
            }

        raw_docs    = self.retrieve(question, top_k=top_k)
        clean_docs  = []
        doc_results = []
        for doc in raw_docs:
            doc_result = self.ml_clf.classify(doc)
            doc_results.append(doc_result)
            if not doc_result["blocked"]:
                clean_docs.append(doc)

        if not clean_docs:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": NO_CLEAN_CONTEXT, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "ml_all_docs",
                "query_clf_result": query_result,
                "doc_clf_results": doc_results,
            }

        prompt = self.build_prompt(question, clean_docs)
        try:
            response       = self.llm.chat(
                messages      = [{"role": "user", "content": prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error          = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response       = ""
            error          = str(e)
            safety_blocked = False

        latency_ms = (time.perf_counter() - t_start) * 1000
        return {
            "response": response, "context_docs": clean_docs,
            "prompt": prompt, "latency_ms": round(latency_ms, 2),
            "error": error, "safety_blocked": safety_blocked,
            "blocked_by": None, "query_clf_result": query_result,
            "doc_clf_results": doc_results,
        }

    def query_with_injected_doc(self,
                                trigger_query: str,
                                poisoned_doc_text: str,
                                top_k: int = config.TOP_K) -> dict:
        t_start      = time.perf_counter()
        query_result = self.ml_clf.classify(trigger_query)
        if query_result["blocked"]:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": BLOCKED_RESPONSE, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "ml_query", "poisoned_doc_injected": True,
            }

        poison_result = self.ml_clf.classify(poisoned_doc_text)
        clean_docs    = self.retrieve(trigger_query, top_k=top_k - 1)

        context_docs  = (clean_docs if poison_result["blocked"]
                         else [poisoned_doc_text] + clean_docs)

        if not context_docs:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": NO_CLEAN_CONTEXT, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "ml_all_docs", "poisoned_doc_injected": True,
            }

        prompt = self.build_prompt(trigger_query, context_docs)
        try:
            response       = self.llm.chat(
                messages      = [{"role": "user", "content": prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error          = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response       = ""
            error          = str(e)
            safety_blocked = False

        latency_ms = (time.perf_counter() - t_start) * 1000
        return {
            "response": response, "context_docs": context_docs,
            "prompt": prompt, "latency_ms": round(latency_ms, 2),
            "error": error, "safety_blocked": safety_blocked,
            "blocked_by": None,
            "poisoned_doc_blocked": poison_result["blocked"],
            "poisoned_doc_injected": True,
        }

    def query_multi_turn(self, turns, attack_turn_index):
        for turn in turns:
            if turn["role"] == "user" and turn["content"] != "__PLACEHOLDER__":
                result = self.ml_clf.classify(turn["content"])
                if result["blocked"]:
                    return {
                        "response": BLOCKED_RESPONSE, "context_docs": [],
                        "prompt": "", "latency_ms": 0,
                        "error": None, "safety_blocked": False,
                        "blocked_by": "ml_query", "t1_response": "",
                    }
        return super().query_multi_turn(turns, attack_turn_index)


class HybridDefendedRAG(BaselineRAG):
    """
    S3: Hybrid defence — sequential gate with weighted fusion.

    Architecture:
      Step 1 — Rule fast path: if rule_score >= S3_HARD_BLOCK_THRESHOLD (0.85),
               block immediately. No ML inference needed.
               Handles 60-70% of obvious direct injection attacks.

      Step 2 — If rule score < 0.85: run ML classifier.
               Fuse: combined = S3_RULE_WEIGHT*rule + S3_ML_WEIGHT*ml
               Block if combined >= 0.5.

    This sequential gate design has three properties for the paper:
      1. Speed: average latency is much closer to S1 than S2 because
         ML is skipped for most obvious attacks (fast-path rate reported)
      2. Coverage: rules handle exact-match patterns, ML handles
         semantic patterns — complementary failure modes
      3. Tunability: weights and threshold are hyperparameters documented
         as grid-searched on the validation set

    Applied identically to user queries and retrieved document chunks.
    """

    SYSTEM_NAME = "S3"

    def __init__(self, db_path: str = config.CLEAN_DB_PATH):
        super().__init__(db_path=db_path)
        self.rule_clf = RuleClassifier()
        self.ml_clf   = MLClassifier()
        self._fast_path_count = 0
        self._total_count     = 0

    @property
    def fast_path_rate(self) -> float:
        """Fraction of classifications handled by rules alone (no ML call)."""
        if self._total_count == 0:
            return 0.0
        return self._fast_path_count / self._total_count

    def _hybrid_classify(self, text: str) -> dict:
        """
        Apply sequential gate + weighted fusion to a single text.

        Returns dict:
            score         : float — final combined score
            blocked       : bool
            fast_path     : bool — True if ML was skipped
            rule_score    : float
            ml_score      : float | None — None if fast-path taken
            triggered_by  : str — "rule_fast_path" | "hybrid" | "benign"
        """
        self._total_count += 1

        rule_result = self.rule_clf.classify(text)
        rule_score  = rule_result["score"]

        # Fast path: high-confidence rule match → block without ML
        if rule_score >= S3_HARD_BLOCK_THRESHOLD:
            self._fast_path_count += 1
            return {
                "score":        rule_score,
                "blocked":      True,
                "fast_path":    True,
                "rule_score":   rule_score,
                "ml_score":     None,
                "triggered_by": "rule_fast_path",
            }

        # Standard path: run ML and fuse scores
        ml_result  = self.ml_clf.classify(text)
        ml_score   = ml_result["score"]
        combined   = (S3_RULE_WEIGHT * rule_score
                      + S3_ML_WEIGHT  * ml_score)
        blocked    = combined >= 0.5

        return {
            "score":        round(combined, 4),
            "blocked":      blocked,
            "fast_path":    False,
            "rule_score":   rule_score,
            "ml_score":     ml_score,
            "triggered_by": "hybrid" if blocked else "benign",
        }

    def query(self, question: str,
              top_k: int = config.TOP_K) -> dict:
        t_start = time.perf_counter()

        query_result = self._hybrid_classify(question)
        if query_result["blocked"]:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": BLOCKED_RESPONSE, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": query_result["triggered_by"],
                "query_clf_result": query_result, "doc_clf_results": [],
            }

        raw_docs    = self.retrieve(question, top_k=top_k)
        clean_docs  = []
        doc_results = []
        for doc in raw_docs:
            doc_result = self._hybrid_classify(doc)
            doc_results.append(doc_result)
            if not doc_result["blocked"]:
                clean_docs.append(doc)

        if not clean_docs:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": NO_CLEAN_CONTEXT, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "hybrid_all_docs",
                "query_clf_result": query_result,
                "doc_clf_results": doc_results,
            }

        prompt = self.build_prompt(question, clean_docs)
        try:
            response       = self.llm.chat(
                messages      = [{"role": "user", "content": prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error          = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response       = ""
            error          = str(e)
            safety_blocked = False

        latency_ms = (time.perf_counter() - t_start) * 1000
        return {
            "response": response, "context_docs": clean_docs,
            "prompt": prompt, "latency_ms": round(latency_ms, 2),
            "error": error, "safety_blocked": safety_blocked,
            "blocked_by": None, "query_clf_result": query_result,
            "doc_clf_results": doc_results,
        }

    def query_with_injected_doc(self,
                                trigger_query: str,
                                poisoned_doc_text: str,
                                top_k: int = config.TOP_K) -> dict:
        t_start      = time.perf_counter()
        query_result = self._hybrid_classify(trigger_query)
        if query_result["blocked"]:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": BLOCKED_RESPONSE, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": query_result["triggered_by"],
                "poisoned_doc_injected": True,
            }

        poison_result = self._hybrid_classify(poisoned_doc_text)
        clean_docs    = self.retrieve(trigger_query, top_k=top_k - 1)
        context_docs  = (clean_docs if poison_result["blocked"]
                         else [poisoned_doc_text] + clean_docs)

        if not context_docs:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": NO_CLEAN_CONTEXT, "context_docs": [],
                "prompt": "", "latency_ms": round(latency_ms, 2),
                "error": None, "safety_blocked": False,
                "blocked_by": "hybrid_all_docs",
                "poisoned_doc_injected": True,
            }

        prompt = self.build_prompt(trigger_query, context_docs)
        try:
            response       = self.llm.chat(
                messages      = [{"role": "user", "content": prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error          = None
            safety_blocked = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response       = ""
            error          = str(e)
            safety_blocked = False

        latency_ms = (time.perf_counter() - t_start) * 1000
        return {
            "response": response, "context_docs": context_docs,
            "prompt": prompt, "latency_ms": round(latency_ms, 2),
            "error": error, "safety_blocked": safety_blocked,
            "blocked_by": None,
            "poisoned_doc_blocked": poison_result["blocked"],
            "poisoned_doc_injected": True,
        }

    def query_multi_turn(self, turns, attack_turn_index):
        for turn in turns:
            if turn["role"] == "user" and turn["content"] != "__PLACEHOLDER__":
                result = self._hybrid_classify(turn["content"])
                if result["blocked"]:
                    return {
                        "response": BLOCKED_RESPONSE, "context_docs": [],
                        "prompt": "", "latency_ms": 0,
                        "error": None, "safety_blocked": False,
                        "blocked_by": result["triggered_by"],
                        "t1_response": "",
                    }
        return super().query_multi_turn(turns, attack_turn_index)