from security.detector import is_malicious
from security.simple_ml_detector import SimpleMLDetector
import time
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

import config
from rag.llm_client import get_llm_client, LLMError, SAFETY_BLOCKED


class BaselineRAG:
    """S2: RAG with rule-based + ML defence."""

    SYSTEM_NAME = "S2"

    def __init__(self, db_path: str = config.CLEAN_DB_PATH):
        self.db_path = db_path
        self.llm     = get_llm_client()

        self._init_chroma()

        # 🔥 S2 ML detector
        self.ml_detector = SimpleMLDetector()
        self.ml_detector.train("results/s0/S0_attack_results.jsonl")

    def _init_chroma(self):
        ef = SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL
        )
        client          = chromadb.PersistentClient(path=self.db_path)
        self.collection = client.get_collection(
            config.COLLECTION_NAME,
            embedding_function=ef,
        )

    # ── Retrieval ─────────────────────────────────────────

    def retrieve(self, query: str,
                 top_k: int = config.TOP_K) -> list[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents"],
        )
        return results["documents"][0] if results["documents"] else []

    # ── Prompt construction ───────────────────────────────

    def build_prompt(self, question: str,
                     context_docs: list[str]) -> str:
        context = "\n\n".join(context_docs) if context_docs \
                  else "No context available."
        return config.S0_PROMPT_TEMPLATE.format(
            context  = context,
            question = question,
        )

    # ── S2 Query (RULE + ML DEFENCE) ──────────────────────

    def query(self, question: str,
              top_k: int = config.TOP_K) -> dict:

        query = question

        # 🔒 S1 — rule-based
        if is_malicious(query):
            return {
                "response": "Blocked (rule-based detector).",
                "context_docs": [],
                "prompt": "",
                "latency_ms": 0,
                "error": None,
                "safety_blocked": False,
            }

        # 🔒 S2 — ML-based
        if self.ml_detector.predict(query):
            return {
                "response": "Blocked (ML detector).",
                "context_docs": [],
                "prompt": "",
                "latency_ms": 0,
                "error": None,
                "safety_blocked": False,
            }

        # ✅ normal RAG flow
        t_start      = time.perf_counter()
        context_docs = self.retrieve(question, top_k=top_k)
        prompt       = self.build_prompt(question, context_docs)

        try:
            response = self.llm.chat(
                messages      = [{"role": "user", "content": prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error           = None
            safety_blocked  = (response == SAFETY_BLOCKED)
        except LLMError as e:
            response        = ""
            error           = str(e)
            safety_blocked  = False

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "response":       response,
            "context_docs":   context_docs,
            "prompt":         prompt,
            "latency_ms":     round(latency_ms, 2),
            "error":          error,
            "safety_blocked": safety_blocked,
        }

    # ── Indirect attack (unchanged — IMPORTANT) ───────────

    def query_with_injected_doc(self,
                                trigger_query: str,
                                poisoned_doc_text: str,
                                top_k: int = config.TOP_K) -> dict:

        t_start = time.perf_counter()

        clean_docs   = self.retrieve(trigger_query, top_k=top_k - 1)
        context_docs = [poisoned_doc_text] + clean_docs

        prompt = self.build_prompt(trigger_query, context_docs)

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
            "response":              response,
            "context_docs":          context_docs,
            "prompt":                prompt,
            "latency_ms":            round(latency_ms, 2),
            "error":                 error,
            "safety_blocked":        safety_blocked,
            "poisoned_doc_injected": True,
        }

    # ── Multi-turn (unchanged) ────────────────────────────

    def query_multi_turn(self,
                         turns: list[dict],
                         attack_turn_index: int) -> dict:

        t_start = time.perf_counter()

        t1_content = next(
            (t["content"] for t in turns if t["role"] == "user"), None
        )
        if t1_content is None:
            return {
                "response": "", "context_docs": [], "prompt": "",
                "latency_ms": 0, "error": "No user turn found",
                "safety_blocked": False, "t1_response": "",
            }

        context_docs = self.retrieve(t1_content, top_k=config.TOP_K)
        t1_prompt    = self.build_prompt(t1_content, context_docs)

        try:
            t1_response = self.llm.chat(
                messages      = [{"role": "user", "content": t1_prompt}],
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
        except LLMError as e:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return {
                "response": "", "context_docs": context_docs,
                "prompt": t1_prompt, "latency_ms": round(latency_ms, 2),
                "error": str(e), "safety_blocked": False, "t1_response": "",
            }

        attack_content = turns[attack_turn_index]["content"]
        final_messages = [
            {"role": "user",      "content": t1_prompt},
            {"role": "assistant", "content": t1_response},
            {"role": "user",      "content": attack_content},
        ]

        try:
            t2_response = self.llm.chat(
                messages      = final_messages,
                system_prompt = config.RAG_SYSTEM_PROMPT,
            )
            error          = None
            safety_blocked = (t2_response == SAFETY_BLOCKED)
        except LLMError as e:
            t2_response    = ""
            error          = str(e)
            safety_blocked = False

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "response":       t2_response,
            "t1_response":    t1_response,
            "context_docs":   context_docs,
            "prompt":         t1_prompt,
            "latency_ms":     round(latency_ms, 2),
            "error":          error,
            "safety_blocked": safety_blocked,
        }