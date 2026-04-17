import time
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import config
from rag.llm_client import get_llm_client, LLMError, SAFETY_BLOCKED


class BaselineRAG:
    SYSTEM_NAME = "S0"

    def __init__(self, db_path: str = config.CLEAN_DB_PATH):
        self.db_path = db_path
        self.llm = get_llm_client()
        self._init_chroma()

    def _init_chroma(self):
        ef = SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=self.db_path)
        self.collection = client.get_collection(
            config.COLLECTION_NAME,
            embedding_function=ef,
        )

    def retrieve(self, query: str, top_k: int = config.TOP_K):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents"],
        )
        return results["documents"][0] if results["documents"] else []

    def build_prompt(self, question: str, context_docs: list[str]):
        context = "\n\n".join(context_docs) if context_docs else "No context available."
        return config.S0_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

    def query(self, question: str, top_k: int = config.TOP_K):
        t_start = time.perf_counter()

        context_docs = self.retrieve(question, top_k=top_k)
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

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "response": response,
            "context_docs": context_docs,
            "prompt": prompt,
            "latency_ms": round(latency_ms, 2),
            "error": error,
            "safety_blocked": safety_blocked,
        }