"""
demo/retriever.py — Tiny in-memory dense retriever for the SecureRAG demo.

Loads 6 fictional policy documents from demo/knowledge_base/, embeds them
with Chroma's ONNX all-MiniLM-L6-v2 (no PyTorch dependency), and serves
cosine-similarity top-k retrieval.

This module is entirely independent of the research ChromaDB (89,500 chunks).
"""

import os
import numpy as np

DEMO_KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")


def _load_documents():
    """Load all .txt files from the demo knowledge base directory."""
    docs = {}
    for fname in sorted(os.listdir(DEMO_KB_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(DEMO_KB_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    return docs


def _get_embedder():
    """Return Chroma's built-in ONNX MiniLM-L6-v2 embedding function."""
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
        ONNXMiniLM_L6_V2,
    )
    return ONNXMiniLM_L6_V2()


class DemoRetriever:
    """
    In-memory dense retriever over the demo knowledge base.

    Attributes:
        doc_names: list of document filenames
        doc_texts: list of document contents
        embeddings: (N, D) float32 array of L2-normalised document embeddings
    """

    def __init__(self):
        raw = _load_documents()
        self.doc_names = list(raw.keys())
        self.doc_texts = list(raw.values())

        ef = _get_embedder()
        vecs = ef(self.doc_texts)
        self.embeddings = np.vstack(
            [np.asarray(v, dtype=np.float32) for v in vecs]
        )
        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        self.embeddings = self.embeddings / norms
        self._ef = ef

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve the top-k most similar documents.

        Returns:
            list of dicts: [{
                "doc_name": str,
                "text": str,
                "similarity": float,
                "rank": int (1-based),
            }, ...]
        """
        q = np.asarray(self._ef([query])[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.embeddings @ q

        k = min(top_k, len(scores))
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            idx = part[np.argsort(-scores[part])]

        results = []
        for rank, i in enumerate(idx[:k], start=1):
            results.append({
                "doc_name": self.doc_names[int(i)],
                "text": self.doc_texts[int(i)],
                "similarity": round(float(scores[int(i)]), 4),
                "rank": rank,
            })
        return results
