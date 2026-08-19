"""
Local dense retriever for defended systems.

The checked-in Chroma store still has document text in SQLite (89,500
chunks) but the HNSW vector files are incomplete (`data.bin` missing),
so chromadb.query() cannot load the index. SentenceTransformer also
cannot be constructed in the current project venv (broken torch wheel).

This helper reads documents from the existing SQLite file and embeds
them with Chroma's ONNX all-MiniLM-L6-v2 function — the same model
family used to build the original collection. Results are cached next
to the DB. S0 (BaselineRAG) is not modified and does not use this path.
"""

import json
import os
import sqlite3

import numpy as np
from tqdm import tqdm

CACHE_EMB = "_eval_onnx_embeddings.npy"
CACHE_IDS = "_eval_onnx_ids.json"
CACHE_META = "_eval_onnx_meta.json"
BATCH = 64


class LocalDenseRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        sqlite_path = os.path.join(db_path, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"Chroma sqlite not found: {sqlite_path}")
        self.sqlite_path = sqlite_path
        self.ids, self.docs, self.embs = self._load_or_build()

    def _load_docs(self):
        conn = sqlite3.connect(self.sqlite_path)
        rows = conn.execute(
            """
            SELECT e.embedding_id, m.string_value
            FROM embeddings e
            JOIN embedding_metadata m
              ON m.id = e.id AND m.key = 'chroma:document'
            ORDER BY e.id
            """
        ).fetchall()
        conn.close()
        ids = [r[0] for r in rows]
        docs = [r[1] or "" for r in rows]
        return ids, docs

    def _embedder(self):
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )
        return ONNXMiniLM_L6_V2()

    def _load_or_build(self):
        emb_path = os.path.join(self.db_path, CACHE_EMB)
        ids_path = os.path.join(self.db_path, CACHE_IDS)
        ids, docs = self._load_docs()
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            cached_ids = json.load(open(ids_path))
            embs = np.load(emb_path)
            if cached_ids == ids and embs.shape[0] == len(ids):
                return ids, docs, embs

        print(f"  Building ONNX MiniLM embeddings for {len(docs)} chunks "
              f"(one-time cache at {emb_path})...")
        ef = self._embedder()
        vectors = []
        for i in tqdm(range(0, len(docs), BATCH), desc="  Embedding KB"):
            batch = docs[i:i + BATCH]
            vectors.extend(ef(batch))
        embs = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        embs = embs / norms
        np.save(emb_path, embs)
        with open(ids_path, "w") as f:
            json.dump(ids, f)
        with open(os.path.join(self.db_path, CACHE_META), "w") as f:
            json.dump(
                {
                    "n": len(ids),
                    "dim": int(embs.shape[1]),
                    "model": "onnx all-MiniLM-L6-v2",
                    "reason": "chroma HNSW data.bin missing; S0 not modified",
                },
                f,
                indent=2,
            )
        return ids, docs, embs

    def retrieve(self, query: str, top_k: int):
        if top_k <= 0:
            return []
        ef = getattr(self, "_ef", None)
        if ef is None:
            self._ef = self._embedder()
            ef = self._ef
        q = np.asarray(ef([query])[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.embs @ q
        if top_k >= len(scores):
            idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, top_k)[:top_k]
            idx = part[np.argsort(-scores[part])]
        chunks = []
        for i in idx:
            chunks.append({"chunk_id": self.ids[int(i)], "text": self.docs[int(i)]})
        return chunks

    def query(self, query_texts, n_results=5, include=None):
        """Chroma-shaped wrapper so retrieve_chunks can stay unchanged."""
        chunks = self.retrieve(query_texts[0], n_results)
        return {
            "ids": [[c["chunk_id"] for c in chunks]],
            "documents": [[c["text"] for c in chunks]],
        }
