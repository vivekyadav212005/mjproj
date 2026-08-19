#!/usr/bin/env python3
"""
Rebuild the Chroma index into chromadb/final_clean/ from the chunk text and
metadata already stored in chromadb/clean/chroma.sqlite3.

chromadb/clean is opened read-only and is never written to. The source store
has a usable SQLite document table but an unusable persisted HNSW segment,
so the vectors are recomputed here with all-MiniLM-L6-v2 and written to a
fresh persistent directory that every system in the final experiment shares.

Usage:
    myenv/bin/python final_experiment/build_final_index.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb  # noqa: E402
from chromadb.utils.embedding_functions import (  # noqa: E402
    SentenceTransformerEmbeddingFunction,
)

import config  # noqa: E402

SOURCE_DB = "chromadb/clean"
TARGET_DB = "chromadb/final_clean"
COLLECTION = config.COLLECTION_NAME
EMBED_MODEL = config.EMBED_MODEL
ADD_BATCH = 1000
MANIFEST = os.path.join(TARGET_DB, "final_index_manifest.json")


def load_chunks(source_db: str):
    """Read every chunk (id, text, metadata) from the source SQLite store."""
    sqlite_path = os.path.join(source_db, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(sqlite_path)

    # Read-only URI: the source store must not be modified.
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT e.id, e.embedding_id, m.key,
               m.string_value, m.int_value, m.float_value, m.bool_value
        FROM embeddings e
        JOIN embedding_metadata m ON m.id = e.id
        ORDER BY e.id
        """
    ).fetchall()
    conn.close()

    ordered_ids: list[str] = []
    texts: dict[str, str] = {}
    metas: dict[str, dict] = {}
    seen: set[str] = set()

    for _rowid, chunk_id, key, s_val, i_val, f_val, b_val in rows:
        if chunk_id not in seen:
            seen.add(chunk_id)
            ordered_ids.append(chunk_id)
            metas[chunk_id] = {}
        value = s_val
        if value is None:
            value = i_val
        if value is None:
            value = f_val
        if value is None:
            value = b_val
        if key == "chroma:document":
            texts[chunk_id] = value or ""
        elif value is not None:
            metas[chunk_id][key] = value

    return ordered_ids, texts, metas


def corpus_fingerprint(ids, texts) -> str:
    h = hashlib.sha256()
    for cid in ids:
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
        h.update(texts.get(cid, "").encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    print(f"Reading chunks from {SOURCE_DB} (read-only)...")
    ids, texts, metas = load_chunks(SOURCE_DB)
    n = len(ids)
    print(f"  chunks           : {n}")
    print(f"  metadata keys    : {sorted({k for m in metas.values() for k in m})}")
    poisoned = sum(
        1 for m in metas.values() if str(m.get("is_poisoned", "false")).lower() == "true"
    )
    print(f"  is_poisoned=true : {poisoned}")

    fingerprint = corpus_fingerprint(ids, texts)
    print(f"  corpus sha256    : {fingerprint}")

    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            prev = json.load(f)
        if prev.get("corpus_sha256") == fingerprint and prev.get("n_chunks") == n:
            print(f"\n{TARGET_DB} already built from this exact corpus. Nothing to do.")
            return 0
        print(f"\nExisting {TARGET_DB} has a different fingerprint; rebuilding.")

    if os.path.exists(TARGET_DB):
        # Only ever removes the target directory this script itself creates.
        shutil.rmtree(TARGET_DB)
    os.makedirs(TARGET_DB, exist_ok=True)

    print(f"\nEmbedding with {EMBED_MODEL} and writing {TARGET_DB}...")
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=TARGET_DB)
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        configuration={"hnsw": {"space": "cosine"}},
    )

    t_embed = time.time()
    for start in range(0, n, ADD_BATCH):
        batch_ids = ids[start:start + ADD_BATCH]
        collection.add(
            ids=batch_ids,
            documents=[texts.get(cid, "") for cid in batch_ids],
            metadatas=[metas.get(cid) or {"_": ""} for cid in batch_ids],
        )
        done = min(start + ADD_BATCH, n)
        rate = done / max(time.time() - t_embed, 1e-9)
        print(f"  {done}/{n} chunks  ({rate:.0f}/s)", flush=True)

    count = collection.count()
    print(f"\nCollection '{COLLECTION}' count: {count}")

    probe = collection.query(
        query_texts=["What are the side effects of metformin?"],
        n_results=config.TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    print("Probe query top-5 doc_ids:",
          [m.get("doc_id") for m in probe["metadatas"][0]])
    print("Probe distances:", [round(d, 4) for d in probe["distances"][0]])

    manifest = {
        "target_db": TARGET_DB,
        "source_db": SOURCE_DB,
        "source_db_modified": False,
        "collection": COLLECTION,
        "embedding_model": EMBED_MODEL,
        "embedding_function": "chromadb SentenceTransformerEmbeddingFunction",
        "normalize_embeddings": False,
        "hnsw_space": "cosine",
        "n_chunks": count,
        "n_chunks_source": n,
        "n_poisoned_chunks": poisoned,
        "corpus_sha256": fingerprint,
        "top_k": config.TOP_K,
        "build_seconds": round(time.time() - t0, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {MANIFEST}")
    print(f"Done in {manifest['build_seconds']}s")
    return 0 if count == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
