# knowledge_base_prep/02_chunk_and_ingest.py
"""
Phase 1B — Step 2: Chunk documents and ingest into ChromaDB (clean DB).

Chunking strategy:
  - Fixed-size: 256 tokens per chunk, 32-token overlap
  - Token counting via whitespace splitting (fast, reproducible,
    consistent with sentence-transformers tokenisation for our purposes)
  - Minimum chunk length: 50 tokens (skip too-short trailing chunks)

ChromaDB metadata schema per chunk:
  {
    "domain":         str   — "healthcare" | "finance" | "general"
    "source_dataset": str   — original HuggingFace dataset ID
    "doc_id":         str   — original document identifier
    "chunk_index":    int   — 0-based index of this chunk within its doc
    "chunk_total":    int   — total chunks for this document
    "is_poisoned":    str   — "false" for all clean chunks (stored as
                              str because ChromaDB metadata values must
                              be str/int/float/bool — use str for clarity)
    "has_qa_pair":    str   — "true" if a QA pair is associated
  }

The is_poisoned field is False for all chunks ingested here.
Poisoned variants are built in script 05.
"""

import os
import json
import uuid
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CORPORA_DIR  = "knowledge_base/corpora"
CHUNKS_DIR   = "knowledge_base/chunks"
CHROMADB_DIR = "chromadb/clean"

CHUNK_TOKENS   = 256
OVERLAP_TOKENS = 32
MIN_TOKENS     = 50
BATCH_SIZE     = 100    # ChromaDB add() batch size
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"   # same as Phase 1A

os.makedirs(CHUNKS_DIR,  exist_ok=True)
os.makedirs(CHROMADB_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser — whitespace split
# We do NOT use a subword tokeniser here because:
# 1. Reproducibility: behaviour is deterministic and dependency-free
# 2. Consistency: all-MiniLM-L6-v2 has a 256 WordPiece token limit,
#    but whitespace tokens are ~0.7× subword tokens on average for
#    English text, so 256 whitespace tokens ≈ 180–230 subword tokens,
#    safely within the model's context window.
# 3. Comparability: most published RAG papers report chunk size in
#    "tokens" using whitespace splitting as the approximation.
# ─────────────────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    return text.split()


def chunk_text(text: str,
               chunk_size: int = CHUNK_TOKENS,
               overlap: int    = OVERLAP_TOKENS,
               min_len: int    = MIN_TOKENS) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.
    Returns list of chunk strings.
    """
    tokens = tokenise(text)
    if len(tokens) < min_len:
        return [text] if len(tokens) > 0 else []

    chunks  = []
    start   = 0
    stride  = chunk_size - overlap

    while start < len(tokens):
        end   = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        if len(tokenise(chunk)) >= min_len:
            chunks.append(chunk)
        start += stride
        if end == len(tokens):
            break

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Process one domain file
# ─────────────────────────────────────────────────────────────────────────────

def process_domain(domain: str, raw_records: list[dict]) -> list[dict]:
    """
    Chunk all documents in raw_records.
    Returns list of chunk dicts ready for ChromaDB ingestion.
    """
    all_chunks = []
    skipped    = 0

    for rec in tqdm(raw_records, desc=f"    Chunking {domain}"):
        text = str(rec.get("document_text", "")).strip()
        if not text or len(text) < 20:
            skipped += 1
            continue

        chunks = chunk_text(text)
        total  = len(chunks)

        for idx, chunk_text_str in enumerate(chunks):
            # Unique chunk ID: domain_docid_chunkindex
            chunk_id = f"{domain}_{rec['doc_id']}_{idx:04d}"

            all_chunks.append({
                "chunk_id":       chunk_id,
                "text":           chunk_text_str,
                "domain":         domain,
                "source_dataset": rec.get("source_dataset", ""),
                "doc_id":         rec["doc_id"],
                "chunk_index":    idx,
                "chunk_total":    total,
                "is_poisoned":    "false",
                "has_qa_pair":    "true" if rec.get("question") else "false",
                # Store QA pair alongside chunk for query-set building
                "_question":      rec.get("question", ""),
                "_answer":        rec.get("answer", ""),
            })

    print(f"    {domain}: {len(raw_records):,} docs → "
          f"{len(all_chunks):,} chunks (skipped {skipped} empty docs)")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Ingest into ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def ingest_to_chroma(all_chunks: list[dict],
                     collection_name: str = "securerag_clean"):
    """
    Ingest chunk list into a persistent ChromaDB collection.
    Uses SentenceTransformer all-MiniLM-L6-v2 for embedding —
    the same model used in Phase 1A deduplication for consistency.
    """
    print(f"\n  Connecting to ChromaDB at {CHROMADB_DIR}...")
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMADB_DIR)

    # Delete existing collection if re-running
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"description": "SecureRAG clean knowledge base"},
    )
    print(f"  Created collection '{collection_name}'")

    # Strip internal QA fields from metadata before storing
    # (ChromaDB metadata must be str/int/float/bool — no nested dicts)
    ids       = []
    documents = []
    metadatas = []

    for chunk in all_chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append({
            "domain":         chunk["domain"],
            "source_dataset": chunk["source_dataset"],
            "doc_id":         chunk["doc_id"],
            "chunk_index":    chunk["chunk_index"],
            "chunk_total":    chunk["chunk_total"],
            "is_poisoned":    chunk["is_poisoned"],
            "has_qa_pair":    chunk["has_qa_pair"],
        })

    # Batch ingest
    total    = len(ids)
    inserted = 0
    print(f"  Ingesting {total:,} chunks in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, total, BATCH_SIZE), desc="  Ingesting"):
        batch_end = min(i + BATCH_SIZE, total)
        collection.add(
            ids       = ids[i:batch_end],
            documents = documents[i:batch_end],
            metadatas = metadatas[i:batch_end],
        )
        inserted += (batch_end - i)

    print(f"  Inserted {inserted:,} chunks into '{collection_name}'")
    print(f"  Collection count: {collection.count():,}")
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 1B — Step 2: Chunk and Ingest into ChromaDB")
    print("=" * 60)

    domains = ["healthcare", "finance", "general"]
    all_chunks_combined = []

    for domain in domains:
        raw_path = os.path.join(CORPORA_DIR, f"{domain}_raw.json")
        if not os.path.exists(raw_path):
            print(f"\nWARNING: {raw_path} not found — run script 01 first")
            continue

        print(f"\nProcessing {domain}...")
        with open(raw_path) as f:
            raw_records = json.load(f)

        chunks = process_domain(domain, raw_records)
        all_chunks_combined.extend(chunks)

        # Save domain-specific chunk file
        chunks_path = os.path.join(CHUNKS_DIR, f"{domain}_chunks.json")
        with open(chunks_path, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"  Saved {len(chunks):,} chunks → {chunks_path}")

    # Ingest all domains into one ChromaDB collection
    print(f"\nTotal chunks across all domains: {len(all_chunks_combined):,}")
    collection = ingest_to_chroma(all_chunks_combined)

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for domain in domains:
        domain_chunks = [c for c in all_chunks_combined if c["domain"] == domain]
        print(f"  {domain:<15}: {len(domain_chunks):,} chunks")
    print(f"  {'TOTAL':<15}: {len(all_chunks_combined):,} chunks")
    print(f"  ChromaDB path  : {CHROMADB_DIR}")
    print(f"  Embedding model: {EMBED_MODEL}")
    print(f"  Chunk size     : {CHUNK_TOKENS} tokens")
    print(f"  Overlap        : {OVERLAP_TOKENS} tokens")


if __name__ == "__main__":
    main()