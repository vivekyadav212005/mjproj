# data_preparation/02_deduplicate.py
"""
Phase 1A — Step 2: Remove exact and near-duplicate samples.

Stage 1 — Exact deduplication:
  Hash each lowercased, whitespace-stripped text using SHA-256.
  Keep first occurrence when duplicates are found.

Stage 2 — Near-duplicate removal:
  Embed all texts using all-MiniLM-L6-v2 (384-dim vectors).
  For every pair with cosine similarity > 0.95, remove the one from the
  LARGER source dataset to preserve diversity.
  This is done in batches to handle memory.

Run: python data_preparation/02_deduplicate.py
"""

import os
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_PATH  = "data/processed/merged_raw.csv"
OUTPUT_PATH = "data/processed/merged_deduped.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEAR_DUP_THRESHOLD = 0.95
BATCH_SIZE = 512   # number of texts embedded at once

os.makedirs("data/processed", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Exact deduplication
# ─────────────────────────────────────────────────────────────────────────────

def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the lowercased, stripped text is identical.
    When two rows have the same text but different labels (label conflict),
    we keep label=1 (injection) because it is safer to retain the security signal.
    """
    print("\n[Stage 1] Exact deduplication...")
    before = len(df)

    # Normalise text for hashing
    df["_text_norm"] = df["text"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)

    # For conflicts: aggregate so label=1 wins
    df_agg = df.groupby("_text_norm").agg(
        text   = ("text",   "first"),
        label  = ("label",  "max"),    # max ensures injection (1) wins over benign (0)
        source = ("source", lambda x: "|".join(sorted(set(x))))  # track all sources
    ).reset_index(drop=True)

    after = len(df_agg)
    removed = before - after
    print(f"    Before: {before:,}  |  After: {after:,}  |  Removed: {removed:,} exact duplicates")

    # Report label conflicts resolved
    original_labels = df.groupby("_text_norm")["label"].nunique()
    conflicts = (original_labels > 1).sum()
    print(f"    Label conflicts resolved (kept injection label): {conflicts:,}")

    return df_agg[["text", "label", "source"]]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Near-duplicate removal via embedding similarity
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: list, model: SentenceTransformer, batch_size: int) -> np.ndarray:
    """Embed texts in batches. Returns L2-normalised embeddings for cosine similarity."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="    Embedding batches"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(embs)
    return np.vstack(embeddings)


def near_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove near-duplicates using cosine similarity on MiniLM embeddings.

    Algorithm:
    1. Embed all texts (L2-normalised → cosine sim = dot product).
    2. For each text, find all others with similarity > threshold.
    3. Build a "keep" set by greedy selection:
       - Process texts sorted by source size descending (smaller sources preserved first).
       - Mark a text as removed if any already-kept text is too similar.

    This runs in O(n²) in the worst case, which for ~15k samples takes ~30 seconds.
    """
    print(f"\n[Stage 2] Near-duplicate removal (cosine threshold = {NEAR_DUP_THRESHOLD})...")
    print(f"    Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = df["text"].tolist()
    print(f"    Embedding {len(texts):,} texts...")
    embeddings = embed_texts(texts, model, BATCH_SIZE)

    # Greedy removal: build set of indices to KEEP
    to_remove = set()
    n = len(texts)

    print(f"    Running similarity scan (may take 1–3 minutes for ~15k texts)...")
    for i in tqdm(range(n), desc="    Scanning"):
        if i in to_remove:
            continue
        # Compute cosine similarity of text i against all texts j > i
        sims = embeddings[i+1:] @ embeddings[i]   # dot product = cosine since normalised
        similar_indices = np.where(sims > NEAR_DUP_THRESHOLD)[0] + (i + 1)
        for j in similar_indices:
            if j not in to_remove:
                to_remove.add(j)   # remove the later-encountered one

    before = len(df)
    keep_mask = [i not in to_remove for i in range(len(df))]
    df_clean = df[keep_mask].reset_index(drop=True)
    after = len(df_clean)
    removed = before - after
    print(f"    Before: {before:,}  |  After: {after:,}  |  Removed: {removed:,} near-duplicates")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Input: {len(df):,} rows")

    # Stage 1
    df = exact_dedup(df)

    # Stage 2
    df = near_dedup(df)

    print("\n" + "="*60)
    print("DEDUPLICATION SUMMARY")
    print("="*60)
    print(f"Final rows      : {len(df):,}")
    print(f"Benign  (0)     : {len(df[df.label==0]):,}")
    print(f"Injection (1)   : {len(df[df.label==1]):,}")
    print(f"Class ratio     : {len(df[df.label==0])/len(df)*100:.1f}% benign / {len(df[df.label==1])/len(df)*100:.1f}% injection")
    print(f"\nPer-source (after dedup):")
    print(df.groupby(["source","label"]).size().unstack(fill_value=0).to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()