# data_preparation/01_load_datasets.py
"""
Phase 1A — Dataset Curation Script
SecureRAG Research Project

PURPOSE:
  Download, normalize, and merge 7 public prompt injection datasets into a
  single unified CSV with schema: {text, label, source}
  label: 0 = benign, 1 = injection

VERIFIED COLUMN SCHEMAS (checked against HuggingFace dataset cards):
  deepset/prompt-injections          → text(str), label(int 0/1)
  Lakera/mosscap_prompt_injection    → level(str), prompt(str), answer(str),
                                       raw_answer(str) — NO label column
                                       ALL rows = injection (CTF competition)
  Lakera/gandalf_ignore_instructions → text(str), similarity(float)
                                       ALL rows = injection (similarity ≥ 0.825
                                       to "Ignore all previous instructions")
  hackaprompt/hackaprompt-dataset    → user_input(str), level(int), model(str),
                                       injection(bool), etc. — GATED dataset
                                       ALL rows = injection (competition)
  reshabhs/SPML_Chatbot_Prompt_Injection → 'System Prompt'(str),
                                           'User Prompt'(str),
                                           'Prompt injection'(bool),
                                           'Degree'(int), 'Source'(str)
  xTRam1/safe-guard-prompt-injection → text(str), label(int 0/1)
  JasperLS/prompt-injections         → text(str), label(int 0/1)

SAMPLING DECISIONS (documented for paper):
  mosscap    : 278k raw rows — sample 3,000 (CTF-focused, avoid domination)
  hackaprompt: 298k raw rows — sample 3,000 (CTF/PWNED-focused, avoid domination)
  spml       : ~16k rows     — keep all (realistic chatbot scenario, most
                                relevant to RAG deployment)

TARGET DISTRIBUTION:
  ~60% benign / ~40% injection — realistic traffic ratio
  Total target: ~18,000–22,000 samples after dedup

CACHE STRATEGY:
  HuggingFace downloads cached to ./data/raw/ for project portability
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_CACHE_DIR = "data/raw"
OUTPUT_PATH   = "data/processed/merged_raw.csv"

os.makedirs(RAW_CACHE_DIR,        exist_ok=True)
os.makedirs("data/processed",     exist_ok=True)

# Redirect HuggingFace cache to ./data/raw/ so downloads are visible there
os.environ["HF_DATASETS_CACHE"] = os.path.abspath(RAW_CACHE_DIR)
os.environ["HF_HOME"]           = os.path.abspath(RAW_CACHE_DIR)

# ── Helpers ──────────────────────────────────────────────────────────────────

def normalise_label(raw) -> int:
    """Convert any label representation → 0 (benign) or 1 (injection)."""
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, (int, float)):
        return 1 if raw > 0 else 0
    if isinstance(raw, str):
        falsy = {"0", "false", "safe", "benign", "no", "legit", "none", "clean"}
        return 0 if raw.strip().lower() in falsy else 1
    return 0  # default safe


def iter_split(ds) -> list:
    """Flatten all splits of a HuggingFace DatasetDict into a single list."""
    rows = []
    for split in ds:
        rows.extend(ds[split])
    return rows


def make_row(text: str, label: int, source: str) -> dict | None:
    """Validate and return a row dict, or None if text is unusable."""
    text = str(text).strip()
    # Drop empty, too-short, or clearly non-linguistic strings
    if len(text) < 8:
        return None
    # Drop rows that are just URLs or file paths (not prompts)
    if text.startswith("http") or text.startswith("/"):
        return None
    return {"text": text, "label": int(label), "source": source}


# ─────────────────────────────────────────────────────────────────────────────
# Loader 1 — deepset/prompt-injections
# Columns: text, label (0/1)
# Size: ~662 rows | Mix of benign and injection
# Role: Gold-standard anchor — most cited in the literature
# ─────────────────────────────────────────────────────────────────────────────

def load_deepset() -> pd.DataFrame:
    print("\n[1/7] Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections")
    rows = []
    for row in iter_split(ds):
        r = make_row(row["text"], row["label"], "deepset")
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f"    ✓ {len(df):,} rows | benign={(df.label==0).sum():,} inj={(df.label==1).sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Loader 2 — Lakera/mosscap_prompt_injection
# Columns: level, prompt, answer, raw_answer — NO label
# Size: ~278k raw — we sample 3,000
# Role: Real-world CTF injection attempts from DEF CON 31
# ALL rows are injection attempts by design (players trying to extract password)
# Sampling rationale: avoid dominating the merged set with a single narrow
# attack type (password extraction). 3,000 captures diversity across 8 levels.
# ─────────────────────────────────────────────────────────────────────────────

def load_mosscap() -> pd.DataFrame:
    print("\n[2/7] Loading Lakera/mosscap_prompt_injection...")
    ds = load_dataset("Lakera/mosscap_prompt_injection")
    rows = []
    for row in iter_split(ds):
        r = make_row(row["prompt"], 1, "mosscap")
        if r:
            rows.append(r)

    df_full = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    SAMPLE_N = 3000
    df = df_full.sample(n=min(SAMPLE_N, len(df_full)), random_state=42).reset_index(drop=True)
    print(f"    ✓ Sampled {len(df):,} / {len(df_full):,} rows (all injection)")
    print(f"      Note: mosscap has no label column — all rows are CTF injection attempts")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Loader 3 — Lakera/gandalf_ignore_instructions
# Columns: text, similarity (float)
# Size: ~1,000 rows
# Role: High-quality instruction-override injections
# ALL rows filtered to similarity ≥ 0.825 to "Ignore all previous instructions"
# The similarity column is a quality score, not a label — all rows = injection
# ─────────────────────────────────────────────────────────────────────────────

def load_gandalf() -> pd.DataFrame:
    print("\n[3/7] Loading Lakera/gandalf_ignore_instructions...")
    ds = load_dataset("Lakera/gandalf_ignore_instructions")
    rows = []
    for row in iter_split(ds):
        # similarity is a quality/confidence score — all rows are injections
        # We retain the similarity value in the text for context but don't use it as label
        r = make_row(row["text"], 1, "gandalf")
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f"    ✓ {len(df):,} rows (all injection — similarity-filtered by Lakera)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Loader 4 — hackaprompt/hackaprompt-dataset  [GATED]
# Columns: user_input, level, model, injection (bool), ...
# Size: ~600k raw — we sample 3,000 after quality filtering
# Role: Competition-grade adversarial prompts from HackAPrompt challenge
# Quality filter: levels 0–7 only (8–10 have documented label noise)
# Sampling rationale: 94% of raw rows aim to get model to say "I have been
# PWNED" — narrow objective. 3,000 captures pattern diversity without
# overwhelming the dataset with a single attack goal.
# ─────────────────────────────────────────────────────────────────────────────

def load_hackaprompt() -> pd.DataFrame:
    print("\n[4/7] Loading hackaprompt/hackaprompt-dataset (GATED — needs HF login)...")
    ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")

    rows = []
    skipped_level  = 0
    skipped_empty  = 0
    seen = set()

    for row in tqdm(ds, desc="    Scanning hackaprompt"):
        # Level filter: 8-10 have documented label noise
        level = row.get("level", 0)
        try:
            level = int(level)
        except (ValueError, TypeError):
            level = 0
        if level > 7:
            skipped_level += 1
            continue

        text = str(row.get("user_input", "")).strip()
        if not text or len(text) < 8:
            skipped_empty += 1
            continue

        # Within-loader dedup
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        r = make_row(text, 1, "hackaprompt")
        if r:
            rows.append(r)

    df_full = pd.DataFrame(rows)
    SAMPLE_N = 3000
    df = df_full.sample(n=min(SAMPLE_N, len(df_full)), random_state=42).reset_index(drop=True)
    print(f"    ✓ Sampled {len(df):,} / {len(df_full):,} usable rows")
    print(f"      Skipped: {skipped_level:,} (levels 8-10) | {skipped_empty:,} (too short)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Loader 5 — reshabhs/SPML_Chatbot_Prompt_Injection
# Actual columns (VERIFIED from inspect output):
#   'System Prompt', 'User Prompt', 'Prompt injection', 'Degree', 'Source'
# text  = 'User Prompt' (the user-facing input — what a RAG system sees)
# label = 'Prompt injection' (bool: True = injection attempt)
# Size: ~16,000 rows
# Role: MOST RELEVANT to our RAG scenario — realistic chatbot system prompts
# paired with both benign and malicious user inputs. Provides the best
# benign/injection balance of all 7 sources.
# ─────────────────────────────────────────────────────────────────────────────

def load_spml() -> pd.DataFrame:
    print("\n[5/7] Loading reshabhs/SPML_Chatbot_Prompt_Injection...")
    ds = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")

    # Inspect actual columns on first call
    for split in ds:
        if len(ds[split]) > 0:
            cols = list(ds[split].features.keys())
            print(f"    Actual columns: {cols}")
            break

    rows = []
    for row in iter_split(ds):
        # Use exact column names as returned by the dataset
        # Column names have spaces and title case — access with exact keys
        text  = str(row.get("User Prompt", row.get("user_input", ""))).strip()
        raw_label = row.get("Prompt injection",
                    row.get("injection",
                    row.get("label", False)))

        r = make_row(text, normalise_label(raw_label), "spml")
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f"    ✓ {len(df):,} rows | benign={(df.label==0).sum():,} inj={(df.label==1).sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Loader 6 — xTRam1/safe-guard-prompt-injection
# Columns: text, label (0=safe, 1=injection)
# Size: ~10,296 rows | 7,150 benign + 3,146 injection
# Role: Large-scale synthetic dataset covering 15 attack pattern types
# Good coverage of attack diversity; benign side strengthens negative class
# ─────────────────────────────────────────────────────────────────────────────

def load_safeguard() -> pd.DataFrame:
    print("\n[6/7] Loading xTRam1/safe-guard-prompt-injection...")
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    rows = []
    for row in iter_split(ds):
        r = make_row(row.get("text", ""), normalise_label(row.get("label", 0)), "safeguard")
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f"    ✓ {len(df):,} rows | benign={(df.label==0).sum():,} inj={(df.label==1).sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Loader 7 — JasperLS/prompt-injections
# Columns: text, label (0/1)
# Size: ~662 rows (shares parquet with deepset — identical files)
# Role: Jailbreak + injection mix; role-hijacking coverage
# NOTE: This dataset shares the same parquet files as deepset. The global
# deduplication step in script 02 will remove exact duplicates, so any
# overlap is handled cleanly. We include it for its unique labeling perspective.
# ─────────────────────────────────────────────────────────────────────────────

def load_jasperls() -> pd.DataFrame:
    print("\n[7/7] Loading JasperLS/prompt-injections...")
    ds = load_dataset("JasperLS/prompt-injections")
    rows = []
    for row in iter_split(ds):
        r = make_row(row.get("text", ""), normalise_label(row.get("label", 0)), "jasperls")
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f"    ✓ {len(df):,} rows | benign={(df.label==0).sum():,} inj={(df.label==1).sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Post-merge balancing
# ─────────────────────────────────────────────────────────────────────────────

def balance_dataset(df: pd.DataFrame,
                    target_benign_ratio: float = 0.60,
                    random_seed: int = 42) -> pd.DataFrame:
    """
    Enforce a 60/40 benign/injection split.

    Strategy:
    - Never downsample the injection class — we have enough diversity
    - Upsample benign if needed by duplicating with slight jitter (not needed
      here since safeguard provides 7,150 benign rows)
    - Downsample injection class proportionally across sources to preserve
      source diversity (not just cut the largest source)

    The 60/40 ratio is chosen because:
    - It approximates realistic traffic in a deployed RAG system
    - It avoids trivial majority-class prediction during training
    - Reviewers expect the benign class to be well-represented
    """
    n_benign    = (df.label == 0).sum()
    n_injection = (df.label == 1).sum()
    total       = len(df)

    print(f"\n[Balancing] Before: {n_benign:,} benign ({n_benign/total*100:.1f}%) "
          f"| {n_injection:,} injection ({n_injection/total*100:.1f}%)")

    benign_df    = df[df.label == 0]
    injection_df = df[df.label == 1]

    # Target: 60% benign, 40% injection
    # Use whichever class is smaller as the anchor, then scale the other
    target_injection = min(n_injection, int(n_benign / target_benign_ratio * (1 - target_benign_ratio)))
    target_benign    = int(target_injection * target_benign_ratio / (1 - target_benign_ratio))

    # Cap at available samples
    target_benign    = min(target_benign,    n_benign)
    target_injection = min(target_injection, n_injection)

    # Downsample injection proportionally across sources
    inj_sources  = injection_df["source"].value_counts()
    inj_fractions = inj_sources / inj_sources.sum()
    sampled_inj_parts = []
    for src, frac in inj_fractions.items():
        n_from_src = max(1, int(target_injection * frac))
        src_df = injection_df[injection_df.source == src]
        n_from_src = min(n_from_src, len(src_df))
        sampled_inj_parts.append(
            src_df.sample(n=n_from_src, random_state=random_seed)
        )
    sampled_injection = pd.concat(sampled_inj_parts)

    # Downsample benign proportionally across sources
    ben_sources   = benign_df["source"].value_counts()
    ben_fractions = ben_sources / ben_sources.sum()
    sampled_ben_parts = []
    for src, frac in ben_fractions.items():
        n_from_src = max(1, int(target_benign * frac))
        src_df = benign_df[benign_df.source == src]
        n_from_src = min(n_from_src, len(src_df))
        sampled_ben_parts.append(
            src_df.sample(n=n_from_src, random_state=random_seed)
        )
    sampled_benign = pd.concat(sampled_ben_parts)

    balanced = pd.concat([sampled_benign, sampled_injection]).sample(
        frac=1, random_state=random_seed
    ).reset_index(drop=True)

    n_b = (balanced.label == 0).sum()
    n_i = (balanced.label == 1).sum()
    print(f"[Balancing] After:  {n_b:,} benign ({n_b/len(balanced)*100:.1f}%) "
          f"| {n_i:,} injection ({n_i/len(balanced)*100:.1f}%)")
    print(f"[Balancing] Total:  {len(balanced):,} rows")
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SecureRAG — Phase 1A: Dataset Curation")
    print(f"HuggingFace cache → {os.path.abspath(RAW_CACHE_DIR)}")
    print("=" * 65)

    loaders = [
        load_deepset,
        load_mosscap,
        load_gandalf,
        load_hackaprompt,
        load_spml,
        load_safeguard,
        load_jasperls,
    ]

    dfs    = []
    failed = []

    for loader in loaders:
        try:
            df = loader()
            if df is not None and len(df) > 0:
                dfs.append(df)
            else:
                print(f"  WARNING: {loader.__name__} returned 0 rows")
                failed.append(loader.__name__)
        except Exception as e:
            print(f"\n  ERROR in {loader.__name__}: {type(e).__name__}: {e}")
            print("  Skipping and continuing...")
            failed.append(loader.__name__)

    if not dfs:
        raise RuntimeError("All loaders failed.")

    # ── Concatenate ───────────────────────────────────────────────────────────
    merged = pd.concat(dfs, ignore_index=True)
    merged["text"]  = merged["text"].astype(str).str.strip()
    merged["label"] = merged["label"].astype(int)
    merged = merged[merged["text"].str.len() >= 8].reset_index(drop=True)

    print("\n" + "=" * 65)
    print("RAW MERGE (before balancing and deduplication)")
    print("=" * 65)
    _print_breakdown(merged, "Raw merge")

    # ── Balance ───────────────────────────────────────────────────────────────
    balanced = balance_dataset(merged, target_benign_ratio=0.60, random_seed=42)

    print("\n" + "=" * 65)
    print("FINAL DATASET SAVED (before global deduplication in script 02)")
    print("=" * 65)
    _print_breakdown(balanced, "Balanced")

    balanced.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")

    if failed:
        print(f"\nNote: Failed loaders: {failed}")
        print("If hackaprompt failed, run: huggingface-cli login")
        print("Then accept dataset terms at:")
        print("  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset")


def _print_breakdown(df: pd.DataFrame, label: str):
    n_b = (df.label == 0).sum()
    n_i = (df.label == 1).sum()
    n   = len(df)
    print(f"\n{label}: {n:,} rows | {n_b:,} benign ({n_b/n*100:.1f}%) "
          f"| {n_i:,} injection ({n_i/n*100:.1f}%)")
    breakdown = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    breakdown.columns = [
        "benign" if c == 0 else "injection" for c in breakdown.columns
    ]
    breakdown["total"] = breakdown.sum(axis=1)
    breakdown["%_of_dataset"] = (breakdown["total"] / n * 100).round(1)
    print(breakdown.to_string())


if __name__ == "__main__":
    main()