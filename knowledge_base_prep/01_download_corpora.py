# knowledge_base_prep/01_download_corpora.py
"""
Phase 1B — Step 1: Download the three domain corpora.

Downloads and saves raw data from:
  Healthcare : qiaojin/PubMedQA  (pqa_labeled config)
               Fields: pubid, question, context.contexts, long_answer,
                       final_decision
  Finance    : gtfintechlab/financial_phrasebank_sentences_allagree
               (parquet mirror — no dataset script dependency)
               Fields: sentence, label
               + gbharti/finance-alpaca for QA pairs
               Fields: output, instruction
  General QA : rajpurkar/squad
               Fields: id, title, context, question, answers.text

Saves each to knowledge_base/corpora/ as JSON for reproducibility.
The raw corpora are saved BEFORE chunking so every downstream step
is reproducible from the same source files.
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm

RAW_CACHE   = "data/raw"
OUTPUT_DIR  = "knowledge_base/corpora"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = os.path.abspath(RAW_CACHE)
os.environ["HF_HOME"]           = os.path.abspath(RAW_CACHE)


# ─────────────────────────────────────────────────────────────────────────────
# Healthcare — PubMedQA (pqa_labeled)
# Schema: pubid(int), question(str), context.contexts(list[str]),
#         long_answer(str), final_decision(str: yes/no/maybe)
# We use context.contexts as the KB document text.
# We use question + long_answer as QA pairs.
# ─────────────────────────────────────────────────────────────────────────────

def download_healthcare():
    print("\n[1/3] Downloading qiaojin/PubMedQA (pqa_labeled)...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"      Loaded {len(ds):,} labeled PubMed QA instances")
    print(f"      Columns: {ds.column_names}")

    records = []
    for row in tqdm(ds, desc="      Processing"):
        # context.contexts is a list of abstract sentences — join them
        contexts = row["context"]["contexts"] if isinstance(
            row["context"], dict) else row["context"]
        full_abstract = " ".join(contexts) if isinstance(
            contexts, list) else str(contexts)

        records.append({
            "doc_id":         f"pubmed_{row['pubid']}",
            "document_text":  full_abstract,
            "question":       row["question"],
            "answer":         row["long_answer"],
            "final_decision": row["final_decision"],
            "domain":         "healthcare",
            "source_dataset": "qiaojin/PubMedQA",
        })

    out_path = os.path.join(OUTPUT_DIR, "healthcare_raw.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"      Saved {len(records):,} records → {out_path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Finance — financial_phrasebank (parquet mirror) + finance-alpaca
#
# financial_phrasebank: sentence(str), label(int 0=negative,1=neutral,2=pos)
#   Used as KB document sentences. Groups of sentences are assembled into
#   paragraph-length documents for chunking.
#
# gbharti/finance-alpaca: instruction(str), output(str)
#   Used as QA pairs (instruction = question, output = answer)
# ─────────────────────────────────────────────────────────────────────────────

def download_finance():
    print("\n[2/3] Downloading finance corpora...")

    # ── Part A: document corpus from financial phrasebank ─────────────────
    # gtfintechlab/financial_phrasebank_sentences_allagree uses numeric
    # config IDs. "5768" = sentences_allagree (2,264 sentences, 100% agreement)
    # "78516" = sentences_75agree, "944601" = sentences_50agree
    # We use "5768" (allagree) for highest annotation quality.
    print("      Part A: gtfintechlab/financial_phrasebank_sentences_allagree")
    paragraphs = []
    try:
        ds_fp = load_dataset(
            "gtfintechlab/financial_phrasebank_sentences_allagree",
            "5768",           # <-- the fix: numeric config ID for allagree
            split="train"
        )
        print(f"      Loaded {len(ds_fp):,} sentences | columns: {ds_fp.column_names}")

        # Group sentences into ~5-sentence paragraph documents
        sentences = [
            str(r.get("sentence", r.get("text", ""))).strip()
            for r in ds_fp
            if r.get("sentence") or r.get("text")
        ]
        GROUP_SIZE = 5
        for i in range(0, len(sentences), GROUP_SIZE):
            group = sentences[i: i + GROUP_SIZE]
            if len(group) >= 2:
                paragraphs.append(" ".join(group))
        print(f"      Assembled {len(paragraphs):,} paragraph documents "
              f"from {len(sentences):,} sentences")

    except Exception as e:
        print(f"      WARNING: phrasebank loader failed: {e}")
        print("      Finance KB will use finance-alpaca documents only.")

    # ── Part B: QA pairs from finance-alpaca ─────────────────────────────
    print("      Part B: gbharti/finance-alpaca (QA pairs)")
    try:
        ds_alpaca = load_dataset("gbharti/finance-alpaca", split="train")
        print(f"      Loaded {len(ds_alpaca):,} QA pairs | "
              f"columns: {ds_alpaca.column_names}")
        qa_pairs = []
        for row in ds_alpaca:
            instruction = str(row.get("instruction", "")).strip()
            output      = str(row.get("output", "")).strip()
            if instruction and output and len(output) > 20:
                qa_pairs.append({
                    "question": instruction,
                    "answer":   output,
                })
        print(f"      Kept {len(qa_pairs):,} non-trivial QA pairs")
    except Exception as e:
        print(f"      WARNING: finance-alpaca failed ({e}) — using empty QA list")
        qa_pairs = []

    # ── Assemble finance records ──────────────────────────────────────────
    records = []

    # Phrasebank paragraph documents (KB corpus, no QA pair)
    for i, para in enumerate(paragraphs):
        records.append({
            "doc_id":         f"finance_phrasebank_{i:05d}",
            "document_text":  para,
            "question":       None,
            "answer":         None,
            "domain":         "finance",
            "source_dataset": "gtfintechlab/financial_phrasebank_sentences_allagree",
        })

    # Finance-alpaca QA pairs (answer text = document, question = query)
    for i, qa in enumerate(qa_pairs):
        records.append({
            "doc_id":         f"finance_alpaca_{i:05d}",
            "document_text":  qa["answer"],
            "question":       qa["question"],
            "answer":         qa["answer"],
            "domain":         "finance",
            "source_dataset": "gbharti/finance-alpaca",
        })

    out_path = os.path.join(OUTPUT_DIR, "finance_raw.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"      Saved {len(records):,} total finance records → {out_path}")
    print(f"        Phrasebank docs : {len(paragraphs):,}")
    print(f"        Finance-alpaca  : {len(qa_pairs):,}")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# General QA — SQuAD
# Schema: id(str), title(str), context(str), question(str),
#         answers.text(list[str]), answers.answer_start(list[int])
# We use context as document text, question+answers.text[0] as QA pairs.
# ─────────────────────────────────────────────────────────────────────────────

def download_general():
    print("\n[3/3] Downloading rajpurkar/squad...")
    ds = load_dataset("rajpurkar/squad", split="train")
    print(f"      Loaded {len(ds):,} QA instances | columns: {ds.column_names}")

    # Deduplicate contexts — SQuAD has many questions per passage
    # We keep unique contexts as documents, and one QA pair per unique context
    seen_contexts = {}
    records       = []

    for row in tqdm(ds, desc="      Processing"):
        ctx  = row["context"].strip()
        q    = row["question"].strip()
        ans  = row["answers"]["text"]
        a    = ans[0].strip() if ans else ""

        if ctx not in seen_contexts:
            seen_contexts[ctx] = {
                "doc_id":         f"squad_{row['id']}",
                "document_text":  ctx,
                "question":       q,
                "answer":         a,
                "title":          row["title"],
                "domain":         "general",
                "source_dataset": "rajpurkar/squad",
            }

    records = list(seen_contexts.values())
    out_path = os.path.join(OUTPUT_DIR, "general_raw.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"      Saved {len(records):,} unique-context records → {out_path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 1B — Step 1: Download Domain Corpora")
    print("=" * 60)

    h_records = download_healthcare()
    f_records = download_finance()
    g_records = download_general()

    summary = {
        "healthcare": len(h_records),
        "finance":    len(f_records),
        "general":    len(g_records),
        "total":      len(h_records) + len(f_records) + len(g_records),
    }

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<15}: {v:,} records")

    summary_path = os.path.join(OUTPUT_DIR, "download_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()