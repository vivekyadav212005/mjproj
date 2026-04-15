# knowledge_base_prep/03_build_query_set.py
"""
Phase 1B — Step 3: Build the 150-query benign test set.

Selects 50 queries per domain from the downloaded corpora.
Each query has:
  - question: str
  - ground_truth_answer: str
  - source_doc_id: str      (links back to the KB document)
  - domain: str
  - query_id: str           (unique, deterministic)

Selection criteria per domain:
  Healthcare  : PubMedQA labeled — pick rows where final_decision is
                "yes" or "no" (not "maybe") to ensure clear ground-truth
                answers. Sort by answer length descending and take top 50
                (longer answers = richer ground truth = more meaningful
                ROUGE-L scores).
  Finance     : finance-alpaca rows where both question and answer are
                non-trivial (question > 20 chars, answer > 50 chars).
                Sample 50 uniformly at random (seed=42).
  General QA  : SQuAD — filter rows with answers longer than 5 tokens.
                Sample 50 uniformly at random (seed=42).

The 150 queries are saved to knowledge_base/queries/benign_queries_150.json
"""

import os
import json
import random
from tqdm import tqdm

CORPORA_DIR = "knowledge_base/corpora"
QUERIES_DIR = "knowledge_base/queries"
RANDOM_SEED = 42
QUERIES_PER_DOMAIN = 50

os.makedirs(QUERIES_DIR, exist_ok=True)
random.seed(RANDOM_SEED)


def build_healthcare_queries(records: list[dict]) -> list[dict]:
    """
    From PubMedQA labeled: pick 50 with yes/no final_decision
    (clear ground truth), sorted by answer length descending.
    """
    # Filter: has question, answer, and decisive final_decision
    candidates = [
        r for r in records
        if r.get("question")
        and r.get("answer")
        and r.get("final_decision") in ("yes", "no")
        and len(str(r["answer"])) > 50
    ]

    # Sort by answer length — longer = richer ground truth
    candidates.sort(key=lambda r: len(r["answer"]), reverse=True)

    selected = candidates[:QUERIES_PER_DOMAIN]
    queries  = []
    for i, r in enumerate(selected):
        queries.append({
            "query_id":            f"hc_{i:03d}",
            "domain":              "healthcare",
            "question":            r["question"].strip(),
            "ground_truth_answer": r["answer"].strip(),
            "source_doc_id":       r["doc_id"],
            "metadata": {
                "final_decision": r.get("final_decision", ""),
                "source_dataset": r["source_dataset"],
            }
        })
    print(f"    Healthcare: {len(queries):,} queries selected "
          f"(from {len(candidates):,} candidates)")
    return queries


def build_finance_queries(records: list[dict]) -> list[dict]:
    """
    From finance-alpaca QA pairs: filter for non-trivial questions
    and answers, sample 50.
    """
    candidates = [
        r for r in records
        if r.get("question")
        and r.get("answer")
        and r.get("source_dataset") == "gbharti/finance-alpaca"
        and len(str(r["question"])) > 20
        and len(str(r["answer"]))   > 50
        # Exclude questions asking for lists or bullet points
        # (harder to evaluate with ROUGE-L)
        and "list" not in str(r["question"]).lower()[:30]
    ]

    random.shuffle(candidates)
    selected = candidates[:QUERIES_PER_DOMAIN]

    queries = []
    for i, r in enumerate(selected):
        queries.append({
            "query_id":            f"fin_{i:03d}",
            "domain":              "finance",
            "question":            r["question"].strip(),
            "ground_truth_answer": r["answer"].strip(),
            "source_doc_id":       r["doc_id"],
            "metadata": {
                "source_dataset": r["source_dataset"],
            }
        })
    print(f"    Finance   : {len(queries):,} queries selected "
          f"(from {len(candidates):,} candidates)")
    return queries


def build_general_queries(records: list[dict]) -> list[dict]:
    """
    From SQuAD: filter for answers longer than 5 tokens, sample 50.
    """
    candidates = [
        r for r in records
        if r.get("question")
        and r.get("answer")
        and len(r["answer"].split()) > 5
        and len(r["question"].split()) > 4
    ]

    random.shuffle(candidates)
    selected = candidates[:QUERIES_PER_DOMAIN]

    queries = []
    for i, r in enumerate(selected):
        queries.append({
            "query_id":            f"gen_{i:03d}",
            "domain":              "general",
            "question":            r["question"].strip(),
            "ground_truth_answer": r["answer"].strip(),
            "source_doc_id":       r["doc_id"],
            "metadata": {
                "title":          r.get("title", ""),
                "source_dataset": r["source_dataset"],
            }
        })
    print(f"    General   : {len(queries):,} queries selected "
          f"(from {len(candidates):,} candidates)")
    return queries


def main():
    print("=" * 60)
    print("Phase 1B — Step 3: Build 150-Query Benign Test Set")
    print("=" * 60)

    all_queries = []

    builders = [
        ("healthcare", build_healthcare_queries),
        ("finance",    build_finance_queries),
        ("general",    build_general_queries),
    ]

    for domain, builder in builders:
        raw_path = os.path.join(CORPORA_DIR, f"{domain}_raw.json")
        if not os.path.exists(raw_path):
            print(f"\nWARNING: {raw_path} not found — run script 01 first")
            continue

        print(f"\n  {domain.capitalize()}...")
        with open(raw_path) as f:
            records = json.load(f)

        queries = builder(records)
        all_queries.extend(queries)

    out_path = os.path.join(QUERIES_DIR, "benign_queries_150.json")
    with open(out_path, "w") as f:
        json.dump(all_queries, f, indent=2)

    print("\n" + "=" * 60)
    print("QUERY SET SUMMARY")
    print("=" * 60)
    for domain in ["healthcare", "finance", "general"]:
        n = sum(1 for q in all_queries if q["domain"] == domain)
        print(f"  {domain:<15}: {n:,} queries")
    print(f"  {'TOTAL':<15}: {len(all_queries):,} queries")
    print(f"\nSaved → {out_path}")
    print(f"Random seed: {RANDOM_SEED} (document in paper)")


if __name__ == "__main__":
    main()