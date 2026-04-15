# knowledge_base_prep/06_verify_retrieval_baseline.py
"""
Phase 1B — Step 6: Verify retrieval baseline on the clean knowledge base.

For each of the 150 benign queries, checks whether the source document
appears in the top-5 retrieved chunks. This establishes:

  1. That the RAG system is functional before any attack or defense
     is applied (a reviewer requirement).
  2. The baseline retrieval recall per domain.
  3. That the KB is correctly indexed and semantically searchable.

Metrics reported:
  - Top-1 recall  : source doc in position 1
  - Top-3 recall  : source doc in positions 1–3
  - Top-5 recall  : source doc in positions 1–5
  - Mean rank     : average position of source doc when found
  - Domain breakdown of all three metrics

A Top-5 recall >= 0.60 is considered sufficient to proceed.
If recall is lower, the corpus may need more documents.

Output: knowledge_base/queries/retrieval_baseline_report.json
"""

import os
import json
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CLEAN_CHROMADB_DIR = "chromadb/clean"
QUERIES_PATH       = "knowledge_base/queries/benign_queries_150.json"
REPORT_PATH        = "knowledge_base/queries/retrieval_baseline_report.json"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME    = "securerag_clean"
TOP_K              = 5


def run_retrieval_baseline():
    print("=" * 60)
    print("Phase 1B — Step 6: Verify Retrieval Baseline")
    print("=" * 60)

    if not os.path.exists(QUERIES_PATH):
        print("ERROR: benign_queries_150.json not found — run script 03 first")
        return

    with open(QUERIES_PATH) as f:
        queries = json.load(f)
    print(f"\nLoaded {len(queries):,} benign queries")

    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CLEAN_CHROMADB_DIR)
    col    = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    print(f"Connected to clean ChromaDB: {col.count():,} chunks")

    results_per_query = []
    domain_stats = {
        "healthcare": {"top1": 0, "top3": 0, "top5": 0, "ranks": [], "total": 0},
        "finance":    {"top1": 0, "top3": 0, "top5": 0, "ranks": [], "total": 0},
        "general":    {"top1": 0, "top3": 0, "top5": 0, "ranks": [], "total": 0},
    }

    print(f"\nRunning retrieval for {len(queries):,} queries (top-k={TOP_K})...")

    for query in tqdm(queries, desc="  Querying"):
        domain     = query["domain"]
        question   = query["question"]
        src_doc_id = query["source_doc_id"]

        # Retrieve top-k chunks
        result = col.query(
            query_texts=[question],
            n_results=TOP_K,
            include=["metadatas", "distances"],
        )

        retrieved_doc_ids = [
            m["doc_id"] for m in result["metadatas"][0]
        ]
        distances = result["distances"][0]

        # Check if source doc appears in retrieved results
        found_rank = None
        for rank_0based, doc_id in enumerate(retrieved_doc_ids):
            if doc_id == src_doc_id:
                found_rank = rank_0based + 1  # 1-indexed
                break

        in_top1 = found_rank == 1
        in_top3 = found_rank is not None and found_rank <= 3
        in_top5 = found_rank is not None

        domain_stats[domain]["total"] += 1
        domain_stats[domain]["top1"]  += int(in_top1)
        domain_stats[domain]["top3"]  += int(in_top3)
        domain_stats[domain]["top5"]  += int(in_top5)
        if found_rank:
            domain_stats[domain]["ranks"].append(found_rank)

        results_per_query.append({
            "query_id":           query["query_id"],
            "domain":             domain,
            "question":           question[:100],
            "source_doc_id":      src_doc_id,
            "retrieved_doc_ids":  retrieved_doc_ids,
            "found_rank":         found_rank,
            "in_top1":            in_top1,
            "in_top3":            in_top3,
            "in_top5":            in_top5,
            "top1_distance":      round(distances[0], 4) if distances else None,
        })

    # Aggregate metrics
    total = len(queries)
    overall = {
        "top1_recall": sum(r["in_top1"] for r in results_per_query) / total,
        "top3_recall": sum(r["in_top3"] for r in results_per_query) / total,
        "top5_recall": sum(r["in_top5"] for r in results_per_query) / total,
    }
    found_ranks = [r["found_rank"] for r in results_per_query if r["found_rank"]]
    overall["mean_rank_when_found"] = (
        round(sum(found_ranks) / len(found_ranks), 2) if found_ranks else None
    )

    domain_summary = {}
    for domain, stats in domain_stats.items():
        n = stats["total"]
        domain_summary[domain] = {
            "total":     n,
            "top1_recall": round(stats["top1"] / n, 3) if n else 0,
            "top3_recall": round(stats["top3"] / n, 3) if n else 0,
            "top5_recall": round(stats["top5"] / n, 3) if n else 0,
            "mean_rank":   round(
                sum(stats["ranks"]) / len(stats["ranks"]), 2
            ) if stats["ranks"] else None,
        }

    report = {
        "overall":          overall,
        "by_domain":        domain_summary,
        "total_queries":    total,
        "top_k":            TOP_K,
        "embed_model":      EMBED_MODEL,
        "per_query_results": results_per_query,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RETRIEVAL BASELINE REPORT")
    print("=" * 60)
    print(f"\n  Overall (n={total}):")
    print(f"    Top-1 recall : {overall['top1_recall']:.3f}")
    print(f"    Top-3 recall : {overall['top3_recall']:.3f}")
    print(f"    Top-5 recall : {overall['top5_recall']:.3f}")
    if overall["mean_rank_when_found"]:
        print(f"    Mean rank    : {overall['mean_rank_when_found']:.2f}")

    print(f"\n  By domain:")
    for domain, stats in domain_summary.items():
        print(f"    {domain:<15} top5={stats['top5_recall']:.3f}  "
              f"top3={stats['top3_recall']:.3f}  top1={stats['top1_recall']:.3f}")

    threshold = 0.60
    if overall["top5_recall"] >= threshold:
        print(f"\n  STATUS: PASS — Top-5 recall {overall['top5_recall']:.3f} "
              f">= {threshold}")
        print("  The clean knowledge base is ready for attack evaluation.")
    else:
        print(f"\n  STATUS: WARNING — Top-5 recall {overall['top5_recall']:.3f} "
              f"< {threshold}")
        print("  Consider adding more domain documents to improve retrieval quality.")

    print(f"\nFull report saved → {REPORT_PATH}")


if __name__ == "__main__":
    run_retrieval_baseline()