# knowledge_base_prep/05_build_poisoned_dbs.py
"""
Phase 1B — Step 5: Build three poisoned ChromaDB instances.

Takes the clean ChromaDB and creates three variants by injecting
poisoned documents at controlled contamination rates:
  - 1%  contamination  → chromadb/poisoned_1pct/
  - 5%  contamination  → chromadb/poisoned_5pct/
  - 10% contamination  → chromadb/poisoned_10pct/

Contamination rate = poisoned_docs / (clean_docs + poisoned_docs).

Since the catalog has only 63 unique attack documents, higher
contamination rates are achieved by injecting multiple copies of
each document with unique IDs. This is standard practice in the
RAG poisoning literature (PoisonedRAG, 2024) and reflects a
realistic scenario where an attacker seeds the same malicious
instruction across many documents in the knowledge base to
increase retrieval probability.
"""

import os
import json
import copy
import math
import random
import shutil
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CLEAN_CHROMADB_DIR = "chromadb/clean"
POISONED_CATALOG   = "knowledge_base/poisoned/poisoned_docs_catalog.json"
POISONED_RATES_DIR = "knowledge_base/poisoned/poisoned_docs_by_rate"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME    = "securerag_clean"
BATCH_SIZE         = 100
RANDOM_SEED        = 42

random.seed(RANDOM_SEED)

CONTAMINATION_RATES = {
    "1pct":  0.01,
    "5pct":  0.05,
    "10pct": 0.10,
}


def get_clean_collection_count() -> int:
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CLEAN_CHROMADB_DIR)
    col    = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    return col.count()


def copy_clean_db(target_dir: str):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(CLEAN_CHROMADB_DIR, target_dir)
    print(f"    Copied clean DB → {target_dir}")


def select_poisoned_subset(catalog: list[dict],
                           n_inject: int) -> list[dict]:
    """
    Build a list of exactly n_inject poisoned document dicts.

    If n_inject <= catalog size: sample proportionally across
    domain x injection_type groups (no repetition).

    If n_inject > catalog size: repeat the full catalog in
    round-robin order, assigning unique doc_ids to each copy
    so ChromaDB does not reject them as duplicates.
    This is the multi-copy injection strategy used in PoisonedRAG.
    """
    if n_inject <= len(catalog):
        # Proportional sampling — no copies needed
        groups: dict[str, list] = {}
        for doc in catalog:
            key = f"{doc['domain']}_{doc['injection_type']}"
            groups.setdefault(key, []).append(doc)

        n_groups     = len(groups)
        base_per_grp = max(1, n_inject // n_groups)
        selected     = []
        for docs in groups.values():
            take = min(base_per_grp, len(docs))
            selected.extend(random.sample(docs, take))

        # Pad to exactly n_inject if rounding left us short
        if len(selected) < n_inject:
            already = {id(d) for d in selected}
            rest    = [d for d in catalog if id(d) not in already]
            need    = n_inject - len(selected)
            selected.extend(random.sample(rest, min(need, len(rest))))

        return selected[:n_inject]

    # n_inject > catalog size: multi-copy injection
    result   = []
    copy_num = 0
    while len(result) < n_inject:
        for doc in catalog:
            if len(result) >= n_inject:
                break
            new_doc           = copy.deepcopy(doc)
            new_doc["doc_id"] = f"{doc['doc_id']}_c{copy_num:05d}"
            new_doc["metadata"] = dict(doc["metadata"])
            new_doc["metadata"]["doc_id"]   = new_doc["doc_id"]
            new_doc["metadata"]["copy_num"] = str(copy_num)
            result.append(new_doc)
        copy_num += 1
    return result


def inject_poisoned_docs(target_dir: str,
                         poisoned_subset: list[dict],
                         rate_label: str) -> int:
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=target_dir)
    col    = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    count_before = col.count()
    ids       = [doc["doc_id"]       for doc in poisoned_subset]
    documents = [doc["poisoned_text"] for doc in poisoned_subset]
    metadatas = [doc["metadata"]      for doc in poisoned_subset]

    for i in tqdm(range(0, len(ids), BATCH_SIZE),
                  desc=f"    Injecting {rate_label}"):
        end = min(i + BATCH_SIZE, len(ids))
        col.add(
            ids       = ids[i:end],
            documents = documents[i:end],
            metadatas = metadatas[i:end],
        )

    count_after = col.count()
    print(f"    {rate_label}: {count_before:,} → {count_after:,} chunks "
          f"(+{count_after - count_before:,} poisoned)")
    return count_after


def main():
    print("=" * 60)
    print("Phase 1B — Step 5: Build Poisoned ChromaDB Instances")
    print("=" * 60)

    if not os.path.exists(POISONED_CATALOG):
        print("ERROR: Poisoned catalog not found — run script 04 first")
        return

    with open(POISONED_CATALOG) as f:
        catalog = json.load(f)
    print(f"\nLoaded {len(catalog):,} unique poisoned documents from catalog")

    clean_count = get_clean_collection_count()
    print(f"Clean DB chunk count: {clean_count:,}")

    os.makedirs(POISONED_RATES_DIR, exist_ok=True)
    summary = {"clean_count": clean_count, "poisoned_variants": {}}

    for rate_label, rate in CONTAMINATION_RATES.items():
        print(f"\n{'─'*50}")
        print(f"Building {rate_label} ({rate*100:.0f}% contamination)...")

        # n_inject = how many docs needed to reach target rate
        # rate = n_inject / (clean_count + n_inject)
        # → n_inject = rate * clean_count / (1 - rate)
        n_inject  = math.ceil(rate * clean_count / (1 - rate))
        n_unique  = len(catalog)
        n_copies  = math.ceil(n_inject / n_unique)

        print(f"  Target injection count : {n_inject:,}")
        print(f"  Unique attack docs     : {n_unique}")
        print(f"  Copies per unique doc  : ~{n_copies}")

        subset = select_poisoned_subset(catalog, n_inject)

        # Save the subset for reproducibility
        subset_path = os.path.join(POISONED_RATES_DIR,
                                   f"poisoned_{rate_label}.json")
        with open(subset_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"  Saved {len(subset):,} docs → {subset_path}")

        # Copy clean DB and inject
        target_dir  = f"chromadb/poisoned_{rate_label}"
        copy_clean_db(target_dir)
        total_after = inject_poisoned_docs(target_dir, subset, rate_label)
        actual_rate = len(subset) / total_after * 100

        summary["poisoned_variants"][rate_label] = {
            "n_injected":    len(subset),
            "n_unique_docs": n_unique,
            "total_chunks":  total_after,
            "actual_rate":   round(actual_rate, 2),
            "target_rate":   round(rate * 100, 1),
            "chromadb_path": target_dir,
        }

        # Breakdown
        for inj_type in ["I1", "I2", "I3"]:
            n = sum(1 for d in subset
                    if d["metadata"].get("injection_type") == inj_type)
            print(f"    {inj_type}: {n:,} injected (incl. copies)")
        for domain in ["healthcare", "finance", "general"]:
            n = sum(1 for d in subset
                    if d["metadata"].get("domain") == domain)
            print(f"    {domain}: {n:,} injected (incl. copies)")

    # Save summary
    summary_path = "knowledge_base/poisoned/poisoning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("POISONING SUMMARY")
    print("=" * 60)
    print(f"  Clean DB    : {clean_count:,} chunks")
    for label, info in summary["poisoned_variants"].items():
        print(f"  {label:<8} : {info['n_injected']:,} injected "
              f"({info['actual_rate']:.2f}% actual | "
              f"target {info['target_rate']}%) "
              f"→ {info['total_chunks']:,} total")
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()