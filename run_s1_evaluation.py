# run_s1_evaluation.py
"""
S1 evaluation: RuleDefendedRAG (RuleClassifier only).

Generator: Ollama llama3:latest.
Classifier: RuleClassifier (not DeBERTa, not SimpleMLDetector).

Output (results/s1/):
    S1_attack_results.jsonl
    S1_attack_results.csv
    S1_asr_by_category.csv
    S1_asr_summary.json
    S1_evaluation_report.txt
"""

import os
import sys
import json
from datetime import datetime
from urllib import request as urllib_request
from urllib import error as urllib_error

import config
from rag.defended_rag import RuleDefendedRAG
from evaluation.attack_runner import AttackRunner
from evaluation.results_reporter import run_full_report

SYSTEM_NAME = "S1"
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "s1")


def load_attack_suite() -> list[dict]:
    if not os.path.exists(config.ATTACK_SUITE_PATH):
        print(f"ERROR: Attack suite not found at {config.ATTACK_SUITE_PATH}")
        sys.exit(1)
    with open(config.ATTACK_SUITE_PATH) as f:
        return json.load(f)


def preflight_checks():
    print("\n[1/3] Checking ChromaDB (clean)...")
    if not os.path.exists(config.CLEAN_DB_PATH):
        print(f"ERROR: {config.CLEAN_DB_PATH} not found.")
        sys.exit(1)
    print(f"  {config.CLEAN_DB_PATH} ✓")

    print("\n[2/3] Checking Ollama (llama3:latest)...")
    try:
        urllib_request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except (urllib_error.URLError, TimeoutError, OSError) as e:
        print(f"ERROR: Ollama not reachable at localhost:11434 ({e})")
        sys.exit(1)
    print("  Ollama reachable ✓")
    print("  Model    : llama3:latest")
    print(f"  Temp     : {config.LLM_TEMPERATURE} (not sent to Ollama)")
    print(f"  Delay    : {config.API_RATE_LIMIT_DELAY}s between calls")


def main():
    print("=" * 65)
    print("SecureRAG — S1 RuleClassifier Evaluation (llama3:latest)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    preflight_checks()

    print("\n[3/3] Loading attack suite...")
    attacks = load_attack_suite()
    n_direct = sum(1 for a in attacks if a["surface"] == "direct")
    n_indirect = sum(1 for a in attacks if a["surface"] == "indirect")
    n_multi = sum(1 for a in attacks if a.get("is_multi_turn"))
    print(f"  Total: {len(attacks)} | Direct: {n_direct} | "
          f"Indirect: {n_indirect} | Multi-turn: {n_multi}")

    print(f"\nInitialising S1 RAG (clean DB: {config.CLEAN_DB_PATH})...")
    rag = RuleDefendedRAG(db_path=config.CLEAN_DB_PATH)
    print("  Classifier: RuleClassifier only")
    print("  RAG system ready.")

    print(f"\nResults dir: {RESULTS_DIR}")
    print("Incremental save: yes (resume-safe)")
    print()

    runner = AttackRunner(
        rag=rag,
        system_name=SYSTEM_NAME,
        results_dir=RESULTS_DIR,
    )
    results = runner.run(attacks)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    asr_data = run_full_report(results, SYSTEM_NAME, RESULTS_DIR)

    ov = asr_data["overall"]
    direct_asr = asr_data["by_surface"].get("direct", {}).get("asr", 0)
    indirect_asr = asr_data["by_surface"].get("indirect", {}).get("asr", 0)

    print("\n" + "=" * 65)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"\n  Overall ASR  : {ov['asr']*100:.1f}% "
          f"({ov['n_success']}/{ov['n_attacks']})")
    print(f"  Direct ASR   : {direct_asr*100:.1f}%")
    print(f"  Indirect ASR : {indirect_asr*100:.1f}%")
    print(f"\n  All outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
