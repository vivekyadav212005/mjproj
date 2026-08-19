# run_s3_evaluation.py
"""
S3 (fixed) evaluation: HybridDefendedRAG (RuleClassifier + SimpleMLDetector).

Writes to results/s3_fixed/ so the original results/s3/ run is preserved.
Generator: Ollama llama3:latest.
Classifier: hybrid Rule + SimpleMLDetector. This is NOT DeBERTa.

Output (results/s3_fixed/):
    S3_fixed_attack_results.jsonl
    S3_fixed_attack_results.csv
    S3_fixed_asr_by_category.csv
    S3_fixed_asr_summary.json
    S3_fixed_evaluation_report.txt
"""

import os
import sys
import json
from datetime import datetime
from urllib import request as urllib_request
from urllib import error as urllib_error

import config
from rag.defended_rag import HybridDefendedRAG
from evaluation.attack_runner import AttackRunner
from evaluation.results_reporter import run_full_report

SYSTEM_NAME = "S3_fixed"
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "s3_fixed")


def main():
    print("=" * 65)
    print("SecureRAG — S3 Hybrid Defence Evaluation (fixed path, llama3:latest)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    if not os.path.exists(config.ATTACK_SUITE_PATH):
        print("ERROR: Attack suite not found. Run Phase 1C first.")
        sys.exit(1)
    if not os.path.exists(config.CLEAN_DB_PATH):
        print(f"ERROR: {config.CLEAN_DB_PATH} not found.")
        sys.exit(1)
    if not os.path.exists("results/s0/S0_attack_results.jsonl"):
        print("ERROR: results/s0/S0_attack_results.jsonl required to train SimpleMLDetector.")
        sys.exit(1)
    try:
        urllib_request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except (urllib_error.URLError, TimeoutError, OSError) as e:
        print(f"ERROR: Ollama not reachable at localhost:11434 ({e})")
        sys.exit(1)

    with open(config.ATTACK_SUITE_PATH) as f:
        attacks = json.load(f)
    print(f"Loaded {len(attacks)} attacks.")
    print("Generator : Ollama llama3:latest")
    print("Classifier: hybrid RuleClassifier + SimpleMLDetector, not DeBERTa")
    print(f"Results   : {RESULTS_DIR} (original results/s3 left untouched)")

    print("Initialising S3 RAG system...")
    rag = HybridDefendedRAG(db_path=config.CLEAN_DB_PATH)
    print("  S3 ready.")

    runner = AttackRunner(rag=rag, system_name=SYSTEM_NAME,
                          results_dir=RESULTS_DIR)
    results = runner.run(attacks)

    print("\n" + "=" * 65)
    asr_data = run_full_report(results, SYSTEM_NAME, RESULTS_DIR)

    ov = asr_data["overall"]
    print(f"\n  Overall ASR  : {ov['asr']*100:.1f}%")
    print(f"  Direct ASR   : "
          f"{asr_data['by_surface'].get('direct', {}).get('asr', 0)*100:.1f}%")
    print(f"  Indirect ASR : "
          f"{asr_data['by_surface'].get('indirect', {}).get('asr', 0)*100:.1f}%")
    print(f"  Fast-path rate: {rag.fast_path_rate*100:.1f}% "
          f"(attacks caught by rules without ML)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
