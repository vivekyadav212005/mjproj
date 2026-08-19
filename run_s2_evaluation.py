# run_s2_evaluation.py
"""
S2 evaluation: MLDefendedRAG (SimpleMLDetector / TF-IDF + logistic regression).

Generator: Ollama llama3:latest.
Classifier: SimpleMLDetector. This is NOT DeBERTa.

Output (results/s2/):
    S2_attack_results.jsonl
    S2_attack_results.csv
    S2_asr_by_category.csv
    S2_asr_summary.json
    S2_evaluation_report.txt
"""

import os
import sys
import json
from datetime import datetime
from urllib import request as urllib_request
from urllib import error as urllib_error

import config
from rag.defended_rag import MLDefendedRAG
from evaluation.attack_runner import AttackRunner
from evaluation.results_reporter import run_full_report

SYSTEM_NAME = "S2"
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "s2")


def main():
    print("=" * 65)
    print("SecureRAG — S2 SimpleMLDetector Evaluation (llama3:latest)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    if not os.path.exists(config.ATTACK_SUITE_PATH):
        print("ERROR: Attack suite not found. Run Phase 1C first.")
        sys.exit(1)
    if not os.path.exists(config.CLEAN_DB_PATH):
        print(f"ERROR: {config.CLEAN_DB_PATH} not found.")
        sys.exit(1)
    s2_model = os.path.join("security", "training", "s2_tfidf_lr.joblib")
    if not os.path.exists(s2_model):
        print(f"ERROR: {s2_model} not found.")
        print("Run: python security/training/prepare_s2_training_data.py")
        print(" then: python security/training/train_simple_ml.py")
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
    print("Classifier: SimpleMLDetector (TF-IDF + logistic regression), not DeBERTa")

    print("Initialising S2 RAG system...")
    rag = MLDefendedRAG(db_path=config.CLEAN_DB_PATH)
    print("  S2 ready.")

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
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
