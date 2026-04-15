# run_s3_evaluation.py
"""
S3 (Hybrid Defence) evaluation entry point.

Usage:
    export GEMINI_API_KEY="AIza..."
    python run_s3_evaluation.py
"""

import os, sys, json
from datetime import datetime
import config
from rag.defended_rag import HybridDefendedRAG
from evaluation.attack_runner import AttackRunner
from evaluation.results_reporter import run_full_report

SYSTEM_NAME = "S3"
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "s3")


def main():
    print("=" * 65)
    print("SecureRAG — S3 Hybrid Defence Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    if not os.path.exists(config.ATTACK_SUITE_PATH):
        print("ERROR: Attack suite not found. Run Phase 1C first.")
        sys.exit(1)
    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    with open(config.ATTACK_SUITE_PATH) as f:
        attacks = json.load(f)
    print(f"Loaded {len(attacks)} attacks.")

    print("Initialising S3 RAG system (loading Rule + DeBERTa)...")
    rag = HybridDefendedRAG(db_path=config.CLEAN_DB_PATH)
    print("  S3 ready.")

    runner  = AttackRunner(rag=rag, system_name=SYSTEM_NAME,
                           results_dir=RESULTS_DIR)
    results = runner.run(attacks)

    print("\n" + "=" * 65)
    asr_data = run_full_report(results, SYSTEM_NAME, RESULTS_DIR)

    ov = asr_data["overall"]
    print(f"\n  Overall ASR  : {ov['asr']*100:.1f}%")
    print(f"  Direct ASR   : "
          f"{asr_data['by_surface'].get('direct',{}).get('asr',0)*100:.1f}%")
    print(f"  Indirect ASR : "
          f"{asr_data['by_surface'].get('indirect',{}).get('asr',0)*100:.1f}%")
    print(f"  Fast-path rate: {rag.fast_path_rate*100:.1f}% "
          f"(attacks caught by rules without ML)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()