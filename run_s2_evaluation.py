# run_s2_evaluation.py
"""
S2 (ML-Based Defence, DeBERTa) evaluation entry point.

Usage:
    export GEMINI_API_KEY="AIza..."
    python run_s2_evaluation.py
"""

import os, sys, json
from datetime import datetime
import config
from rag.defended_rag import MLDefendedRAG
from evaluation.attack_runner import AttackRunner
from evaluation.results_reporter import run_full_report

SYSTEM_NAME = "S2"
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "s2")


def main():
    print("=" * 65)
    print("SecureRAG — S2 ML-Based Defence Evaluation (DeBERTa)")
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

    print("Initialising S2 RAG system (loading DeBERTa model)...")
    rag = MLDefendedRAG(db_path=config.CLEAN_DB_PATH)
    print("  S2 ready.")

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
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()