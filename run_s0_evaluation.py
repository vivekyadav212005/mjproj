# run_s0_evaluation.py
"""
Main entry point for S0 (Baseline RAG) evaluation.

Runs all 160 attacks against the undefended S0 system and produces
ASR tables for Section 4 of the paper.

Setup:
    pip install -r requirements_phase2.txt
    export GEMINI_API_KEY="AIza..."     # from https://aistudio.google.com
    python run_s0_evaluation.py

Rate limit note:
    Gemini free tier = 10 RPM, 250 RPD.
    config.API_RATE_LIMIT_DELAY = 6.5s → ~9 RPM (safe).
    160 attacks takes ~18 min for API calls + LLM response time.
    If you hit 250 RPD limit, run remaining attacks next day —
    the script is resume-safe (skips already-completed attack IDs).

Output (results/s0/):
    S0_attack_results.jsonl      — per-attack results, one JSON per line
    S0_asr_by_category.csv       — Table 2 data for the paper
    S0_asr_summary.json          — complete ASR metrics
    S0_evaluation_report.txt     — paper-ready text tables
"""

import os
import sys
import json
from datetime import datetime

import config
from rag.baseline_rag import BaselineRAG
from evaluation.attack_runner import AttackRunner
from evaluation.results_reporter import run_full_report

SYSTEM_NAME = "S0"
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "s0")


def load_attack_suite() -> list[dict]:
    if not os.path.exists(config.ATTACK_SUITE_PATH):
        print(f"ERROR: Attack suite not found at {config.ATTACK_SUITE_PATH}")
        print("Run Phase 1C scripts first:")
        print("  python attack_suite_prep/01_build_direct_attacks.py")
        print("  python attack_suite_prep/02_build_indirect_attacks.py")
        print("  python attack_suite_prep/03_validate_and_export.py")
        sys.exit(1)
    with open(config.ATTACK_SUITE_PATH) as f:
        attacks = json.load(f)
    return attacks


def preflight_checks():
    print("\n[1/4] Checking ChromaDB (clean)...")
    if not os.path.exists(config.CLEAN_DB_PATH):
        print(f"ERROR: {config.CLEAN_DB_PATH} not found.")
        print("Run Phase 1B scripts first.")
        sys.exit(1)
    print(f"  {config.CLEAN_DB_PATH} ✓")

    print(f"\n[2/4] Checking Gemini API key...")
    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set.")
        print("1. Get a free key at https://aistudio.google.com")
        print("2. export GEMINI_API_KEY='AIza...'")
        sys.exit(1)
    masked = config.GEMINI_API_KEY[:8] + "..." + config.GEMINI_API_KEY[-4:]
    print(f"  Key found: {masked} ✓")
    print(f"  Model    : {config.GEMINI_MODEL}")
    print(f"  Temp     : {config.LLM_TEMPERATURE}")
    print(f"  Delay    : {config.API_RATE_LIMIT_DELAY}s between calls")

    print(f"\n[3/4] Importing Gemini SDK...")
    try:
        from google import genai
        print(f"  google-genai {genai.__version__} ✓")
    except ImportError:
        print("ERROR: google-genai not installed.")
        print("Run: pip install \"google-genai>=1.0.0\"")
        sys.exit(1)


def main():
    print("=" * 65)
    print("SecureRAG — S0 Baseline Evaluation (Gemini 2.5 Flash)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    preflight_checks()

    print(f"\n[4/4] Loading attack suite...")
    attacks    = load_attack_suite()
    n_direct   = sum(1 for a in attacks if a["surface"] == "direct")
    n_indirect = sum(1 for a in attacks if a["surface"] == "indirect")
    n_multi    = sum(1 for a in attacks if a.get("is_multi_turn"))
    print(f"  Total: {len(attacks)} | Direct: {n_direct} | "
          f"Indirect: {n_indirect} | Multi-turn: {n_multi}")

    print(f"\nInitialising S0 RAG (clean DB: {config.CLEAN_DB_PATH})...")
    rag = BaselineRAG(db_path=config.CLEAN_DB_PATH)
    print("  RAG system ready.")

    print(f"\nResults dir: {RESULTS_DIR}")
    print("Incremental save: yes (resume-safe)")
    print()

    runner  = AttackRunner(
        rag         = rag,
        system_name = SYSTEM_NAME,
        results_dir = RESULTS_DIR,
    )
    results = runner.run(attacks)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    asr_data = run_full_report(results, SYSTEM_NAME, RESULTS_DIR)

    ov           = asr_data["overall"]
    direct_asr   = asr_data["by_surface"].get("direct",   {}).get("asr", 0)
    indirect_asr = asr_data["by_surface"].get("indirect", {}).get("asr", 0)

    print("\n" + "=" * 65)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"\n  Overall ASR  : {ov['asr']*100:.1f}% "
          f"({ov['n_success']}/{ov['n_attacks']})")
    print(f"  Direct ASR   : {direct_asr*100:.1f}%")
    print(f"  Indirect ASR : {indirect_asr*100:.1f}%")
    print(f"  Safety blocks: {ov['n_blocked']} "
          f"({ov['blocked_rate']*100:.1f}%)")
    print(f"\n  All outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()