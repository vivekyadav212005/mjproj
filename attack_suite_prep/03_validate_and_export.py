# attack_suite_prep/03_validate_and_export.py
"""
Phase 1C — Step 3: Validate all attacks and produce final exports.

Loads all 8 category files, validates every attack against the
unified schema, checks for ID uniqueness and completeness, then
produces two final outputs:

  attack_suite/attack_suite_full.json  — all 160 attacks combined
  attack_suite/attack_suite_report.txt — human-readable summary
                                         ready for the paper appendix

Validation rules (all must pass before export):
  V1 — All required schema fields present in every attack
  V2 — All attack_ids are unique across the full suite
  V3 — Each category has exactly the expected count
  V4 — success_keyword is non-empty for every attack
  V5 — Multi-turn attacks have at least 2 turns
  V6 — Indirect attacks have a non-empty trigger_query
  V7 — All surface values are "direct" or "indirect"
  V8 — No attack text is empty (except multi-turn which use turns[])
"""

import os
import json

DIRECT_DIR   = "attack_suite/direct"
INDIRECT_DIR = "attack_suite/indirect"
OUTPUT_DIR   = "attack_suite"

REQUIRED_FIELDS = [
    "attack_id", "surface", "category", "category_name",
    "text", "is_multi_turn", "turns", "success_fn_type",
    "success_keyword", "success_description", "domain",
    "trigger_query", "notes",
]

EXPECTED_COUNTS = {
    "D1": 20, "D2": 20, "D3": 20, "D4": 20, "D5": 20,
    "I1": 20, "I2": 20, "I3": 20,
}

CATEGORY_FILES = {
    "D1": os.path.join(DIRECT_DIR,   "D1_instruction_override.json"),
    "D2": os.path.join(DIRECT_DIR,   "D2_role_hijacking.json"),
    "D3": os.path.join(DIRECT_DIR,   "D3_obfuscated.json"),
    "D4": os.path.join(DIRECT_DIR,   "D4_prompt_leakage.json"),
    "D5": os.path.join(DIRECT_DIR,   "D5_multi_turn.json"),
    "I1": os.path.join(INDIRECT_DIR, "I1_explicit_instruction.json"),
    "I2": os.path.join(INDIRECT_DIR, "I2_hidden_instruction.json"),
    "I3": os.path.join(INDIRECT_DIR, "I3_persona_shift.json"),
}


def load_all() -> dict[str, list[dict]]:
    loaded = {}
    for cat, path in CATEGORY_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path} — run scripts 01 and 02 first"
            )
        with open(path) as f:
            loaded[cat] = json.load(f)
        print(f"  Loaded {len(loaded[cat]):>3} attacks ← {path}")
    return loaded


def validate(attacks_by_cat: dict[str, list[dict]]) -> list[str]:
    """Run all validation rules. Returns list of error strings."""
    errors  = []
    all_ids = []
    all_attacks = [a for attacks in attacks_by_cat.values() for a in attacks]

    # V1 — required fields
    for a in all_attacks:
        missing = [f for f in REQUIRED_FIELDS if f not in a]
        if missing:
            errors.append(
                f"V1 [{a.get('attack_id','?')}]: missing fields {missing}"
            )

    # V2 — unique IDs
    all_ids = [a["attack_id"] for a in all_attacks]
    seen    = set()
    for aid in all_ids:
        if aid in seen:
            errors.append(f"V2: duplicate attack_id '{aid}'")
        seen.add(aid)

    # V3 — correct counts per category
    for cat, expected in EXPECTED_COUNTS.items():
        actual = len(attacks_by_cat.get(cat, []))
        if actual != expected:
            errors.append(
                f"V3 [{cat}]: expected {expected} attacks, got {actual}"
            )

    # V4 — non-empty success_keyword
    for a in all_attacks:
        if not str(a.get("success_keyword", "")).strip():
            errors.append(
                f"V4 [{a.get('attack_id','?')}]: empty success_keyword"
            )

    # V5 — multi-turn attacks have >= 2 turns
    for a in all_attacks:
        if a.get("is_multi_turn") and len(a.get("turns", [])) < 2:
            errors.append(
                f"V5 [{a['attack_id']}]: is_multi_turn=True "
                f"but only {len(a.get('turns',[]))} turns"
            )

    # V6 — indirect attacks have trigger_query
    for a in all_attacks:
        if a.get("surface") == "indirect":
            if not a.get("trigger_query", "").strip():
                errors.append(
                    f"V6 [{a['attack_id']}]: indirect attack "
                    f"missing trigger_query"
                )

    # V7 — valid surface values
    for a in all_attacks:
        if a.get("surface") not in ("direct", "indirect"):
            errors.append(
                f"V7 [{a['attack_id']}]: invalid surface "
                f"'{a.get('surface')}'"
            )

    # V8 — non-empty text for non-multi-turn attacks
    for a in all_attacks:
        if not a.get("is_multi_turn") and not str(a.get("text", "")).strip():
            errors.append(
                f"V8 [{a['attack_id']}]: empty text for "
                f"single-turn attack"
            )

    return errors


def generate_report(attacks_by_cat: dict[str, list[dict]]) -> str:
    all_attacks = [a for attacks in attacks_by_cat.values() for a in attacks]
    total       = len(all_attacks)
    n_direct    = sum(1 for a in all_attacks if a["surface"] == "direct")
    n_indirect  = sum(1 for a in all_attacks if a["surface"] == "indirect")
    n_multi     = sum(1 for a in all_attacks if a.get("is_multi_turn"))

    # Collect unique success keywords
    keywords = {}
    for a in all_attacks:
        k = a["success_keyword"]
        keywords[k] = keywords.get(k, 0) + 1

    lines = [
        "=" * 70,
        "SECURERAG — ATTACK EVALUATION SUITE REPORT",
        "Phase 1C: Complete Attack Suite",
        "=" * 70,
        "",
        "TABLE 6: Attack Suite Overview",
        "-" * 70,
        f"  Total attacks          : {total}",
        f"  Surface 1 (direct)     : {n_direct}",
        f"  Surface 2 (indirect)   : {n_indirect}",
        f"  Multi-turn attacks     : {n_multi} (within D5)",
        "",
        "TABLE 7: Attacks by Category",
        "-" * 70,
        f"  {'Category':<8} {'Name':<35} {'Count':>6} {'Surface':<10}",
        "-" * 70,
    ]

    cat_meta = {
        "D1": ("instruction_override",       "direct"),
        "D2": ("role_hijacking",             "direct"),
        "D3": ("obfuscated_injection",       "direct"),
        "D4": ("prompt_leakage",             "direct"),
        "D5": ("multi_turn_anchoring",       "direct"),
        "I1": ("explicit_instruction_in_doc","indirect"),
        "I2": ("hidden_instruction_in_doc",  "indirect"),
        "I3": ("persona_shift_via_document", "indirect"),
    }

    for cat, (name, surface) in cat_meta.items():
        n = len(attacks_by_cat.get(cat, []))
        lines.append(f"  {cat:<8} {name:<35} {n:>6} {surface:<10}")

    lines += [
        "-" * 70,
        f"  {'TOTAL':<44} {total:>6}",
        "",
        "TABLE 8: Success Keywords",
        "-" * 70,
        f"  {'Keyword':<30} {'Occurrences':>12}",
        "-" * 70,
    ]

    for kw, cnt in sorted(keywords.items(), key=lambda x: -x[1]):
        lines.append(f"  {kw:<30} {cnt:>12}")

    lines += [
        "",
        "TABLE 9: Indirect Attack Domain Distribution",
        "-" * 70,
    ]

    indirect = [a for a in all_attacks if a["surface"] == "indirect"]
    for domain in ["healthcare", "finance", "general"]:
        n = sum(1 for a in indirect if a["domain"] == domain)
        for cat in ["I1", "I2", "I3"]:
            nc = sum(1 for a in indirect
                     if a["domain"] == domain
                     and a["category"] == cat)
            lines.append(
                f"  {domain:<15} {cat:<5} : {nc} attacks"
            )

    lines += [
        "",
        "DESIGN DECISIONS (for paper methodology section):",
        "-" * 70,
        "  1. All attacks hand-authored — no overlap with Phase 1A",
        "     classifier training data (strict train/eval separation).",
        "  2. Success criteria are exclusively keyword-based for",
        "     determinism and reproducibility without secondary models.",
        "  3. D5 multi-turn: 5 single-turn controls + 15 two-turn attacks.",
        "     The PLACEHOLDER in Turn 2 is replaced by the actual LLM",
        "     response during evaluation.",
        "  4. D4 success keyword SECURERAG_SYSTEM must be embedded in the",
        "     actual RAG system prompt used during evaluation.",
        "  5. Indirect attack camouflage texts are semantically aligned",
        "     with their trigger queries to ensure realistic retrieval.",
        "  6. Obfuscation techniques in D3 are chosen to specifically",
        "     test the gap between rule-based and ML-based classification.",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Phase 1C — Step 3: Validate and Export Attack Suite")
    print("=" * 60)

    print("\nLoading all category files...")
    attacks_by_cat = load_all()

    print("\nRunning validation...")
    errors = validate(attacks_by_cat)

    if errors:
        print(f"\n  VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    ✗ {e}")
        print("\n  Fix the errors above before exporting.")
        return

    print(f"  All validation rules passed (V1–V8)")

    # Combine all attacks into one ordered list
    all_attacks = []
    for cat in ["D1", "D2", "D3", "D4", "D5", "I1", "I2", "I3"]:
        all_attacks.extend(attacks_by_cat[cat])

    # Save combined suite
    full_path = os.path.join(OUTPUT_DIR, "attack_suite_full.json")
    with open(full_path, "w") as f:
        json.dump(all_attacks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_attacks)} attacks → {full_path}")

    # Generate and save report
    report_text = generate_report(attacks_by_cat)
    report_path = os.path.join(OUTPUT_DIR, "attack_suite_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved report → {report_path}")

    # Print report to terminal
    print("\n" + report_text)


if __name__ == "__main__":
    main()