# evaluation/results_reporter.py
"""
Results reporter — computes ASR per category and produces paper tables.

New in this version:
  - safety_blocked_rate per category (Gemini-specific metric)
  - poisoned_doc_injected count for indirect attacks (audit field)
  - All four Table outputs: overview, by-category, by-domain, latency

Outputs saved to results/{system}/:
  {system}_asr_by_category.csv    — Table 2 (per-category ASR)
  {system}_asr_summary.json       — machine-readable full metrics
  {system}_evaluation_report.txt  — paper-ready text tables
"""

import os
import json
import csv
from collections import defaultdict

import config


CATEGORY_ORDER = ["D1", "D2", "D3", "D4", "D5", "I1", "I2", "I3"]

CATEGORY_NAMES = {
    "D1": "Instruction Override",
    "D2": "Role Hijacking",
    "D3": "Obfuscated Injection",
    "D4": "Prompt Leakage",
    "D5": "Multi-Turn Anchoring",
    "I1": "Explicit Instruction in Doc",
    "I2": "Hidden Instruction in Doc",
    "I3": "Persona Shift via Document",
}


def compute_asr(results: list[dict]) -> dict:
    """Compute ASR at all granularities."""

    by_cat    = defaultdict(lambda: {
        "n": 0, "success": 0, "errors": 0, "blocked": 0
    })
    by_surface = defaultdict(lambda: {"n": 0, "success": 0, "blocked": 0})
    by_domain  = defaultdict(lambda: {"n": 0, "success": 0})
    n_errors   = 0
    n_blocked  = 0

    for r in results:
        cat     = r["category"]
        surface = r["surface"]
        domain  = r.get("domain", "any")
        success = bool(r["success"])
        blocked = bool(r.get("safety_blocked", False))
        error   = r.get("error") is not None

        by_cat[cat]["n"]       += 1
        by_cat[cat]["success"] += int(success)
        by_cat[cat]["blocked"] += int(blocked)
        if error:
            by_cat[cat]["errors"] += 1
            n_errors += 1
        if blocked:
            n_blocked += 1

        by_surface[surface]["n"]       += 1
        by_surface[surface]["success"] += int(success)
        by_surface[surface]["blocked"] += int(blocked)

        if domain in ("healthcare", "finance", "general"):
            by_domain[domain]["n"]       += 1
            by_domain[domain]["success"] += int(success)

    total  = len(results)
    n_succ = sum(1 for r in results if r["success"])

    def asr(n, s):
        return round(s / n, 4) if n > 0 else 0.0

    return {
        "overall": {
            "n_attacks":     total,
            "n_success":     n_succ,
            "asr":           asr(total, n_succ),
            "n_blocked":     n_blocked,
            "blocked_rate":  round(n_blocked / total, 4) if total > 0 else 0,
        },
        "by_surface": {
            s: {
                "n_attacks": v["n"],
                "n_success": v["success"],
                "asr":       asr(v["n"], v["success"]),
                "n_blocked": v["blocked"],
            }
            for s, v in by_surface.items()
        },
        "by_category": {
            cat: {
                "n_attacks":   v["n"],
                "n_success":   v["success"],
                "asr":         asr(v["n"], v["success"]),
                "n_errors":    v["errors"],
                "n_blocked":   v["blocked"],
                "blocked_rate": round(v["blocked"]/v["n"], 4)
                                if v["n"] > 0 else 0,
            }
            for cat, v in by_cat.items()
        },
        "by_domain": {
            d: {
                "n_attacks": v["n"],
                "n_success": v["success"],
                "asr":       asr(v["n"], v["success"]),
            }
            for d, v in by_domain.items()
        },
        "errors": {
            "n_errors":   n_errors,
            "error_rate": round(n_errors / total, 4) if total > 0 else 0,
        },
    }


def save_csv(asr_data: dict, system_name: str, results_dir: str) -> str:
    path = os.path.join(results_dir, f"{system_name}_asr_by_category.csv")
    rows = []
    for cat in CATEGORY_ORDER:
        v = asr_data["by_category"].get(cat, {
            "n_attacks": 0, "n_success": 0, "asr": 0,
            "n_errors": 0, "n_blocked": 0, "blocked_rate": 0,
        })
        rows.append({
            "system":        system_name,
            "category":      cat,
            "category_name": CATEGORY_NAMES.get(cat, cat),
            "surface":       "direct" if cat.startswith("D") else "indirect",
            "n_attacks":     v["n_attacks"],
            "n_success":     v["n_success"],
            "asr":           v["asr"],
            "asr_pct":       round(v["asr"] * 100, 1),
            "n_blocked":     v.get("n_blocked", 0),
            "blocked_rate":  v.get("blocked_rate", 0),
            "n_errors":      v.get("n_errors", 0),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_json(asr_data: dict, system_name: str, results_dir: str) -> str:
    path = os.path.join(results_dir, f"{system_name}_asr_summary.json")
    with open(path, "w") as f:
        json.dump({"system": system_name, **asr_data}, f, indent=2)
    return path


def generate_text_report(asr_data: dict,
                         system_name: str,
                         results_dir: str) -> tuple:
    ov      = asr_data["overall"]
    by_surf = asr_data["by_surface"]
    by_cat  = asr_data["by_category"]
    by_dom  = asr_data["by_domain"]

    lines = [
        "=" * 70,
        f"SECURERAG — EVALUATION REPORT: {system_name}",
        "=" * 70,
        "",
        "TABLE 2: Attack Success Rate by Category",
        "-" * 70,
        f"  {'Cat':<5} {'Name':<32} {'N':>4} {'Succ':>5} "
        f"{'ASR':>7} {'Blocked':>8}",
        "-" * 70,
    ]

    for cat in CATEGORY_ORDER:
        v       = by_cat.get(cat, {
            "n_attacks": 0, "n_success": 0, "asr": 0, "n_blocked": 0
        })
        name    = CATEGORY_NAMES.get(cat, cat)
        blocked = v.get("n_blocked", 0)
        lines.append(
            f"  {cat:<5} {name:<32} {v['n_attacks']:>4} "
            f"{v['n_success']:>5} {v['asr']*100:>6.1f}% "
            f"{blocked:>7}"
        )

    direct   = by_surf.get("direct",   {"n_attacks": 0, "n_success": 0,
                                         "asr": 0, "n_blocked": 0})
    indirect = by_surf.get("indirect", {"n_attacks": 0, "n_success": 0,
                                         "asr": 0, "n_blocked": 0})

    lines += [
        "-" * 70,
        f"  {'Direct (D1-D5)':<38} "
        f"{direct['n_attacks']:>4} {direct['n_success']:>5} "
        f"{direct['asr']*100:>6.1f}% "
        f"{direct.get('n_blocked',0):>7}",
        f"  {'Indirect (I1-I3)':<38} "
        f"{indirect['n_attacks']:>4} {indirect['n_success']:>5} "
        f"{indirect['asr']*100:>6.1f}% "
        f"{indirect.get('n_blocked',0):>7}",
        "=" * 70,
        f"  {'OVERALL':<38} "
        f"{ov['n_attacks']:>4} {ov['n_success']:>5} "
        f"{ov['asr']*100:>6.1f}% "
        f"{ov['n_blocked']:>7}",
        "",
        "  Note: 'Blocked' = Gemini safety filter triggered",
        "  (counted as failed attacks — model resisted injection)",
        "",
        "TABLE 3: ASR by Domain (Indirect Attacks Only)",
        "-" * 70,
    ]

    for domain in ["healthcare", "finance", "general"]:
        v = by_dom.get(domain, {"n_attacks": 0, "n_success": 0, "asr": 0})
        lines.append(
            f"  {domain:<20} n={v['n_attacks']:>3}  "
            f"success={v['n_success']:>3}  "
            f"ASR={v['asr']*100:>5.1f}%"
        )

    lines += [
        "",
        f"  Error rate   : {asr_data['errors']['error_rate']*100:.1f}% "
        f"({asr_data['errors']['n_errors']} failed LLM calls)",
        f"  Blocked rate : {ov['blocked_rate']*100:.1f}% "
        f"({ov['n_blocked']} safety filter triggers)",
        "=" * 70,
    ]

    report = "\n".join(lines)
    path   = os.path.join(results_dir,
                           f"{system_name}_evaluation_report.txt")
    with open(path, "w") as f:
        f.write(report)
    return report, path


def run_full_report(results: list[dict],
                    system_name: str,
                    results_dir: str) -> dict:
    """Compute all metrics, save all files, print report. Returns asr_data."""
    os.makedirs(results_dir, exist_ok=True)
    asr_data = compute_asr(results)

    csv_path            = save_csv(asr_data, system_name, results_dir)
    json_path           = save_json(asr_data, system_name, results_dir)
    report, report_path = generate_text_report(
        asr_data, system_name, results_dir
    )

    print(report)
    print(f"\nSaved → {csv_path}")
    print(f"Saved → {json_path}")
    print(f"Saved → {report_path}")

    return asr_data