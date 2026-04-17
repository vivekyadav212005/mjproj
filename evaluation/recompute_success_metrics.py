#!/usr/bin/env python3
"""
Recompute per-row `success` from an existing *_attack_results.jsonl using the
current `is_success()` rules, then regenerate ASR artifacts (CSV/JSON/report).

Does not call the LLM or rerun attacks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.attack_runner import is_success
from evaluation.results_reporter import (
    compute_asr,
    generate_text_report,
    save_csv,
    save_json,
)


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recompute success flags + ASR metrics from results JSONL"
    )
    p.add_argument(
        "--jsonl",
        required=True,
        help="Path to e.g. results/s0/S0_attack_results.jsonl",
    )
    args = p.parse_args()

    jsonl_path = os.path.abspath(args.jsonl)
    results_dir = os.path.dirname(jsonl_path)
    base = os.path.basename(jsonl_path)
    if not base.endswith("_attack_results.jsonl"):
        raise SystemExit(
            "Expected filename like {SYSTEM}_attack_results.jsonl"
        )
    system_name = base.replace("_attack_results.jsonl", "")

    rows = load_jsonl(jsonl_path)
    for r in rows:
        kw = r.get("success_keyword", "")
        resp = r.get("response", "") or ""
        r["success"] = bool(is_success(resp, kw))

    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{system_name}_attack_results_",
        suffix=".jsonl",
        dir=results_dir,
    )
    os.close(fd)
    try:
        write_jsonl(tmp_path, rows)
        os.replace(tmp_path, jsonl_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    os.makedirs(results_dir, exist_ok=True)
    asr_data = compute_asr(rows)
    csv_path = save_csv(asr_data, system_name, results_dir)
    json_path = save_json(asr_data, system_name, results_dir)
    _, report_path = generate_text_report(asr_data, system_name, results_dir)

    print(f"Updated: {jsonl_path}")
    print(f"Saved:   {csv_path}")
    print(f"Saved:   {json_path}")
    print(f"Saved:   {report_path}")


if __name__ == "__main__":
    main()
