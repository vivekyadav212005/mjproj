#!/usr/bin/env python3
"""
Frozen 160-attack benchmark for the final experiment.

    100 direct   D1-D5          attack_suite/attack_suite_full.json
     20 indirect I1 (original)  attack_suite/attack_suite_full.json
     20 indirect I2 (hard)      attack_suite_hard_indirect/I2_hard.json
     20 indirect I3 (original)  attack_suite/attack_suite_full.json

Attacks are read and ordered only. No attack text, trigger, keyword, or
label is generated or modified here, and neither source file is written to.
"""

from __future__ import annotations

import hashlib
import json
import os

ORIGINAL_SUITE = os.path.join("attack_suite", "attack_suite_full.json")
HARD_I2_SUITE = os.path.join("attack_suite_hard_indirect", "I2_hard.json")

DIRECT_CATEGORIES = ["D1", "D2", "D3", "D4", "D5"]
EXPECTED = {"D1": 20, "D2": 20, "D3": 20, "D4": 20, "D5": 20,
            "I1": 20, "I2": 20, "I3": 20}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_benchmark() -> tuple[list[dict], dict]:
    with open(ORIGINAL_SUITE) as f:
        original = json.load(f)
    with open(HARD_I2_SUITE) as f:
        hard_i2 = json.load(f)

    by_cat = {c: [a for a in original if a["category"] == c] for c in DIRECT_CATEGORIES}
    i1 = [a for a in original if a["category"] == "I1"]
    i3 = [a for a in original if a["category"] == "I3"]
    i2 = [a for a in hard_i2 if a["category"] == "I2"]

    attacks: list[dict] = []
    for cat in DIRECT_CATEGORIES:
        attacks.extend(by_cat[cat])
    attacks.extend(i1)
    attacks.extend(i2)
    attacks.extend(i3)

    counts = {}
    for a in attacks:
        counts[a["category"]] = counts.get(a["category"], 0) + 1
    if counts != EXPECTED:
        raise ValueError(f"benchmark composition mismatch: {counts} != {EXPECTED}")
    if len(attacks) != 160:
        raise ValueError(f"expected 160 attacks, got {len(attacks)}")
    ids = [a["attack_id"] for a in attacks]
    if len(set(ids)) != 160:
        raise ValueError("duplicate attack_id in assembled benchmark")

    id_hash = hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()
    content_hash = hashlib.sha256(
        json.dumps(attacks, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    manifest = {
        "n_attacks": len(attacks),
        "composition": {
            "direct": sum(1 for a in attacks if a["surface"] == "direct"),
            "indirect": sum(1 for a in attacks if a["surface"] == "indirect"),
            "by_category": counts,
            "D5_single_turn": sum(
                1 for a in attacks
                if a["category"] == "D5" and not a.get("is_multi_turn")
            ),
            "D5_multi_turn": sum(
                1 for a in attacks
                if a["category"] == "D5" and a.get("is_multi_turn")
            ),
        },
        "sources": {
            "direct_D1_D5": ORIGINAL_SUITE,
            "indirect_I1": ORIGINAL_SUITE + " (original)",
            "indirect_I2": HARD_I2_SUITE + " (hard_indirect_v1)",
            "indirect_I3": ORIGINAL_SUITE + " (original)",
        },
        "source_file_sha256": {
            ORIGINAL_SUITE: _sha256_file(ORIGINAL_SUITE),
            HARD_I2_SUITE: _sha256_file(HARD_I2_SUITE),
        },
        "attack_id_sha256": id_hash,
        "attack_content_sha256": content_hash,
        "attack_ids": ids,
        "modified_by_experiment": False,
    }
    return attacks, manifest


if __name__ == "__main__":
    atk, man = load_benchmark()
    print(json.dumps({k: v for k, v in man.items() if k != "attack_ids"}, indent=2))
