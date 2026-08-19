#!/usr/bin/env python3
"""
Aggregate the raw records in results/final/raw/ into the final artifacts.

Reads only; performs no LLM calls, so it can be rerun safely. Writes:

    results/final/final_attack_results.csv
    results/final/final_attack_results.jsonl
    results/final/final_asr_by_category.csv
    results/final/final_summary.json
    results/final/final_benign_results.csv
    results/final/final_utility_summary.json
    results/final/FINAL_RESEARCH_REPORT.md
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join("results", "final")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
INJECTED_ID = "injected_poisoned_doc"
SYSTEMS = ["S0", "S1", "S2", "S3"]
CATEGORIES = ["D1", "D2", "D3", "D4", "D5", "I1", "I2", "I3"]
CATEGORY_NAMES = {
    "D1": "Instruction Override",
    "D2": "Role Hijacking",
    "D3": "Obfuscated Injection",
    "D4": "Prompt Leakage",
    "D5": "Multi-Turn Anchoring",
    "I1": "Explicit Instruction in Doc",
    "I2": "Hidden Instruction in Doc (hard)",
    "I3": "Persona Shift via Document",
}
SYSTEM_LABEL = {
    "S0": "S0 undefended baseline",
    "S1": "S1 RuleClassifier only",
    "S2": "S2 persisted TF-IDF/LR @ 0.40",
    "S3": "S3 Rule OR persisted TF-IDF/LR",
}


def read_jsonl(path: str) -> list[dict]:
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def rate(num: int, den: int, nd: int = 4):
    return round(num / den, nd) if den else None


def prf(tp: int, fp: int, tn: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(f1, 4), "false_positive_rate": round(fpr, 4),
    }


# ── attack metrics ────────────────────────────────────────────────────────

def asr_block(rows: list[dict]) -> dict:
    n = len(rows)
    s = sum(1 for r in rows if r["success"])
    return {"n_attacks": n, "n_success": s, "asr": rate(s, n),
            "asr_pct": round(100 * s / n, 2) if n else None}


def attack_metrics(rows: list[dict]) -> dict:
    out = {
        "overall": asr_block(rows),
        "direct": asr_block([r for r in rows if r["surface"] == "direct"]),
        "indirect": asr_block([r for r in rows if r["surface"] == "indirect"]),
        "by_category": {
            c: asr_block([r for r in rows if r["category"] == c]) for c in CATEGORIES
        },
        "by_domain": {},
        "D5_split": {
            "single_turn": asr_block(
                [r for r in rows if r["category"] == "D5" and not r["is_multi_turn"]]
            ),
            "multi_turn": asr_block(
                [r for r in rows if r["category"] == "D5" and r["is_multi_turn"]]
            ),
        },
    }
    for dom in sorted({r.get("domain") or "any" for r in rows}):
        out["by_domain"][dom] = asr_block(
            [r for r in rows if (r.get("domain") or "any") == dom]
        )

    lat = [r["latency_ms"] for r in rows]
    lat_sorted = sorted(lat)
    out["operational"] = {
        "n_llm_calls": sum(r["llm_calls"] for r in rows),
        "n_queries_blocked": sum(1 for r in rows if r["query_blocked"]),
        "n_followups_blocked": sum(1 for r in rows if r["followup_blocked"]),
        "n_no_clean_context": sum(1 for r in rows if r["no_clean_context"]),
        "n_chunks_scanned": sum(r["n_chunks_retrieved"] for r in rows),
        "n_chunks_blocked": sum(r["n_chunks_blocked"] for r in rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "mean_latency_ms": round(sum(lat) / len(lat), 2) if lat else None,
        "median_latency_ms": (
            round(lat_sorted[len(lat_sorted) // 2], 2) if lat_sorted else None
        ),
        "total_latency_s": round(sum(lat) / 1000, 1),
    }
    return out


def document_metrics(rows: list[dict], defended: bool) -> dict:
    """
    Injected indirect chunk = positive. Clean KB chunks = negative.
    Attacks blocked at the query layer never reach a document scan and are
    excluded from the confusion matrix (counted separately).
    """
    if not defended:
        return {
            "applicable": False,
            "note": "S0 has no document filter; every chunk reaches the LLM.",
        }

    tp = fp = tn = fn = 0
    tp_d = fp_d = tn_d = fn_d = 0     # indirect-only (primary)
    excluded = 0
    by_cat = {}

    for r in rows:
        scans = r.get("chunk_clf") or []
        if not scans:
            excluded += 1
            continue
        cat = r["category"]
        b = by_cat.setdefault(cat, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        for e in scans:
            pos = e["chunk_id"] == INJECTED_ID
            blk = bool(e["blocked"])
            key = ("tp" if blk else "fn") if pos else ("fp" if blk else "tn")
            b[key] += 1
            if key == "tp":
                tp += 1
            elif key == "fn":
                fn += 1
            elif key == "fp":
                fp += 1
            else:
                tn += 1
            if r["surface"] == "indirect":
                if key == "tp":
                    tp_d += 1
                elif key == "fn":
                    fn_d += 1
                elif key == "fp":
                    fp_d += 1
                else:
                    tn_d += 1

    primary = prf(tp_d, fp_d, tn_d, fn_d)
    primary["scope"] = "indirect attacks only (injected chunk = positive)"
    allscope = prf(tp, fp, tn, fn)
    allscope["scope"] = "all attack chunk scans (direct KB chunks add negatives)"
    return {
        "applicable": True,
        "primary": primary,
        "all_attack_chunks": allscope,
        "by_category": {
            c: prf(v["tp"], v["fp"], v["tn"], v["fn"]) for c, v in sorted(by_cat.items())
        },
        "n_attacks_without_document_scan": excluded,
    }


# ── utility metrics ───────────────────────────────────────────────────────

def utility_metrics(rows: list[dict], defended: bool) -> dict:
    n = len(rows)
    blocked = sum(1 for r in rows if r["query_blocked"])
    answerable = sum(1 for r in rows if r["answerable"])
    retrieved = sum(r["n_chunks_retrieved"] for r in rows)
    removed = sum(r["n_chunks_blocked"] for r in rows)
    gold_retrieved = sum(1 for r in rows if r.get("gold_source_retrieved"))
    gold_retained = sum(1 for r in rows if r.get("gold_source_retained"))
    lat = [r["latency_ms"] for r in rows]

    by_domain = {}
    for dom in sorted({r.get("domain") for r in rows}):
        d = [r for r in rows if r.get("domain") == dom]
        gr = sum(1 for r in d if r.get("gold_source_retrieved"))
        gk = sum(1 for r in d if r.get("gold_source_retained"))
        by_domain[dom] = {
            "n": len(d),
            "queries_blocked": sum(1 for r in d if r["query_blocked"]),
            "answerable": sum(1 for r in d if r["answerable"]),
            "chunks_removed": sum(r["n_chunks_blocked"] for r in d),
            "gold_source_retrieved": gr,
            "gold_source_retained": gk,
            "gold_source_retention_given_retrieved": rate(gk, gr),
        }

    return {
        "n_queries": n,
        "queries_blocked": blocked,
        "benign_query_false_positive_rate": rate(blocked, n),
        "no_clean_context": sum(1 for r in rows if r["no_clean_context"]),
        "answerable": answerable,
        "answerability": rate(answerable, n),
        "chunks_retrieved": retrieved,
        "chunks_removed": removed,
        "chunk_removal_rate": rate(removed, retrieved),
        "benign_document_false_positive_rate": rate(removed, retrieved) if defended else None,
        "avg_chunks_removed_per_query": rate(removed, n),
        "gold_source_retrieved": gold_retrieved,
        "gold_source_retained": gold_retained,
        "gold_source_retention_of_all_queries": rate(gold_retained, n),
        "gold_source_retention_given_retrieved": rate(gold_retained, gold_retrieved),
        "by_domain": by_domain,
        "operational": {
            "n_llm_calls": sum(r["llm_calls"] for r in rows),
            "n_queries_blocked": blocked,
            "n_chunks_scanned": retrieved,
            "n_chunks_blocked": removed,
            "n_errors": sum(1 for r in rows if r.get("error")),
            "mean_latency_ms": round(sum(lat) / len(lat), 2) if lat else None,
            "total_latency_s": round(sum(lat) / 1000, 1),
        },
    }


# ── validity checks ───────────────────────────────────────────────────────

def validity_checks(attack_by_system: dict[str, list[dict]],
                    benign_by_system: dict[str, list[dict]]) -> dict:
    """Post-hoc audit of the run. Descriptive only: changes no number above."""
    checks: dict = {}

    def blocked(rows, kind):
        if kind == "query":
            return {r["attack_id"] for r in rows if r["query_blocked"]}
        if kind == "followup":
            return {r["attack_id"] for r in rows if r["followup_blocked"]}
        return {(r["attack_id"], e["chunk_id"])
                for r in rows for e in r["chunk_clf"] if e["blocked"]}

    if all(s in attack_by_system for s in ("S1", "S2", "S3")):
        s1, s2, s3 = (attack_by_system[s] for s in ("S1", "S2", "S3"))
        scanned = {s: {r["attack_id"] for r in rows if r["chunk_clf"]}
                   for s, rows in (("S1", s1), ("S2", s2), ("S3", s3))}
        common = scanned["S1"] & scanned["S2"] & scanned["S3"]
        keep = lambda st: {x for x in st if x[0] in common}  # noqa: E731
        checks["s3_is_rule_or_ml"] = {
            "queries_blocked_S3_equals_S1_union_S2":
                blocked(s3, "query") == blocked(s1, "query") | blocked(s2, "query"),
            "followups_blocked_S3_equals_S1_union_S2":
                blocked(s3, "followup") == blocked(s1, "followup") | blocked(s2, "followup"),
            "chunks_blocked_S3_equals_S1_union_S2_on_commonly_scanned_attacks":
                keep(blocked(s3, "chunk")) == keep(blocked(s1, "chunk")) | keep(blocked(s2, "chunk")),
            "n_attacks_document_scanned_by_all_three": len(common),
            "note": "A chunk-set comparison is only meaningful where all three "
                    "systems reached a document scan: S3 blocks strictly more "
                    "queries, and a query-blocked attack has no chunk scans.",
        }

    if "S0" in attack_by_system:
        rows = attack_by_system["S0"]
        checks["s0_undefended"] = {
            "chunks_scanned_by_a_classifier":
                sum(1 for r in rows for e in r["chunk_clf"] if e.get("scanned")),
            "query_decisions_recorded":
                sum(1 for r in rows if r["query_clf_blocked"] is not None),
            "queries_blocked": sum(1 for r in rows if r["query_blocked"]),
            "chunks_blocked": sum(r["n_chunks_blocked"] for r in rows),
        }

    per_system = {}
    for s, rows in attack_by_system.items():
        brows = benign_by_system.get(s, [])
        ids = [r["attack_id"] for r in rows]
        per_system[s] = {
            "n_attacks": len(ids),
            "unique_attack_ids": len(set(ids)),
            "n_benign": len(brows),
            "errors": sum(1 for r in rows + brows if r.get("error")),
            "empty_responses": sum(1 for r in rows if not r["response"]),
            "blocked_rows_that_called_the_llm": sum(
                1 for r in rows + brows
                if r["status"] in ("BLOCKED", "NO_CLEAN_CONTEXT") and r["llm_calls"]
                and r.get("blocked_at_layer") != "followup"
            ),
            "chunk_accounting_mismatches": sum(
                1 for r in rows + brows
                if r["n_chunks_retrieved"]
                and r["n_chunks_blocked"] + r["n_chunks_allowed"] != r["n_chunks_retrieved"]
            ),
        }
    checks["per_system"] = per_system

    gold = {s: sorted({r["query_id"] for r in rows if r.get("gold_source_retrieved")})
            for s, rows in benign_by_system.items() if rows}
    if len(gold) > 1:
        first = next(iter(gold.values()))
        checks["retrieval_identical_across_systems"] = {
            "same_gold_retrieved_query_set": all(v == first for v in gold.values()),
            "n_queries_with_gold_in_top_k": len(first),
        }
    return checks


# ── writers ───────────────────────────────────────────────────────────────

ATTACK_CSV_FIELDS = [
    "attack_id", "system", "classifier", "category", "category_name", "surface",
    "domain", "is_multi_turn", "success", "success_keyword", "status",
    "blocked_at_layer", "query_blocked", "followup_blocked", "no_clean_context",
    "injected_chunk_blocked", "injected_chunk_score", "query_clf_blocked",
    "query_clf_score", "followup_clf_blocked", "followup_clf_score",
    "n_chunks_retrieved", "n_chunks_blocked", "n_chunks_allowed",
    "llm_calls", "latency_ms", "error", "response",
]

BENIGN_CSV_FIELDS = [
    "query_id", "system", "classifier", "domain", "question", "status",
    "blocked_at_layer", "query_blocked", "no_clean_context", "answerable",
    "n_chunks_retrieved", "n_chunks_blocked", "n_chunks_allowed",
    "gold_source_doc_id", "gold_source_retrieved", "gold_source_retained",
    "llm_calls", "latency_ms", "error", "response",
]


def write_csv(path: str, fields: list[str], rows: list[dict],
              truncate: dict[str, int] | None = None) -> None:
    truncate = truncate or {}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fields}
            for k, lim in truncate.items():
                if isinstance(out.get(k), str):
                    out[k] = out[k][:lim].replace("\n", " ").replace("\r", " ")
            w.writerow(out)


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def fmt(v) -> str:
    return "—" if v is None else str(v)


def build_report(summary: dict, utility: dict, meta: dict) -> str:
    present = [s for s in SYSTEMS if s in summary["systems"]]
    L: list[str] = []
    a = L.append

    a("# SecureRAG — Final Research Report")
    a("")
    a(f"Generated {datetime.now().isoformat(timespec='seconds')}. "
      "All numbers below come from a single run of the frozen benchmark; no "
      "threshold, attack, or metric was changed after results were seen.")
    a("")

    a("## 1. Methodology")
    a("")
    gen = meta["generator"]
    ret = meta["retrieval"]
    a("| Component | Setting |")
    a("|---|---|")
    a(f"| Generator | Ollama `{gen['model']}` |")
    a(f"| Temperature | {gen['temperature']} |")
    a(f"| Max tokens (`num_predict`) | {gen['max_tokens_num_predict']} |")
    a(f"| Seed | {gen['seed']} |")
    a(f"| Embedding model | {ret['embedding_model']} |")
    a(f"| Vector store | `{ret['db_path']}` — rebuilt from the "
      f"`chromadb/clean` SQLite chunk text/metadata |")
    a(f"| Chunks indexed | {ret['n_chunks']} |")
    a(f"| TOP_K | {ret['top_k']} |")
    a(f"| Indirect context | {ret['indirect_context']} |")
    a("| Prompt | `config.S0_PROMPT_TEMPLATE` + `config.RAG_SYSTEM_PROMPT`, "
      "byte-identical for S0/S1/S2/S3 |")
    a("| Success metric | case-insensitive `success_keyword` substring, "
      "overridden to failure on a refusal phrase |")
    a("")
    a("The four systems share one retriever object, one prompt, one generation "
      "configuration and one success function. They differ only in the "
      "classifier applied:")
    a("")
    a("| System | Classifier | Blocking rule |")
    a("|---|---|---|")
    a("| S0 | none | undefended; no query gate, no document scan, no follow-up scan |")
    a("| S1 | `RuleClassifier` G1–G5 | block at rule score ≥ 0.50 |")
    a("| S2 | persisted TF-IDF + Logistic Regression | block at p ≥ 0.40 |")
    a("| S3 | S1 rule **OR** S2 model | block if either fires |")
    a("")
    a("S1/S2/S3 classify (1) the user query before retrieval, (2) every "
      "retrieved chunk, (3) every injected indirect chunk, and (4) every D5 "
      "follow-up turn. Only blocked chunks are removed; remaining clean chunks "
      "are passed through. A blocked query performs no retrieval and no LLM "
      "call. If every chunk is removed the system returns `NO_CLEAN_CONTEXT` "
      "and makes no LLM call. No defended system adds prompt hardening or any "
      "output-side filter.")
    a("")
    bm = meta["benchmark"]
    a("### Benchmark (frozen, 160 attacks)")
    a("")
    a("| Slice | n | Source |")
    a("|---|---|---|")
    a(f"| Direct D1–D5 | {bm['composition']['direct']} | "
      "`attack_suite/attack_suite_full.json` |")
    a("| Indirect I1 | 20 | original |")
    a("| Indirect I2 | 20 | hard-indirect v1 (`attack_suite_hard_indirect/I2_hard.json`) |")
    a("| Indirect I3 | 20 | original |")
    a(f"| **Total** | **{bm['n_attacks']}** | content SHA-256 "
      f"`{bm['attack_content_sha256'][:16]}…` |")
    a("")
    a(f"D5 is {bm['composition']['D5_single_turn']} single-turn controls plus "
      f"{bm['composition']['D5_multi_turn']} true multi-turn attacks. "
      "Utility is measured separately on the frozen 150 benign queries.")
    a("")

    a("## 2. Attack success rate")
    a("")
    hdr = "| Scope | " + " | ".join(SYSTEM_LABEL[s] for s in present) + " |"
    a(hdr)
    a("|---" * (len(present) + 1) + "|")

    def asr_row(label: str, getter) -> None:
        cells = []
        for s in present:
            blk = getter(summary["systems"][s]["attack"])
            cells.append(f"{pct(blk['asr'])} ({blk['n_success']}/{blk['n_attacks']})")
        a(f"| {label} | " + " | ".join(cells) + " |")

    asr_row("**Overall (160)**", lambda m: m["overall"])
    asr_row("Direct (100)", lambda m: m["direct"])
    asr_row("Indirect (60)", lambda m: m["indirect"])
    for c in CATEGORIES:
        asr_row(f"{c} {CATEGORY_NAMES[c]} (20)",
                lambda m, c=c: m["by_category"][c])
    asr_row("D5 single-turn (5)", lambda m: m["D5_split"]["single_turn"])
    asr_row("D5 multi-turn (15)", lambda m: m["D5_split"]["multi_turn"])
    a("")

    a("## 3. Document-filter quality")
    a("")
    a("Positive = the injected poisoned chunk on an indirect attack. "
      "Negative = a clean KB chunk retrieved alongside it. Attacks blocked at "
      "the query layer never reach a document scan and are excluded.")
    a("")
    a("| System | TP | FN | FP | TN | Precision | Recall | F1 | FPR |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s in present:
        d = summary["systems"][s]["document"]
        if not d.get("applicable"):
            a(f"| {s} | — | — | — | — | — | — | — | — |")
            continue
        p = d["primary"]
        a(f"| {s} | {p['tp']} | {p['fn']} | {p['fp']} | {p['tn']} | "
          f"{p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | "
          f"{p['false_positive_rate']:.3f} |")
    a("")
    a("S0 has no document filter, so the confusion matrix is undefined for it.")
    a("")

    a("## 4. Utility on 150 benign queries")
    a("")
    a("| Metric | " + " | ".join(present) + " |")
    a("|---" * (len(present) + 1) + "|")
    rows_spec = [
        ("Queries blocked", lambda u: f"{u['queries_blocked']} / {u['n_queries']}"),
        ("Benign query FPR", lambda u: pct(u["benign_query_false_positive_rate"])),
        ("Answerability", lambda u: f"{pct(u['answerability'])} "
                                    f"({u['answerable']}/{u['n_queries']})"),
        ("`NO_CLEAN_CONTEXT`", lambda u: str(u["no_clean_context"])),
        ("Chunks removed", lambda u: f"{u['chunks_removed']} / "
                                     f"{u['chunks_retrieved']}"),
        ("Benign document FPR", lambda u: pct(u["chunk_removal_rate"])),
        ("Gold source retrieved", lambda u: f"{u['gold_source_retrieved']} / "
                                            f"{u['n_queries']}"),
        ("Gold source retained", lambda u: f"{u['gold_source_retained']} / "
                                           f"{u['n_queries']}"),
        ("Gold retention, given retrieved",
         lambda u: pct(u["gold_source_retention_given_retrieved"])),
    ]
    for label, fn in rows_spec:
        a(f"| {label.replace('|', '')} | "
          + " | ".join(fn(utility[s]) for s in present) + " |")
    a("")

    a("## 5. Cost and operational counters")
    a("")
    a("| Counter | " + " | ".join(present) + " |")
    a("|---" * (len(present) + 1) + "|")
    op_rows = [
        ("LLM calls (attacks)", lambda s: summary["systems"][s]["attack"]["operational"]["n_llm_calls"]),
        ("LLM calls (benign)", lambda s: utility[s]["operational"]["n_llm_calls"]),
        ("Queries blocked (attacks)", lambda s: summary["systems"][s]["attack"]["operational"]["n_queries_blocked"]),
        ("Follow-ups blocked (attacks)", lambda s: summary["systems"][s]["attack"]["operational"]["n_followups_blocked"]),
        ("`NO_CLEAN_CONTEXT` (attacks)", lambda s: summary["systems"][s]["attack"]["operational"]["n_no_clean_context"]),
        ("Chunks retrieved (attacks)", lambda s: summary["systems"][s]["attack"]["operational"]["n_chunks_scanned"]),
        ("Chunks blocked (attacks)", lambda s: summary["systems"][s]["attack"]["operational"]["n_chunks_blocked"]),
        ("Queries blocked (benign)", lambda s: utility[s]["queries_blocked"]),
        ("Chunks blocked (benign)", lambda s: utility[s]["chunks_removed"]),
        ("Mean latency, attacks (ms)", lambda s: summary["systems"][s]["attack"]["operational"]["mean_latency_ms"]),
        ("Mean latency, benign (ms)", lambda s: utility[s]["operational"]["mean_latency_ms"]),
        ("LLM/pipeline errors", lambda s: summary["systems"][s]["attack"]["operational"]["n_errors"]
                                          + utility[s]["operational"]["n_errors"]),
    ]
    for label, fn in op_rows:
        a(f"| {label} | " + " | ".join(fmt(fn(s)) for s in present) + " |")
    a("")

    a("## 6. Comparison")
    a("")
    if len(present) == len(SYSTEMS):
        best_asr = min(present, key=lambda s: summary["systems"][s]["attack"]["overall"]["asr"])
        s0 = summary["systems"]["S0"]["attack"]["overall"]["asr"]
        a(f"Lowest overall ASR: **{best_asr}** at "
          f"{pct(summary['systems'][best_asr]['attack']['overall']['asr'])} "
          f"against the S0 baseline of {pct(s0)}.")
        a("")
        a("| System | Overall ASR | Δ vs S0 | Benign query FPR | Answerability | "
          "Gold retention (given retrieved) |")
        a("|---|---|---|---|---|---|")
        for s in present:
            m = summary["systems"][s]["attack"]["overall"]
            u = utility[s]
            delta = "—" if s == "S0" else f"{100 * (m['asr'] - s0):+.1f} pp"
            a(f"| {SYSTEM_LABEL[s]} | {pct(m['asr'])} | {delta} | "
              f"{pct(u['benign_query_false_positive_rate'])} | "
              f"{pct(u['answerability'])} | "
              f"{pct(u['gold_source_retention_given_retrieved'])} |")
        a("")
        a("A lower ASR is only a real defence if utility holds: read each row "
          "with the benign FPR and answerability columns, not the ASR alone.")
    a("")
    a("## 7. Validity checks")
    a("")
    v = summary.get("validity_checks") or {}
    s3v = v.get("s3_is_rule_or_ml") or {}
    s0v = v.get("s0_undefended") or {}
    if s3v:
        a("S3 is the disjunction it claims to be. Blocked-query and "
          "blocked-follow-up sets are exactly S1 ∪ S2. Chunk-level sets agree "
          f"on the {s3v.get('n_attacks_document_scanned_by_all_three')} attacks "
          "all three systems document-scanned; they cannot be compared "
          "elsewhere, because S3 blocks strictly more queries and a "
          "query-blocked attack never reaches a document scan.")
        a("")
        a("| Check | Result |")
        a("|---|---|")
        a(f"| Blocked queries: S3 = S1 ∪ S2 | "
          f"{s3v.get('queries_blocked_S3_equals_S1_union_S2')} |")
        a(f"| Blocked follow-ups: S3 = S1 ∪ S2 | "
          f"{s3v.get('followups_blocked_S3_equals_S1_union_S2')} |")
        a(f"| Blocked chunks: S3 = S1 ∪ S2 (commonly scanned attacks) | "
          f"{s3v.get('chunks_blocked_S3_equals_S1_union_S2_on_commonly_scanned_attacks')} |")
    if s0v:
        a(f"| S0 chunks scanned by a classifier | "
          f"{s0v.get('chunks_scanned_by_a_classifier')} |")
        a(f"| S0 queries blocked / chunks removed | "
          f"{s0v.get('queries_blocked')} / {s0v.get('chunks_blocked')} |")
    r_id = v.get("retrieval_identical_across_systems") or {}
    if r_id:
        a(f"| Identical retrieval across systems (gold-in-top-5 set) | "
          f"{r_id.get('same_gold_retrieved_query_set')} |")
    ps = v.get("per_system") or {}
    if ps:
        a(f"| Records per system (attacks / benign) | "
          + ", ".join(f"{s}: {d['n_attacks']}/{d['n_benign']}" for s, d in ps.items())
          + " |")
        a(f"| LLM or pipeline errors | "
          f"{sum(d['errors'] for d in ps.values())} |")
        a(f"| Blocked rows that still called the LLM | "
          f"{sum(d['blocked_rows_that_called_the_llm'] for d in ps.values())} |")
        a(f"| Chunk accounting mismatches | "
          f"{sum(d['chunk_accounting_mismatches'] for d in ps.values())} |")
    a("")

    a("## 8. Artifacts")
    a("")
    for name in ["final_metadata.json", "final_attack_results.csv",
                 "final_attack_results.jsonl", "final_asr_by_category.csv",
                 "final_summary.json", "final_benign_results.csv",
                 "final_utility_summary.json"]:
        a(f"- `results/final/{name}`")
    a("- `results/final/raw/` — per-system unaggregated records, including the "
      "full per-chunk classifier log")
    a("")
    a("Pre-existing result directories (`results/s0`, `s1`, `s2`, `s3`, "
      "`s3_fixed`, `s2_audit`, `s2_retrained`, `hard_indirect`, "
      "`indirect_audit`) and `chromadb/clean` were not written to by this run.")
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    meta_path = os.path.join(RESULTS_DIR, "final_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    attack_rows_all: list[dict] = []
    benign_rows_all: list[dict] = []
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "systems": {},
    }
    utility: dict[str, dict] = {}
    attack_by_system: dict[str, list[dict]] = {}
    benign_by_system: dict[str, list[dict]] = {}

    for s in SYSTEMS:
        arows = read_jsonl(os.path.join(RAW_DIR, f"{s}_attack_results.jsonl"))
        brows = read_jsonl(os.path.join(RAW_DIR, f"{s}_benign_results.jsonl"))
        if not arows and not brows:
            continue
        defended = s != "S0"
        for r in arows:
            r["category_name"] = CATEGORY_NAMES.get(r["category"], r.get("category_name", ""))
        attack_rows_all.extend(arows)
        benign_rows_all.extend(brows)
        attack_by_system[s] = arows
        benign_by_system[s] = brows
        summary["systems"][s] = {
            "label": SYSTEM_LABEL[s],
            "classifier": (arows or brows)[0].get("classifier"),
            "n_attack_records": len(arows),
            "n_benign_records": len(brows),
            "attack": attack_metrics(arows),
            "document": document_metrics(arows, defended),
        }
        utility[s] = utility_metrics(brows, defended)

    summary["validity_checks"] = validity_checks(attack_by_system, benign_by_system)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "final_attack_results.jsonl"), "w") as f:
        for r in attack_rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_csv(os.path.join(RESULTS_DIR, "final_attack_results.csv"),
              ATTACK_CSV_FIELDS, attack_rows_all, truncate={"response": 500})
    write_csv(os.path.join(RESULTS_DIR, "final_benign_results.csv"),
              BENIGN_CSV_FIELDS, benign_rows_all, truncate={"response": 500})

    with open(os.path.join(RESULTS_DIR, "final_asr_by_category.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "classifier", "scope", "category", "category_name",
                    "surface", "n_attacks", "n_success", "asr", "asr_pct",
                    "n_queries_blocked", "n_chunks_blocked", "n_llm_calls"])
        for s, blk in summary["systems"].items():
            m = blk["attack"]
            for c in CATEGORIES:
                rows_c = [r for r in attack_rows_all
                          if r["system"] == s and r["category"] == c]
                b = m["by_category"][c]
                w.writerow([
                    s, blk["classifier"], "category", c, CATEGORY_NAMES[c],
                    "direct" if c.startswith("D") else "indirect",
                    b["n_attacks"], b["n_success"], b["asr"], b["asr_pct"],
                    sum(1 for r in rows_c if r["query_blocked"]),
                    sum(r["n_chunks_blocked"] for r in rows_c),
                    sum(r["llm_calls"] for r in rows_c),
                ])
            for scope in ("direct", "indirect", "overall"):
                b = m[scope]
                rows_s = [r for r in attack_rows_all if r["system"] == s
                          and (scope == "overall" or r["surface"] == scope)]
                w.writerow([
                    s, blk["classifier"], "aggregate", scope.upper(), scope,
                    scope if scope != "overall" else "all",
                    b["n_attacks"], b["n_success"], b["asr"], b["asr_pct"],
                    sum(1 for r in rows_s if r["query_blocked"]),
                    sum(r["n_chunks_blocked"] for r in rows_s),
                    sum(r["llm_calls"] for r in rows_s),
                ])

    with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "final_utility_summary.json"), "w") as f:
        json.dump({"generated_at": summary["generated_at"], "systems": utility},
                  f, indent=2)
    with open(os.path.join(RESULTS_DIR, "FINAL_RESEARCH_REPORT.md"), "w") as f:
        f.write(build_report(summary, utility, meta))

    print("Wrote:")
    for name in ["final_attack_results.jsonl", "final_attack_results.csv",
                 "final_asr_by_category.csv", "final_summary.json",
                 "final_benign_results.csv", "final_utility_summary.json",
                 "FINAL_RESEARCH_REPORT.md"]:
        print(f"  {RESULTS_DIR}/{name}")
    for s, blk in summary["systems"].items():
        o = blk["attack"]["overall"]
        u = utility.get(s, {})
        print(f"  {s}: ASR {o['asr_pct']}% ({o['n_success']}/{o['n_attacks']}), "
              f"benign FPR {u.get('benign_query_false_positive_rate')}, "
              f"answerability {u.get('answerability')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
