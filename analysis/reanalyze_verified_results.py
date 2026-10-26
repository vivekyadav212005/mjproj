#!/usr/bin/env python3
"""Recompute SecureRAG statistics from frozen result files.

Does not call an LLM, does not modify results/final or results/s4,
and does not retune any threshold. Writes tables and SVG figures under
results/analysis/.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "analysis")
FIG = os.path.join(OUT, "figures")

SYSTEMS = ["S0", "S1", "S2", "S3", "S4"]
CATEGORIES = ["D1", "D2", "D3", "D4", "D5", "I1", "I2", "I3"]
Z95 = 1.959963984540054


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson(k: int, n: int, z: float = Z95) -> dict:
    if n <= 0:
        return {"k": k, "n": n, "p": None, "low": None, "high": None}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + z2 / (4 * n * n)) / denom
    return {
        "k": k,
        "n": n,
        "p": p,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def binom_pmf(k: int, n: int) -> float:
    return math.comb(n, k) / (2 ** n)


def mcnemar_exact(b: int, c: int) -> dict:
    """Two-sided exact McNemar test. b and c are the discordant counts."""
    n = b + c
    if n == 0:
        return {"b_s3_only": b, "c_s4_only": c, "n_discordant": 0, "p_value": 1.0}
    point = binom_pmf(b, n)
    p = sum(binom_pmf(k, n) for k in range(n + 1) if binom_pmf(k, n) <= point + 1e-15)
    return {
        "b_first_only": b,
        "c_second_only": c,
        "n_discordant": n,
        "p_value": min(1.0, p),
    }


def index_by(rows: list[dict], key: str) -> dict:
    return {row[key]: row for row in rows}


def risk_from_parts(document, window, proportion, alignment, weights) -> float:
    penalty = 0.0 if alignment is None else max(0.0, 0.20 - float(alignment))
    raw = (
        weights[0] * float(document)
        + weights[1] * float(window)
        + weights[2] * float(proportion)
        + weights[3] * penalty
    )
    return round(min(max(raw, 0.0), 1.0), 6)


def level(risk: float, suspicious: float = 0.45, malicious: float = 0.80) -> str:
    if risk >= malicious:
        return "MALICIOUS"
    if risk >= suspicious:
        return "SUSPICIOUS"
    return "SAFE"


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_bar_with_ci(path, labels, values, lows, highs, ylabel, title, colors):
    width, height = 760, 460
    left, right, top, bottom = 70, 20, 50, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = 1.0
    def y(v):
        return top + (1 - v / ymax) * plot_h
    n = len(labels)
    gap = plot_w / n
    bar_w = gap * 0.55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16">{title}</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Times New Roman, Times, serif" font-size="12">{tick:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#222"/>')
    for i, label in enumerate(labels):
        cx = left + gap * i + gap / 2
        x = cx - bar_w / 2
        parts.append(f'<rect x="{x:.1f}" y="{y(values[i]):.1f}" width="{bar_w:.1f}" height="{y(0)-y(values[i]):.1f}" fill="{colors[i]}"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{y(lows[i]):.1f}" x2="{cx:.1f}" y2="{y(highs[i]):.1f}" stroke="#111" stroke-width="1.4"/>')
        parts.append(f'<line x1="{cx-6:.1f}" y1="{y(lows[i]):.1f}" x2="{cx+6:.1f}" y2="{y(lows[i]):.1f}" stroke="#111"/>')
        parts.append(f'<line x1="{cx-6:.1f}" y1="{y(highs[i]):.1f}" x2="{cx+6:.1f}" y2="{y(highs[i]):.1f}" stroke="#111"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y(values[i])-6:.1f}" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="11">{values[i]*100:.1f}%</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height-28}" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="13">{label}</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="13">{ylabel}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def svg_grouped(path, categories, series, ylabel, title):
    width, height = 980, 500
    left, right, top, bottom = 70, 20, 56, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = 1.0
    def y(v):
        return top + (1 - v / ymax) * plot_h
    n = len(categories)
    m = len(series)
    group = plot_w / n
    bar_w = group / (m + 1.2)
    colors = ["#bdbdbd", "#6baed6", "#fd8d3c", "#e6550d", "#31a354"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16">{title}</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Times New Roman, Times, serif" font-size="12">{tick:.0%}</text>')
    for i, cat in enumerate(categories):
        gx = left + group * i
        for j, (name, vals) in enumerate(series):
            x = gx + bar_w * (j + 0.4)
            v = vals[i]
            parts.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bar_w*0.9:.1f}" height="{y(0)-y(v):.1f}" fill="{colors[j % len(colors)]}"/>')
        parts.append(f'<text x="{gx+group/2:.1f}" y="{height-42}" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="13">{cat}</text>')
    legend_x = left
    for j, (name, _) in enumerate(series):
        x = legend_x + j * 90
        parts.append(f'<rect x="{x}" y="{height-24}" width="12" height="12" fill="{colors[j]}"/>')
        parts.append(f'<text x="{x+16}" y="{height-14}" font-family="Times New Roman, Times, serif" font-size="12">{name}</text>')
    parts.append(f'<text x="16" y="{top+plot_h/2}" transform="rotate(-90 16 {top+plot_h/2})" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="13">{ylabel}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def svg_scatter(path, points, xlabel, ylabel, title):
    width, height = 720, 480
    left, right, top, bottom = 80, 30, 48, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    def x(v):
        return left + v * plot_w
    def y(v):
        return top + (1 - v) * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="16">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{x(tick):.1f}" y="{height-36}" text-anchor="middle" font-size="12" font-family="Times New Roman, Times, serif">{tick:.0%}</text>')
        parts.append(f'<text x="{left-8}" y="{y(tick)+4:.1f}" text-anchor="end" font-size="12" font-family="Times New Roman, Times, serif">{tick:.0%}</text>')
        parts.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left+plot_w}" y2="{y(tick):.1f}" stroke="#f0f0f0"/>')
    colors = {"S0": "#bdbdbd", "S1": "#6baed6", "S2": "#fd8d3c", "S3": "#e6550d", "S4": "#31a354"}
    for name, asr, removal in points:
        parts.append(f'<circle cx="{x(removal):.1f}" cy="{y(asr):.1f}" r="7" fill="{colors[name]}" stroke="#222"/>')
        parts.append(f'<text x="{x(removal)+10:.1f}" y="{y(asr)+4:.1f}" font-family="Times New Roman, Times, serif" font-size="13">{name}</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="13">{xlabel}</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="13">{ylabel}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    final_attacks = load_jsonl(os.path.join(ROOT, "results/final/raw")) if False else None
    by_system = {}
    for name in ("S0", "S1", "S2", "S3"):
        by_system[name] = index_by(
            load_jsonl(os.path.join(ROOT, "results/final/raw", f"{name}_attack_results.jsonl")),
            "attack_id",
        )
    by_system["S4"] = index_by(
        load_jsonl(os.path.join(ROOT, "results/s4/raw/S4_attack_results.jsonl")),
        "attack_id",
    )
    benign = {}
    for name in ("S0", "S1", "S2", "S3"):
        benign[name] = load_jsonl(os.path.join(ROOT, "results/final/raw", f"{name}_benign_results.jsonl"))
    benign["S4"] = load_jsonl(os.path.join(ROOT, "results/s4/raw/S4_benign_results.jsonl"))

    ids = sorted(by_system["S0"])
    assert all(set(by_system[s]) == set(ids) for s in SYSTEMS)
    assert all(len(benign[s]) == 150 for s in SYSTEMS)

    overall_rows = []
    category_rows = []
    asr_for_plot = {}
    ci_low, ci_high, asr_vals = [], [], []
    for system in SYSTEMS:
        rows = [by_system[system][i] for i in ids]
        k = sum(1 for row in rows if row["success"])
        interval = wilson(k, len(rows))
        asr_for_plot[system] = interval["p"]
        asr_vals.append(interval["p"])
        ci_low.append(interval["low"])
        ci_high.append(interval["high"])
        overall_rows.append({
            "system": system,
            "successes": k,
            "n": len(rows),
            "asr": round(interval["p"], 6),
            "wilson95_low": round(interval["low"], 6),
            "wilson95_high": round(interval["high"], 6),
            "query_blocked": sum(1 for row in rows if row.get("query_blocked")),
            "followup_blocked": sum(1 for row in rows if row.get("followup_blocked")),
            "llm_calls": sum(int(row.get("llm_calls") or 0) for row in rows),
            "mean_latency_ms": round(sum(float(row["latency_ms"]) for row in rows) / len(rows), 2),
        })
        for category in CATEGORIES:
            subset = [row for row in rows if row["category"] == category]
            kk = sum(1 for row in subset if row["success"])
            interval = wilson(kk, len(subset))
            category_rows.append({
                "system": system,
                "category": category,
                "successes": kk,
                "n": len(subset),
                "asr": round(interval["p"], 6),
                "wilson95_low": round(interval["low"], 6),
                "wilson95_high": round(interval["high"], 6),
            })

    def paired(first: str, second: str) -> dict:
        b = c = both = neither = 0
        for attack_id in ids:
            s1 = bool(by_system[first][attack_id]["success"])
            s2 = bool(by_system[second][attack_id]["success"])
            if s1 and s2:
                both += 1
            elif s1 and not s2:
                b += 1
            elif s2 and not s1:
                c += 1
            else:
                neither += 1
        out = mcnemar_exact(b, c)
        out.update({
            "first": first,
            "second": second,
            "both_success": both,
            "neither_success": neither,
            "n": len(ids),
        })
        return out

    comparisons = [paired("S0", "S3"), paired("S0", "S4"), paired("S3", "S4"), paired("S1", "S2")]
    s3_only = [
        attack_id for attack_id in ids
        if by_system["S3"][attack_id]["success"] and not by_system["S4"][attack_id]["success"]
    ]
    s4_only = [
        attack_id for attack_id in ids
        if by_system["S4"][attack_id]["success"] and not by_system["S3"][attack_id]["success"]
    ]
    q_s3 = {i for i in ids if by_system["S3"][i].get("query_blocked")}
    q_s4 = {i for i in ids if by_system["S4"][i].get("query_blocked")}

    # S4 component audit on saved chunk logs.
    attack_chunks = []
    for row in by_system["S4"].values():
        for chunk in row.get("s4_audit") or []:
            attack_chunks.append(chunk)
    benign_chunks = []
    benign_query_stats = []
    gold_sanitized_queries = 0
    gold_quarantined_queries = 0
    gold_retained = 0
    gold_retrieved = 0
    for row in benign["S4"]:
        audits = row.get("s4_audit") or []
        benign_chunks.extend(audits)
        gold = row.get("gold_source_doc_id")
        if row.get("gold_source_retrieved"):
            gold_retrieved += 1
        if row.get("gold_source_retained"):
            gold_retained += 1
        matched = [
            chunk for chunk in audits
            if gold and (
                gold == chunk.get("chunk_id")
                or gold in str(chunk.get("chunk_id"))
                or str(chunk.get("chunk_id")).startswith(str(gold))
            )
        ]
        if any(c.get("quarantine_state") == "quarantined" for c in matched):
            gold_quarantined_queries += 1
        if any(c.get("quarantine_state") == "sanitized" for c in matched):
            gold_sanitized_queries += 1
        benign_query_stats.append({
            "query_id": row["query_id"],
            "sanitized_chunks": row.get("s4_sanitized_chunks"),
            "quarantined_chunks": row.get("s4_quarantined_chunks"),
            "windows_removed": row.get("s4_windows_removed"),
            "gold_source_retained": row.get("gold_source_retained"),
            "gold_chunks_matched": len(matched),
        })

    def summarize_chunks(chunks: list[dict], split: str) -> dict:
        policies = Counter(c.get("policy") for c in chunks)
        states = Counter(c.get("quarantine_state") for c in chunks)
        pairs = Counter((c.get("policy"), c.get("quarantine_state")) for c in chunks)
        penalties = []
        alignments = []
        for chunk in chunks:
            alignment = chunk.get("alignment")
            if alignment is None:
                penalties.append(0.0)
            else:
                alignments.append(float(alignment))
                penalties.append(max(0.0, 0.20 - float(alignment)))
        windows_total = sum(int((c.get("window_counts") or {}).get("total") or 0) for c in chunks)
        windows_suspicious = sum(int((c.get("window_counts") or {}).get("suspicious") or 0) for c in chunks)
        return {
            "split": split,
            "n_chunks": len(chunks),
            "policy": dict(policies),
            "state": dict(states),
            "policy_state": {f"{a}|{b}": n for (a, b), n in pairs.items()},
            "safe_but_sanitized": pairs.get(("SAFE", "sanitized"), 0),
            "alignment_missing": sum(1 for c in chunks if c.get("alignment") is None),
            "injected": sum(1 for c in chunks if c.get("injected")),
            "alignment_min": min(alignments) if alignments else None,
            "alignment_max": max(alignments) if alignments else None,
            "penalty_max": max(penalties) if penalties else None,
            "penalty_nonzero": sum(1 for p in penalties if p > 0),
            "max_alignment_contribution": round(0.05 * (max(penalties) if penalties else 0.0), 6),
            "windows_total": windows_total,
            "windows_suspicious": windows_suspicious,
        }

    component = {
        "attack_scanned_chunks": summarize_chunks(attack_chunks, "attack"),
        "benign_chunks": summarize_chunks(benign_chunks, "benign"),
        "benign_gold": {
            "gold_retrieved_queries": gold_retrieved,
            "gold_retained_queries": gold_retained,
            "queries_whose_matched_gold_chunk_was_sanitized": gold_sanitized_queries,
            "queries_whose_matched_gold_chunk_was_quarantined": gold_quarantined_queries,
            "queries_with_zero_gold_chunk_id_match": sum(1 for row in benign_query_stats if row["gold_source_retained"] and row["gold_chunks_matched"] == 0),
        },
    }

    # Predeclared weight variants. Alignment penalty is zero on this log,
    # so any change isolated to the alignment weight cannot move a score.
    variants = {
        "published": (0.35, 0.45, 0.15, 0.05),
        "no_alignment": (0.35, 0.45, 0.15, 0.00),
        "alignment_only_weight_increased": (0.35, 0.45, 0.15, 1.00),
        "document_only": (1.00, 0.00, 0.00, 0.00),
        "window_only": (0.00, 1.00, 0.00, 0.00),
        "proportion_only": (0.00, 0.00, 1.00, 0.00),
        "equal_without_alignment": (1 / 3, 1 / 3, 1 / 3, 0.00),
    }
    variant_rows = []
    for name, weights in variants.items():
        for split, chunks in (("attack", attack_chunks), ("benign", benign_chunks)):
            counts = Counter()
            code_would_quarantine = 0
            for chunk in chunks:
                window = max(chunk.get("window_scores") or [0.0])
                proportion = 0.0
                counts_w = chunk.get("window_counts") or {}
                if counts_w.get("total"):
                    proportion = counts_w["suspicious"] / counts_w["total"]
                risk = risk_from_parts(
                    chunk.get("document_score") or 0.0,
                    window,
                    proportion,
                    chunk.get("alignment"),
                    weights,
                )
                counts[level(risk)] += 1
                if level(risk) == "MALICIOUS":
                    code_would_quarantine += 1
            variant_rows.append({
                "variant": name,
                "split": split,
                "weights": list(weights),
                "n": len(chunks),
                "SAFE": counts["SAFE"],
                "SUSPICIOUS": counts["SUSPICIOUS"],
                "MALICIOUS": counts["MALICIOUS"],
                "spec_quarantine": code_would_quarantine,
            })

    # Threshold sensitivity at published weights. Counts label flips only.
    threshold_rows = []
    published_levels = []
    for chunk in attack_chunks:
        window = max(chunk.get("window_scores") or [0.0])
        counts_w = chunk.get("window_counts") or {}
        proportion = (counts_w["suspicious"] / counts_w["total"]) if counts_w.get("total") else 0.0
        published_levels.append(risk_from_parts(
            chunk.get("document_score") or 0.0, window, proportion, chunk.get("alignment"),
            variants["published"],
        ))
    for suspicious in (0.35, 0.45, 0.55):
        for malicious in (0.70, 0.80, 0.90):
            counts = Counter(level(r, suspicious, malicious) for r in published_levels)
            threshold_rows.append({
                "suspicious_threshold": suspicious,
                "malicious_threshold": malicious,
                "n_attack_chunks_scanned": len(published_levels),
                "SAFE": counts["SAFE"],
                "SUSPICIOUS": counts["SUSPICIOUS"],
                "MALICIOUS": counts["MALICIOUS"],
                "note": "label counts only; attack success was not recomputed",
            })

    utility_rows = []
    points = []
    for system in SYSTEMS:
        rows = benign[system]
        removed = sum(int(r.get("n_chunks_blocked") or 0) for r in rows)
        retrieved = sum(int(r.get("n_chunks_retrieved") or 0) for r in rows)
        queries_with_removal = sum(1 for r in rows if int(r.get("n_chunks_blocked") or 0) > 0)
        no_ctx = sum(1 for r in rows if r.get("no_clean_context"))
        answerable = sum(1 for r in rows if r.get("answerable"))
        gold_r = sum(1 for r in rows if r.get("gold_source_retrieved"))
        gold_k = sum(1 for r in rows if r.get("gold_source_retained"))
        q_blocked = sum(1 for r in rows if r.get("query_blocked"))
        utility_rows.append({
            "system": system,
            "queries": len(rows),
            "queries_blocked": q_blocked,
            "queries_with_any_chunk_removed": queries_with_removal,
            "chunks_removed": removed,
            "chunks_retrieved": retrieved,
            "chunk_removal_rate": round(removed / retrieved, 6) if retrieved else None,
            "no_clean_context": no_ctx,
            "answerable": answerable,
            "gold_retrieved": gold_r,
            "gold_retained": gold_k,
            "gold_retention_given_retrieved": round(gold_k / gold_r, 6) if gold_r else None,
            "mean_latency_ms": round(sum(float(r["latency_ms"]) for r in rows) / len(rows), 2),
        })
        points.append((system, asr_for_plot[system], removed / retrieved))

    s4_benign_altered = sum(
        1 for row in benign["S4"]
        if int(row.get("s4_sanitized_chunks") or 0) or int(row.get("s4_quarantined_chunks") or 0)
    )

    report = {
        "source": {
            "s0_s3_attacks": "results/final/raw/S{0,1,2,3}_attack_results.jsonl",
            "s4_attacks": "results/s4/raw/S4_attack_results.jsonl",
            "benign": "matching raw benign JSONL files",
            "llm_rerun": False,
            "thresholds_changed": False,
        },
        "overall": overall_rows,
        "by_category": category_rows,
        "mcnemar": comparisons,
        "s3_s4_attack_ids": {
            "success_s3_not_s4": s3_only,
            "success_s4_not_s3": s4_only,
            "query_blocked_s3_not_s4": sorted(q_s3 - q_s4),
            "query_blocked_s4_not_s3": sorted(q_s4 - q_s3),
        },
        "components": component,
        "weight_variants_label_counts": variant_rows,
        "threshold_sensitivity_label_counts": threshold_rows,
        "utility": utility_rows,
        "s4_benign_queries_with_any_sanitized_or_quarantined_chunk": s4_benign_altered,
        "definitions": {
            "asr": "successes / 160 attacks; success is the frozen keyword rule",
            "chunk_removal_rate": "chunks with blocked/quarantined state / chunks retrieved",
            "answerable": "pipeline status allows an answer; not factual correctness",
            "gold_retention_given_retrieved": "queries whose designated source id remains after filtering, divided by queries in which it was retrieved",
            "wilson": "95% Wilson score interval; chunks and attacks are treated as independent, which overstates precision for clustered chunks",
        },
    }
    with open(os.path.join(OUT, "verified_statistics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_csv(os.path.join(OUT, "overall_asr.csv"), overall_rows)
    write_csv(os.path.join(OUT, "asr_by_category.csv"), category_rows)
    write_csv(os.path.join(OUT, "utility.csv"), utility_rows)
    write_csv(os.path.join(OUT, "weight_variant_labels.csv"), [
        {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
        for row in variant_rows
    ])
    write_csv(os.path.join(OUT, "threshold_sensitivity_labels.csv"), threshold_rows)

    colors = ["#bdbdbd", "#6baed6", "#fd8d3c", "#e6550d", "#31a354"]
    svg_bar_with_ci(
        os.path.join(FIG, "fig1_overall_asr.svg"),
        SYSTEMS, asr_vals, ci_low, ci_high,
        "Attack success rate",
        "Attack success rate with 95% Wilson intervals (n = 160)",
        colors,
    )
    series = []
    for system in SYSTEMS:
        vals = []
        for category in CATEGORIES:
            match = next(r for r in category_rows if r["system"] == system and r["category"] == category)
            vals.append(match["asr"])
        series.append((system, vals))
    svg_grouped(
        os.path.join(FIG, "fig2_asr_by_category.svg"),
        CATEGORIES, series,
        "Attack success rate",
        "Attack success rate by category",
    )
    svg_scatter(
        os.path.join(FIG, "fig3_security_vs_chunk_removal.svg"),
        points,
        "Benign chunk removal rate (whole chunks dropped / 750)",
        "Attack success rate",
        "Security versus whole-chunk removal",
    )
    print(json.dumps({
        "overall": overall_rows,
        "mcnemar_s3_s4": comparisons[2],
        "discordant_ids": report["s3_s4_attack_ids"],
        "attack_components": component["attack_scanned_chunks"],
        "benign_components": component["benign_chunks"],
        "gold": component["benign_gold"],
        "s4_benign_queries_altered": s4_benign_altered,
        "utility": utility_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
