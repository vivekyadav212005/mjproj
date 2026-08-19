"""S4 SecureRAG+ Metrics Calculator.

Computes S4-specific metrics from raw evaluation logs, including:
- Fast-path block rate (direct queries blocked).
- Sanitization rate (percentage of chunks sanitized).
- Quarantine rate (percentage of chunks fully quarantined).
- Average risk score across scenarios.
- Leakage rate (flagged text reaching the final context).
- Benign query utility and accuracy.
"""
from __future__ import annotations

import json
import os


def calculate_s4_metrics(results: list[dict]) -> dict:
    """Calculate S4-specific metrics from raw evaluation records."""
    total_queries = len(results)
    if total_queries == 0:
        return {
            "total_queries": 0, "queries_blocked": 0, "fast_path_rate": 0.0,
            "total_chunks": 0, "quarantined_chunks": 0, "sanitized_chunks": 0,
            "allowed_chunks": 0, "quarantine_rate": 0.0, "sanitization_rate": 0.0,
            "leakage_rate": 0.0, "average_risk": 0.0,
        }

    queries_blocked = sum(1 for r in results if r.get("blocked_at_layer") == "query")
    fast_path_rate = queries_blocked / total_queries

    total_chunks = 0
    quarantined_chunks = 0
    sanitized_chunks = 0
    allowed_chunks = 0
    risk_scores = []
    leakage_count = 0

    for r in results:
        audit_log = r.get("s4_audit") or []
        for entry in audit_log:
            total_chunks += 1
            risk_scores.append(float(entry.get("risk") or 0.0))

            q_state = entry.get("quarantine_state")
            if q_state == "quarantined":
                quarantined_chunks += 1
            elif q_state == "sanitized":
                sanitized_chunks += 1
            elif q_state == "allowed":
                allowed_chunks += 1

            if entry.get("flagged_text_reached_context", False):
                leakage_count += 1

    quarantine_rate = quarantined_chunks / total_chunks if total_chunks else 0.0
    sanitization_rate = sanitized_chunks / total_chunks if total_chunks else 0.0
    leakage_rate = leakage_count / total_chunks if total_chunks else 0.0
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

    return {
        "total_queries": total_queries,
        "queries_blocked": queries_blocked,
        "fast_path_rate": round(fast_path_rate, 4),
        "total_chunks": total_chunks,
        "quarantined_chunks": quarantined_chunks,
        "sanitized_chunks": sanitized_chunks,
        "allowed_chunks": allowed_chunks,
        "quarantine_rate": round(quarantine_rate, 4),
        "sanitization_rate": round(sanitization_rate, 4),
        "leakage_rate": round(leakage_rate, 4),
        "average_risk": round(avg_risk, 4),
    }


def format_s4_report(metrics: dict, system_name: str = "S4") -> str:
    """Format S4 metrics as a human-readable text report."""
    lines = [
        f"==================================================",
        f"SecureRAG+ — {system_name} Defense Metrics Report",
        f"==================================================",
        f"Queries Processed  : {metrics['total_queries']}",
        f"Queries Blocked    : {metrics['queries_blocked']}",
        f"Fast-path Rate     : {metrics['fast_path_rate']*100:.2f}%",
        f"--------------------------------------------------",
        f"Total Chunks Scanned: {metrics['total_chunks']}",
        f"  - Allowed Chunks : {metrics['allowed_chunks']}",
        f"  - Sanitized      : {metrics['sanitized_chunks']} ({metrics['sanitization_rate']*100:.2f}%)",
        f"  - Quarantined    : {metrics['quarantined_chunks']} ({metrics['quarantine_rate']*100:.2f}%)",
        f"--------------------------------------------------",
        f"Leakage Rate       : {metrics['leakage_rate']*100:.2f}% (flagged text in final prompt)",
        f"Average Risk Score : {metrics['average_risk']:.4f}",
        f"==================================================",
    ]
    return "\n".join(lines)
