"""Retrieval-aware, interpretable S4 risk scoring (no threshold tuning)."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class S4RiskConfig:
    document_weight: float = 0.35
    window_weight: float = 0.45
    proportion_weight: float = 0.15
    alignment_weight: float = 0.05
    suspicious_threshold: float = 0.45
    malicious_threshold: float = 0.80
    low_alignment_floor: float = 0.20


def cosine_similarity(left, right) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_l = sqrt(sum(float(a) ** 2 for a in left))
    norm_r = sqrt(sum(float(b) ** 2 for b in right))
    return round(dot / (norm_l * norm_r), 6) if norm_l and norm_r else 0.0


class RetrievalRiskScorer:
    def __init__(self, config: S4RiskConfig | None = None):
        self.config = config or S4RiskConfig()

    def alignment(self, query: str, document: str, embedding_function=None,
                  retrieval_distance: float | None = None) -> float | None:
        if embedding_function is not None:
            vectors = embedding_function([query, document])
            return cosine_similarity(vectors[0], vectors[1])
        if retrieval_distance is not None:
            return round(1.0 / (1.0 + max(float(retrieval_distance), 0.0)), 6)
        return None

    def score(self, *, document_score: float, max_sentence_score: float,
              suspicious_window_proportion: float,
              query_document_alignment: float | None) -> dict:
        c = self.config
        penalty = 0.0 if query_document_alignment is None else max(0.0, c.low_alignment_floor - query_document_alignment)
        risk = min(max(c.document_weight * document_score + c.window_weight * max_sentence_score + c.proportion_weight * suspicious_window_proportion + c.alignment_weight * penalty, 0.0), 1.0)
        risk = round(risk, 6)
        return {"risk_score": risk,
                "risk_level": "MALICIOUS" if risk >= c.malicious_threshold else ("SUSPICIOUS" if risk >= c.suspicious_threshold else "SAFE"),
                "alignment_penalty": round(penalty, 6),
                "weights": {"document": c.document_weight, "window": c.window_weight, "proportion": c.proportion_weight, "alignment": c.alignment_weight}}
