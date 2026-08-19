"""Fine-grained, auditable S4 document scanning.

This module deliberately reuses a supplied existing classifier (the S3
RuleOrMLClassifier in the experiment) rather than training another detector.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass(frozen=True)
class S4Window:
    index: int
    text: str


class SentenceWindowScanner:
    """Split text into bounded sentence windows and classify every window."""

    def __init__(self, classifier, max_window_chars: int = 420):
        if max_window_chars < 80:
            raise ValueError("max_window_chars must be at least 80")
        self.classifier = classifier
        self.max_window_chars = max_window_chars

    def split(self, text: str) -> list[S4Window]:
        """Return non-overlapping sentence/windows preserving all text."""
        parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text or "") if p.strip()]
        windows: list[S4Window] = []
        for part in parts:
            for start in range(0, len(part), self.max_window_chars):
                piece = part[start:start + self.max_window_chars].strip()
                if piece:
                    windows.append(S4Window(len(windows), piece))
        return windows

    def scan(self, text: str) -> dict:
        windows = self.split(text)
        audit = []
        for window in windows:
            decision = self.classifier.classify(window.text)
            audit.append({
                "window_index": window.index, "window_text": window.text,
                "score": float(decision.get("score") or 0.0),
                "rule_score": decision.get("rule_score"), "ml_score": decision.get("ml_score"),
                "triggered_groups": decision.get("triggered_groups") or [],
                "blocked": bool(decision.get("blocked")), "decided_by": decision.get("decided_by"),
            })
        suspicious = [w for w in audit if w["blocked"]]
        return {
            "windows": audit, "total_windows": len(audit),
            "suspicious_window_count": len(suspicious),
            "suspicious_window_proportion": round(len(suspicious) / len(audit), 6) if audit else 0.0,
            "max_sentence_score": max((w["score"] for w in audit), default=0.0),
            "flagged_text": [w["window_text"] for w in suspicious],
            "clean_text": " ".join(w["window_text"] for w in audit if not w["blocked"]),
        }
