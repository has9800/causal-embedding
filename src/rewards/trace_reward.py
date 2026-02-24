from __future__ import annotations


def normalize_trace_score(score_0_10: float) -> float:
    return max(0.0, min(1.0, score_0_10 / 10.0))
