from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LocalFilterResult:
    passed: bool
    confidence: float
    rationale: str


class LocalFilterCritic:
    """Cheap first-pass trace filter. Uses simple heuristics as warm-trial fallback."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def evaluate(self, prompt: str, trace: str) -> LocalFilterResult:
        length_score = min(len(trace.split()) / 80.0, 1.0)
        causal_tokens = ["because", "causes", "therefore", "if", "then"]
        hit_count = sum(1 for tok in causal_tokens if tok in trace.lower())
        causal_score = min(hit_count / 3.0, 1.0)
        confidence = 0.6 * length_score + 0.4 * causal_score
        passed = confidence >= 0.55
        rationale = "passes heuristic threshold" if passed else "fails heuristic threshold"
        return LocalFilterResult(passed=passed, confidence=confidence, rationale=rationale)
