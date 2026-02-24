from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceRecord:
    prompt: str
    trace: str
    local_passed: bool = False
    local_confidence: float = 0.0
    trace_score: float = 0.0
    probe_score: float = 0.0
    combined_reward: float = 0.0
    critic_justification: str = ""


@dataclass
class PipelineState:
    step: int = 0
    prompt: str = ""
    trace: str = ""
    should_use_premium: bool = False
    hidden_features: Optional[Any] = None
    trace_score_raw: float = 0.0
    trace_score_norm: float = 0.0
    probe_score_norm: float = 0.0
    combined_reward: float = 0.0
    local_filter_passed: bool = False
    critic_justification: str = ""
    preference_rows: List[Dict[str, str]] = field(default_factory=list)
    history: List[TraceRecord] = field(default_factory=list)
    stop: bool = False
