from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CandidateTrace:
    trace: str
    local_passed: bool = False
    local_confidence: float = 0.0
    trace_score: float = 0.0
    probe_score: float = 0.0
    combined_reward: float = 0.0
    critic_justification: str = ""


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
    candidate_traces: List[CandidateTrace] = field(default_factory=list)
    best_trace: str = ""
    worst_trace: str = ""
    preference_rows: List[Dict[str, str]] = field(default_factory=list)
    history: List[TraceRecord] = field(default_factory=list)
    stop: bool = False
