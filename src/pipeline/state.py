from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TypedDict


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


class PipelineState(TypedDict, total=False):
    step: int
    prompt: str
    candidate_traces: List[CandidateTrace]
    best_trace: str
    worst_trace: str
    preference_rows: List[Dict[str, str]]
    history: List[TraceRecord]
    stop: bool
