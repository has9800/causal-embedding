from __future__ import annotations


def combine_rewards(trace_reward: float, probe_reward: float, trace_weight: float = 0.5) -> float:
    probe_weight = 1.0 - trace_weight
    return trace_weight * trace_reward + probe_weight * probe_reward
