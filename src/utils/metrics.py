from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeometryMetrics:
    mean_probe_reward: float
    mean_trace_reward: float
    mean_combined_reward: float


def summarize_rewards(trace_rewards, probe_rewards, combined_rewards) -> GeometryMetrics:
    n = max(len(combined_rewards), 1)
    return GeometryMetrics(
        mean_probe_reward=sum(probe_rewards) / max(len(probe_rewards), 1),
        mean_trace_reward=sum(trace_rewards) / max(len(trace_rewards), 1),
        mean_combined_reward=sum(combined_rewards) / n,
    )
