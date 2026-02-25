from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


def compute_kl_divergence(base_logits: torch.Tensor, current_logits: torch.Tensor) -> float:
    """Compute KL(current || base) over final-token distributions."""
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    kl = F.kl_div(base_log_probs, current_log_probs, reduction="batchmean", log_target=True)
    return float(kl.item())
