from __future__ import annotations

import torch


def probe_confidence_to_reward(logits: torch.Tensor, causal_index: int = 0) -> float:
    probs = torch.softmax(logits, dim=-1)
    causal_prob = float(probs[..., causal_index].mean().item())
    return max(0.0, min(1.0, causal_prob))
