from __future__ import annotations

from contextlib import ExitStack
from typing import Dict, Iterable, List

import torch


def _get_blocks(model) -> List:
    """Return transformer blocks for raw HF and PEFT-wrapped causal LMs."""
    inner = getattr(model, "base_model", model)
    inner = getattr(inner, "model", inner)
    return inner.transformer.h


class HiddenStateExtractor:
    """Capture intermediate hidden states using forward hooks."""

    def __init__(self, model, layers: Iterable[int]) -> None:
        self.model = model
        self.layers = list(layers)
        self.captured: Dict[int, torch.Tensor] = {}

    def _hook_factory(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            self.captured[layer_idx] = output[0].detach() if isinstance(output, tuple) else output.detach()

        return _hook

    def run(self, **forward_kwargs) -> Dict[int, torch.Tensor]:
        self.captured = {}
        blocks = _get_blocks(self.model)
        with ExitStack() as stack:
            for idx in self.layers:
                stack.enter_context(blocks[idx].register_forward_hook(self._hook_factory(idx)))
            with torch.no_grad():
                self.model(**forward_kwargs)
        return self.captured


def pooled_layer_features(
    hidden_by_layer: Dict[int, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pooled = []
    for _layer, hidden in sorted(hidden_by_layer.items()):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled.append(summed / lengths)
        else:
            pooled.append(hidden.mean(dim=1))
    return torch.cat(pooled, dim=-1)
