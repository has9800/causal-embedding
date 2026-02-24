from __future__ import annotations

from contextlib import ExitStack
from typing import Dict, Iterable, List

import torch


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
        blocks: List = self.model.transformer.h
        with ExitStack() as stack:
            for idx in self.layers:
                stack.enter_context(blocks[idx].register_forward_hook(self._hook_factory(idx)))
            with torch.no_grad():
                self.model(**forward_kwargs)
        return self.captured


def pooled_layer_features(hidden_by_layer: Dict[int, torch.Tensor]) -> torch.Tensor:
    pooled = []
    for _layer, hidden in sorted(hidden_by_layer.items()):
        pooled.append(hidden.mean(dim=1))
    return torch.cat(pooled, dim=-1)
