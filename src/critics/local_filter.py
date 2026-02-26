from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LocalFilterResult:
    passed: bool
    confidence: float
    rationale: str


class BaseFilterCritic(ABC):
    @abstractmethod
    def evaluate(self, prompt: str, trace: str) -> LocalFilterResult:
        raise NotImplementedError


class HeuristicFilterCritic(BaseFilterCritic):
    """PLACEHOLDER — heuristic only, replace before real training."""

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


class ModelFilterCritic(BaseFilterCritic):
    """Score causal traces using a local language model."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        model_name = cfg["models"]["local_filter_model_name"]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self, prompt: str, trace: str) -> LocalFilterResult:
        import re

        import torch

        scoring_prompt = f"""Score the following causal reasoning trace on a scale of 1-10 based on:
- Correctness: Is the causal mechanism factually accurate?
- Completeness: Does it have a clear Cause, Mechanism, and Effect?
- Specificity: Does it explain HOW the cause produces the effect?
- Depth: Does it go beyond surface-level restating of the question?

Example 1:
Question: Why does smoking cause lung cancer?
Trace: Cause: smoking deposits tar in lung tissue. Mechanism: tar contains carcinogens that damage DNA repair mechanisms in bronchial epithelial cells, leading to uncontrolled cell proliferation. Effect: malignant tumors develop in lung tissue.
Score: 9

Example 2:
Question: Why does smoking cause lung cancer?
Trace: Cause: smoking is bad. Mechanism: it causes cancer. Effect: lung cancer.
Score: 2

Example 3:
Question: Why does deforestation increase flooding?
Trace: Cause: trees are removed from hillsides. Mechanism: without root systems to anchor soil and canopy to intercept rainfall, water runs off compacted ground as surface flow with shortened lag time. Effect: downstream flood peaks are higher and faster.
Score: 8

Example 4:
Question: Why does deforestation increase flooding?
Trace: Cause: deforestation releases water vapor which forms clouds. Mechanism: high cloud cover increases local rainfall. Effect: the more rainforest there is, the more rainfall.
Score: 2

Now score this trace:
Question: {prompt}
Trace: {trace}
Score:"""

        inputs = self.tokenizer(scoring_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        match = re.search(r"(\d+)", generated)
        if match:
            score = min(max(int(match.group(1)), 1), 10)
        else:
            score = 5

        confidence = score / 10.0
        passed = score >= 4
        rationale = f"model_critic_score={score}/10 raw_output='{generated[:50]}'"
        return LocalFilterResult(passed=passed, confidence=confidence, rationale=rationale)

    def offload(self) -> None:
        """Move model to CPU to free VRAM during DPO training."""
        self.model = self.model.to("cpu")

        import torch

        torch.cuda.empty_cache()

    def reload(self) -> None:
        """Move model back to GPU for scoring."""
        self.model = self.model.to("cuda")


def build_filter_critic(cfg: Dict[str, Any]) -> BaseFilterCritic:
    mode = cfg.get("critic", {}).get("local_filter_type", "heuristic")
    if mode == "model":
        return ModelFilterCritic(cfg)
    return HeuristicFilterCritic(cfg)
