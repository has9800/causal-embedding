from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import torch
from langgraph.types import RunnableConfig

from src.pipeline.state import CandidateTrace, PipelineState, TraceRecord
from src.rewards.combined_reward import combine_rewards
from src.rewards.probe_reward import probe_confidence_to_reward
from src.rewards.trace_reward import normalize_trace_score
from src.utils.embedding_extractor import pooled_layer_features
from src.utils.metrics import compute_kl_divergence


FEW_SHOT_PREFIX = """Prompt: Why does ice melt when salt is applied?
Trace: Cause: salt dissolves into the liquid film on ice. Mechanism: dissolved ions lower the freezing point through colligative effects. Effect: ice melts at temperatures below 0C.

Prompt: Why does exercise reduce blood pressure?
Trace: Cause: regular aerobic exercise is performed. Mechanism: repeated cardiac demand improves arterial elasticity and reduces peripheral resistance. Effect: resting blood pressure decreases over weeks.

"""


def _runtime(config: RunnableConfig) -> Dict[str, Any]:
    return config["configurable"]["runtime"]


def generate_traces(state: PipelineState, config: RunnableConfig) -> Dict[str, Any]:
    runtime = _runtime(config)
    bundle = runtime["student"]
    prompt = state["prompt"]

    full_prompt = FEW_SHOT_PREFIX + f"Prompt: {prompt}\nTrace:"
    inputs = bundle.tokenizer(full_prompt, return_tensors="pt").to(bundle.model.device)
    prefix_len = inputs["input_ids"].shape[1]

    num_generations = runtime["cfg"]["training"]["dpo"]["num_generations_per_prompt"]
    if num_generations < 2:
        raise ValueError(
            "num_generations_per_prompt must be >= 2 for DPO preference pairs, "
            f"got {num_generations}"
        )

    candidates: List[CandidateTrace] = []
    for _ in range(num_generations):
        outputs = bundle.model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.8,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][prefix_len:]
        completion = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if "Prompt:" in completion:
            completion = completion[:completion.index("Prompt:")].strip()

        candidates.append(CandidateTrace(trace=completion))
    return {"candidate_traces": candidates}


def score_all(state: PipelineState, config: RunnableConfig) -> Dict[str, Any]:
    runtime = _runtime(config)
    bundle = runtime["student"]
    extractor = runtime["extractor"]
    probe = runtime["probe"]

    candidates = state["candidate_traces"]
    for candidate in candidates:
        result = runtime["local_filter"].evaluate(state["prompt"], candidate.trace)
        candidate.local_passed = result.passed
        candidate.local_confidence = result.confidence
        should_use_premium = result.passed and runtime["cfg"]["critic"]["use_premium_critic"]

        if should_use_premium:
            score, justification = runtime["premium_critic"].score_trace(state["prompt"], candidate.trace)
        elif result.passed:
            score = 6.0
            justification = "local-only warm trial score for passed trace"
        else:
            score = runtime["cfg"]["critic"]["default_filtered_score"]
            justification = "filtered out by local gate"

        trace_score_norm = normalize_trace_score(score)

        inputs = bundle.tokenizer(candidate.trace, return_tensors="pt", truncation=True, max_length=1024).to(bundle.model.device)
        hidden_by_layer = extractor.run(**inputs)
        features = pooled_layer_features(hidden_by_layer, attention_mask=inputs.get("attention_mask"))
        with torch.no_grad():
            logits = probe(features.cpu().float())
        probe_score_norm = probe_confidence_to_reward(logits)

        candidate.trace_score = trace_score_norm
        candidate.probe_score = probe_score_norm
        candidate.combined_reward = combine_rewards(
            trace_score_norm,
            probe_score_norm,
            trace_weight=runtime["cfg"]["rewards"]["trace_weight"],
        )
        candidate.critic_justification = justification
    return {"candidate_traces": candidates}


def rank_and_pair(state: PipelineState, config: RunnableConfig) -> Dict[str, Any]:
    runtime = _runtime(config)
    ranked = sorted(state["candidate_traces"], key=lambda x: x.combined_reward)
    worst = ranked[0]
    best = ranked[-1]

    preference_row = {
        "prompt": state["prompt"],
        "chosen": best.trace,
        "rejected": worst.trace,
    }

    record = TraceRecord(
        prompt=state["prompt"],
        trace=best.trace,
        local_passed=best.local_passed,
        local_confidence=best.local_confidence,
        trace_score=best.trace_score,
        probe_score=best.probe_score,
        combined_reward=best.combined_reward,
        critic_justification=best.critic_justification,
    )
    runtime["trace_logger"].log(asdict(record))

    preference_rows = list(state.get("preference_rows", []))
    preference_rows.append(preference_row)
    history = list(state.get("history", []))
    history.append(record)

    return {
        "best_trace": best.trace,
        "worst_trace": worst.trace,
        "preference_rows": preference_rows,
        "history": history,
    }


def update_model(state: PipelineState, config: RunnableConfig) -> Dict[str, Any]:
    runtime = _runtime(config)
    best = max(state["candidate_traces"], key=lambda x: x.combined_reward)
    worst = min(state["candidate_traces"], key=lambda x: x.combined_reward)
    runtime["metric_logger"].log(
        {
            "step": state.get("step", 0),
            "num_candidates": len(state["candidate_traces"]),
            "best_reward": best.combined_reward,
            "worst_reward": worst.combined_reward,
            "reward_gap": best.combined_reward - worst.combined_reward,
        }
    )
    return {}


def log_kl_metrics(state: PipelineState, config: RunnableConfig) -> Dict[str, Any]:
    runtime = _runtime(config)
    bundle = runtime["student"]
    reference_prompts: List[str] = runtime["reference_prompts"]
    model = bundle.model
    tokenizer = bundle.tokenizer

    kls = []
    for prompt in reference_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            current_logits = model(**inputs).logits[:, -1, :]
            with model.disable_adapter():
                base_logits = model(**inputs).logits[:, -1, :]
        kls.append(compute_kl_divergence(base_logits, current_logits))

    return {"mean_kl_divergence": sum(kls) / max(len(kls), 1)}


def human_checkpoint(state: PipelineState, config: RunnableConfig) -> Dict[str, Any]:
    return {}
