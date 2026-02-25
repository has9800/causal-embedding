from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import torch

from src.pipeline.state import CandidateTrace, PipelineState, TraceRecord
from src.rewards.combined_reward import combine_rewards
from src.rewards.probe_reward import probe_confidence_to_reward
from src.rewards.trace_reward import normalize_trace_score
from src.utils.embedding_extractor import pooled_layer_features
from src.utils.metrics import compute_kl_divergence


def _runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["configurable"]["runtime"]


def generate_traces(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    bundle = runtime["student"]
    prompt = state.prompt
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.model.device)
    num_generations = runtime["cfg"]["training"]["dpo"]["num_generations_per_prompt"]

    state.candidate_traces = []
    for _ in range(num_generations):
        outputs = bundle.model.generate(**inputs, max_new_tokens=180, do_sample=True, temperature=0.8)
        completion = bundle.tokenizer.decode(outputs[0], skip_special_tokens=True)
        state.candidate_traces.append(CandidateTrace(trace=completion))
    return state


def score_all(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    bundle = runtime["student"]
    extractor = runtime["extractor"]
    probe = runtime["probe"]

    for candidate in state.candidate_traces:
        result = runtime["local_filter"].evaluate(state.prompt, candidate.trace)
        candidate.local_passed = result.passed
        candidate.local_confidence = result.confidence
        should_use_premium = result.passed and runtime["cfg"]["critic"]["use_premium_critic"]

        if should_use_premium:
            score, justification = runtime["premium_critic"].score_trace(state.prompt, candidate.trace)
        elif result.passed:
            score = 6.0
            justification = "local-only warm trial score for passed trace"
        else:
            score = runtime["cfg"]["critic"]["default_filtered_score"]
            justification = "filtered out by local gate"

        trace_score_norm = normalize_trace_score(score)

        inputs = bundle.tokenizer(candidate.trace, return_tensors="pt", truncation=True, max_length=1024).to(bundle.model.device)
        hidden_by_layer = extractor.run(**inputs)
        features = pooled_layer_features(hidden_by_layer)
        with torch.no_grad():
            logits = probe(features.cpu())
        probe_score_norm = probe_confidence_to_reward(logits)

        candidate.trace_score = trace_score_norm
        candidate.probe_score = probe_score_norm
        candidate.combined_reward = combine_rewards(
            trace_score_norm,
            probe_score_norm,
            trace_weight=runtime["cfg"]["rewards"]["trace_weight"],
        )
        candidate.critic_justification = justification
    return state


def rank_and_pair(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    ranked = sorted(state.candidate_traces, key=lambda x: x.combined_reward)
    worst = ranked[0]
    best = ranked[-1]

    state.best_trace = best.trace
    state.worst_trace = worst.trace
    state.preference_rows.append(
        {
            "prompt": state.prompt,
            "chosen": state.best_trace,
            "rejected": state.worst_trace,
        }
    )

    record = TraceRecord(
        prompt=state.prompt,
        trace=best.trace,
        local_passed=best.local_passed,
        local_confidence=best.local_confidence,
        trace_score=best.trace_score,
        probe_score=best.probe_score,
        combined_reward=best.combined_reward,
        critic_justification=best.critic_justification,
    )
    state.history.append(record)
    runtime["trace_logger"].log(asdict(record))
    return state


def update_model(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    best = max(state.candidate_traces, key=lambda x: x.combined_reward)
    worst = min(state.candidate_traces, key=lambda x: x.combined_reward)
    runtime["metric_logger"].log(
        {
            "step": state.step,
            "num_candidates": len(state.candidate_traces),
            "best_reward": best.combined_reward,
            "worst_reward": worst.combined_reward,
            "reward_gap": best.combined_reward - worst.combined_reward,
        }
    )
    return state


def log_kl_metrics(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
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

    runtime["metric_logger"].log({"step": state.step, "mean_kl_divergence": sum(kls) / max(len(kls), 1)})
    return state


def human_checkpoint(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    interval = runtime["cfg"]["pipeline"]["human_checkpoint_every"]
    if state.step > 0 and state.step % interval == 0:
        runtime["metric_logger"].log(
            {
                "step": state.step,
                "event": "human_checkpoint",
                "message": "paused for human review",
            }
        )
    state.step += 1
    if state.step >= runtime["cfg"]["pipeline"]["max_steps"]:
        state.stop = True
    else:
        prompts: List[str] = runtime["prompts"]
        state.prompt = prompts[state.step % len(prompts)]
    return state
