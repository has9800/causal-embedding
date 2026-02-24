from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import torch

from src.pipeline.state import PipelineState, TraceRecord
from src.rewards.combined_reward import combine_rewards
from src.rewards.probe_reward import probe_confidence_to_reward
from src.rewards.trace_reward import normalize_trace_score
from src.utils.embedding_extractor import pooled_layer_features


def _runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["configurable"]["runtime"]


def generate_trace(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    bundle = runtime["student"]
    prompt = state.prompt
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.model.device)
    outputs = bundle.model.generate(**inputs, max_new_tokens=180, do_sample=True, temperature=0.8)
    completion = bundle.tokenizer.decode(outputs[0], skip_special_tokens=True)
    state.trace = completion
    return state


def local_filter(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    result = runtime["local_filter"].evaluate(state.prompt, state.trace)
    state.local_filter_passed = result.passed
    state.should_use_premium = result.passed and runtime["cfg"]["critic"]["use_premium_critic"]
    return state


def critic_evaluate(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    if state.should_use_premium:
        score, justification = runtime["premium_critic"].score_trace(state.prompt, state.trace)
    elif state.local_filter_passed:
        score = 6.0
        justification = "local-only warm trial score for passed trace"
    else:
        score = runtime["cfg"]["critic"]["default_filtered_score"]
        justification = "filtered out by local gate"
    state.trace_score_raw = score
    state.trace_score_norm = normalize_trace_score(score)
    state.critic_justification = justification
    return state


def extract_embeddings(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    extractor = runtime["extractor"]
    bundle = runtime["student"]
    inputs = bundle.tokenizer(state.trace, return_tensors="pt", truncation=True, max_length=1024).to(
        bundle.model.device
    )
    hidden_by_layer = extractor.run(**inputs)
    state.hidden_features = pooled_layer_features(hidden_by_layer)
    return state


def probe_score(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    probe = runtime["probe"]
    features = state.hidden_features
    with torch.no_grad():
        logits = probe(features.cpu())
    state.probe_score_norm = probe_confidence_to_reward(logits)
    return state


def compute_reward(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    state.combined_reward = combine_rewards(
        state.trace_score_norm,
        state.probe_score_norm,
        trace_weight=runtime["cfg"]["rewards"]["trace_weight"],
    )
    return state


def update_model(state: PipelineState, config: Dict[str, Any]) -> PipelineState:
    runtime = _runtime(config)
    state.preference_rows.append(
        {
            "prompt": state.prompt,
            "chosen": state.trace,
            "rejected": "",
        }
    )
    record = TraceRecord(
        prompt=state.prompt,
        trace=state.trace,
        local_passed=state.local_filter_passed,
        trace_score=state.trace_score_norm,
        probe_score=state.probe_score_norm,
        combined_reward=state.combined_reward,
        critic_justification=state.critic_justification,
    )
    state.history.append(record)

    runtime["trace_logger"].log(asdict(record))
    runtime["metric_logger"].log(
        {
            "step": state.step,
            "trace_reward": state.trace_score_norm,
            "probe_reward": state.probe_score_norm,
            "combined_reward": state.combined_reward,
        }
    )
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
