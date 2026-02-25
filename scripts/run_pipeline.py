from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.critics.claude_critic import ClaudeCritic
from src.critics.local_filter import build_filter_critic
from src.models.probe import CausalProbe
from src.models.student import StudentModelFactory
from src.pipeline.graph import build_graph
from src.pipeline.state import PipelineState
from src.training.dpo_trainer import run_dpo
from src.utils.embedding_extractor import HiddenStateExtractor
from src.utils.logging import JsonlLogger


def _load_prompts(path: str):
    lines = Path(path).read_text().splitlines()
    return [json.loads(line)["prompt"] for line in lines if line.strip()]


def _compute_probe_input_dim(student_model, probe_layers: list[int]) -> int:
    hidden_size = student_model.config.hidden_size
    return int(hidden_size) * len(probe_layers)


def _candidate_for_trace(candidates, trace: str):
    for candidate in candidates:
        if candidate.trace == trace:
            return candidate
    return None


def main() -> None:
    cfg = yaml.safe_load(Path("config/training_config.yaml").read_text())

    factory = StudentModelFactory(cfg)
    trained_model = cfg["models"].get("trained_model", "iteration")
    student = factory.load(production=trained_model == "production")

    input_dim = _compute_probe_input_dim(student.model, cfg["probe"]["layers"])
    probe = CausalProbe(input_dim=input_dim, num_classes=cfg["probe"]["num_classes"])
    probe.load_state_dict(torch.load(cfg["probe"]["checkpoint_path"], map_location="cpu"))
    probe.eval()

    prompts = _load_prompts(cfg["pipeline"]["prompts_path"])
    runtime = {
        "cfg": cfg,
        "student": student,
        "local_filter": build_filter_critic(cfg),
        "premium_critic": ClaudeCritic(cfg),
        "probe": probe,
        "extractor": HiddenStateExtractor(student.model, cfg["probe"]["layers"]),
        "trace_logger": JsonlLogger(cfg["logging"]["traces_file"]),
        "metric_logger": JsonlLogger(cfg["logging"]["metrics_file"]),
        "prompts": prompts,
        "reference_prompts": prompts[:5],
    }

    graph = build_graph()
    num_rounds = cfg["pipeline"]["num_rounds"]
    batch_size = cfg["pipeline"]["batch_size_per_round"]
    dpo_root = Path(cfg["training"]["dpo"]["output_dir"])

    for round_index in range(1, num_rounds + 1):
        runtime["round_index"] = round_index
        preference_rows = []
        reward_gaps = []
        chosen_probe_scores = []
        rejected_probe_scores = []
        trace_score_gaps = []
        kl_values = []

        for batch_step in range(batch_size):
            prompt = prompts[((round_index - 1) * batch_size + batch_step) % len(prompts)]
            initial_state: PipelineState = {
                "step": batch_step,
                "prompt": prompt,
                "candidate_traces": [],
                "preference_rows": [],
                "history": [],
                "stop": False,
            }
            final_state = graph.invoke(initial_state, config={"configurable": {"runtime": runtime}})
            preference_rows.extend(final_state.get("preference_rows", []))

            candidates = final_state.get("candidate_traces", [])
            if not candidates:
                continue

            best_trace = final_state.get("best_trace", "")
            worst_trace = final_state.get("worst_trace", "")
            chosen_candidate = _candidate_for_trace(candidates, best_trace)
            rejected_candidate = _candidate_for_trace(candidates, worst_trace)
            if chosen_candidate is None or rejected_candidate is None:
                continue

            chosen_probe_scores.append(chosen_candidate.probe_score)
            rejected_probe_scores.append(rejected_candidate.probe_score)
            reward_gaps.append(chosen_candidate.combined_reward - rejected_candidate.combined_reward)
            trace_score_gaps.append(chosen_candidate.trace_score - rejected_candidate.trace_score)
            if "mean_kl_divergence" in final_state:
                kl_values.append(final_state["mean_kl_divergence"])

        if not preference_rows:
            runtime["metric_logger"].log({"round": round_index, "event": "round_skipped", "reason": "no preference rows"})
            continue

        round_dir = dpo_root / f"round_{round_index}"
        run_dpo(student.model, student.tokenizer, preference_rows, cfg, output_dir=str(round_dir))

        mean_chosen_probe = sum(chosen_probe_scores) / max(len(chosen_probe_scores), 1)
        mean_rejected_probe = sum(rejected_probe_scores) / max(len(rejected_probe_scores), 1)
        mean_reward_gap = sum(reward_gaps) / max(len(reward_gaps), 1)
        mean_trace_gap = sum(trace_score_gaps) / max(len(trace_score_gaps), 1)
        mean_kl = sum(kl_values) / max(len(kl_values), 1)

        runtime["metric_logger"].log(
            {
                "round": round_index,
                "batch_size_per_round": batch_size,
                "pairs_collected": len(preference_rows),
                "mean_reward_gap": mean_reward_gap,
                "mean_probe_chosen": mean_chosen_probe,
                "mean_probe_rejected": mean_rejected_probe,
                "probe_score_gap": mean_chosen_probe - mean_rejected_probe,
                "mean_trace_score_gap": mean_trace_gap,
                "mean_kl_divergence": mean_kl,
                "dpo_checkpoint_dir": str(round_dir),
            }
        )

        checkpoint_every = cfg["pipeline"]["checkpoint_every_rounds"]
        if checkpoint_every > 0 and round_index % checkpoint_every == 0:
            runtime["metric_logger"].log(
                {
                    "round": round_index,
                    "event": "round_checkpoint",
                    "message": "completed configured round checkpoint interval",
                }
            )

        human_every = cfg["pipeline"]["human_checkpoint_every_rounds"]
        if human_every > 0 and round_index % human_every == 0:
            runtime["metric_logger"].log(
                {
                    "round": round_index,
                    "event": "human_checkpoint",
                    "message": "paused for human review",
                }
            )


if __name__ == "__main__":
    main()
