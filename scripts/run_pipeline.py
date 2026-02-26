from __future__ import annotations

import json
import random
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
from src.utils.embedding_extractor import HiddenStateExtractor, pooled_layer_features
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


def retrain_probe(student, cfg):
    """Re-extract embeddings from current model and retrain the causal probe."""
    probe_cfg = cfg["probe"]
    layers = probe_cfg["layers"]
    extractor = HiddenStateExtractor(student.model, layers)
    tokenizer = student.tokenizer
    device = student.model.device

    label_map = {"causal": 0, "correlational": 1, "unrelated": 2}
    probe_data_path = probe_cfg.get("dataset_path", "data/probe_training/probe_sentences.jsonl")
    sentences = []
    labels = []
    with open(probe_data_path) as f:
        for line in f:
            row = json.loads(line)
            sentences.append(row["sentence"])
            labels.append(label_map[row["label"]])

    all_features = []
    student.model.eval()
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=256).to(device)
        hidden_by_layer = extractor.run(**inputs)
        features = pooled_layer_features(hidden_by_layer, attention_mask=inputs.get("attention_mask"))
        all_features.append(features.cpu().float())

    X = torch.cat(all_features, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    n = len(y)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    X_train, X_val = X[perm[:split]], X[perm[split:]]
    y_train, y_val = y[perm[:split]], y[perm[split:]]

    input_dim = X.shape[1]
    num_classes = probe_cfg.get("num_classes", 3)
    probe = CausalProbe(input_dim=input_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    for _epoch in range(10):
        probe.train()
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)

    torch.save(probe.state_dict(), probe_cfg["checkpoint_path"])
    probe.eval()
    print(f"Probe retrained: val_accuracy={best_val_acc:.4f}")
    return probe, best_val_acc


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
    human_review_cfg = cfg["pipeline"].get("human_review", {})
    human_review_enabled = human_review_cfg.get("enabled", False)
    validation_rounds = int(human_review_cfg.get("validation_rounds", 0))
    review_percent = float(human_review_cfg.get("review_percent", 0.0))
    spot_check_percent = float(human_review_cfg.get("spot_check_percent", 0.0))
    spot_check_interval = int(human_review_cfg.get("spot_check_interval", 0))

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

        if human_review_enabled:
            in_validation = round_index <= validation_rounds
            in_spot_check = (
                not in_validation and spot_check_interval > 0 and round_index % spot_check_interval == 0
            )

            if in_validation or in_spot_check:
                review_pct = review_percent if in_validation else spot_check_percent
                num_to_review = max(1, int(len(preference_rows) * review_pct))
                num_to_review = min(num_to_review, len(preference_rows))

                review_indices = set(random.sample(range(len(preference_rows)), num_to_review))
                approved_rows = []
                reviewed_count = 0
                skip_remaining_review = False

                for row_index, row in enumerate(preference_rows):
                    if skip_remaining_review or row_index not in review_indices:
                        approved_rows.append(row)
                        continue

                    reviewed_count += 1
                    print(f"\n{'=' * 60}")
                    print(f"ROUND {round_index} — REVIEW {reviewed_count}/{num_to_review}")
                    print(f"{'=' * 60}")
                    print(f"PROMPT: {row['prompt']}\n")
                    print(f"CHOSEN TRACE (high reward):\n{row['chosen']}\n")
                    print(f"REJECTED TRACE (low reward):\n{row['rejected']}\n")

                    response = input("Approve this pair? [a]pprove / [r]eject / [s]kip round: ").strip().lower()

                    if response == "r":
                        runtime["metric_logger"].log(
                            {
                                "round": round_index,
                                "event": "pair_rejected",
                                "prompt": row["prompt"],
                                "reason": "human reviewer rejected",
                            }
                        )
                        continue

                    approved_rows.append(row)
                    if response == "s":
                        skip_remaining_review = True

                preference_rows = approved_rows

                runtime["metric_logger"].log(
                    {
                        "round": round_index,
                        "event": "human_review_complete",
                        "reviewed": reviewed_count,
                        "approved": len(preference_rows),
                        "phase": "validation" if in_validation else "spot_check",
                    }
                )

        if not preference_rows:
            runtime["metric_logger"].log(
                {"round": round_index, "event": "round_skipped", "reason": "no pairs after review"}
            )
            continue

        round_dir = dpo_root / f"round_{round_index}"
        run_dpo(student.model, student.tokenizer, preference_rows, cfg, output_dir=str(round_dir))

        probe, best_val_acc = retrain_probe(student, cfg)
        runtime["probe"] = probe
        runtime["extractor"] = HiddenStateExtractor(student.model, cfg["probe"]["layers"])
        print(f"Round {round_index}: probe and extractor updated for new model weights")

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
                "probe_val_accuracy_after_dpo": best_val_acc,
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
