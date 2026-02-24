from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.critics.claude_critic import ClaudeCritic
from src.critics.local_filter import LocalFilterCritic
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


def main() -> None:
    cfg = yaml.safe_load(Path("config/training_config.yaml").read_text())

    factory = StudentModelFactory(cfg)
    student = factory.load(production=False)

    probe = CausalProbe(input_dim=256, num_classes=cfg["probe"]["num_classes"])
    probe.load_state_dict(torch.load(cfg["probe"]["checkpoint_path"], map_location="cpu"))
    probe.eval()

    prompts = _load_prompts(cfg["pipeline"]["prompts_path"])
    runtime = {
        "cfg": cfg,
        "student": student,
        "local_filter": LocalFilterCritic(cfg),
        "premium_critic": ClaudeCritic(cfg),
        "probe": probe,
        "extractor": HiddenStateExtractor(student.model, cfg["probe"]["layers"]),
        "trace_logger": JsonlLogger(cfg["logging"]["traces_file"]),
        "metric_logger": JsonlLogger(cfg["logging"]["metrics_file"]),
        "prompts": prompts,
    }

    graph = build_graph()
    state = PipelineState(prompt=prompts[0])
    final_state = graph.invoke(state, config={"configurable": {"runtime": runtime}})

    if final_state.preference_rows:
        run_dpo(student.model, student.tokenizer, final_state.preference_rows, cfg)


if __name__ == "__main__":
    main()
