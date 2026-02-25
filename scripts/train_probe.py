from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig

from src.models.probe import CausalProbe, train_probe


def _compute_probe_input_dim(training_cfg: dict, probe_layers: list[int]) -> int:
    hidden_size = AutoConfig.from_pretrained(training_cfg["models"]["student_model_name"]).hidden_size
    return int(hidden_size) * len(probe_layers)


def main() -> None:
    probe_cfg = yaml.safe_load(Path("config/probe_config.yaml").read_text())
    training_cfg = yaml.safe_load(Path("config/training_config.yaml").read_text())

    rows = [json.loads(line) for line in Path(probe_cfg["training"]["dataset_path"]).read_text().splitlines() if line.strip()]
    input_dim = _compute_probe_input_dim(training_cfg, training_cfg["probe"]["layers"])

    model = CausalProbe(
        input_dim=input_dim,
        num_classes=probe_cfg["probe"]["num_classes"],
    )
    result = train_probe(
        model=model,
        dataset_rows=rows,
        epochs=probe_cfg["training"]["epochs"],
        batch_size=probe_cfg["training"]["batch_size"],
        learning_rate=probe_cfg["training"]["learning_rate"],
        val_split=probe_cfg["training"]["val_split"],
    )
    output_path = Path(probe_cfg["training"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print({"train_loss": result.train_loss, "val_accuracy": result.val_accuracy, "saved_to": str(output_path), "input_dim": input_dim})


if __name__ == "__main__":
    main()
