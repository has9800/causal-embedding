from __future__ import annotations

from pathlib import Path

import yaml

from src.models.student import StudentModelFactory
from src.training.sft_trainer import run_sft


def main() -> None:
    cfg = yaml.safe_load(Path("config/training_config.yaml").read_text())
    factory = StudentModelFactory(cfg)
    student = factory.load(production=False)
    trainer = run_sft(student.model, student.tokenizer, cfg)
    trainer.save_model(cfg["training"]["sft"]["output_dir"])


if __name__ == "__main__":
    main()
