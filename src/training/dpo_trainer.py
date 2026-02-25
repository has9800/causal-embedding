from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from trl import DPOConfig, DPOTrainer


def build_preference_dataset(rows: List[Dict[str, str]]) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame(rows))


def run_dpo(
    model,
    tokenizer,
    rows: List[Dict[str, str]],
    cfg: Dict[str, Any],
    output_dir: str | None = None,
) -> DPOTrainer:
    dpo_cfg = cfg["training"]["dpo"]
    dataset = build_preference_dataset(rows)
    target_output_dir = output_dir or dpo_cfg["output_dir"]
    Path(target_output_dir).mkdir(parents=True, exist_ok=True)

    trainer_cfg = DPOConfig(
        output_dir=target_output_dir,
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        beta=dpo_cfg["beta"],
        logging_steps=10,
        save_steps=dpo_cfg["checkpoint_every_steps"],
    )
    trainer = DPOTrainer(
        model=model,
        args=trainer_cfg,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(target_output_dir)
    return trainer
