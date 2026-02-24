from __future__ import annotations

from typing import Any, Dict

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def run_sft(model, tokenizer, cfg: Dict[str, Any]) -> SFTTrainer:
    sft_cfg = cfg["training"]["sft"]
    dataset = load_dataset("json", data_files=sft_cfg["dataset_path"], split="train")
    trainer_cfg = SFTConfig(
        output_dir=sft_cfg["output_dir"],
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        max_seq_length=sft_cfg["max_seq_length"],
        logging_steps=10,
        save_steps=100,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=trainer_cfg,
        dataset_text_field="text",
    )
    trainer.train()
    return trainer
