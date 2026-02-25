from __future__ import annotations

from typing import Any, Dict

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def _format_example(row: Dict[str, str]) -> Dict[str, str]:
    if "text" in row:
        return {"text": row["text"].strip()}
    prompt = row["prompt"].strip()
    response = row["response"].strip()
    return {"text": f"{prompt}\n\n{response}"}


def run_sft(model, tokenizer, cfg: Dict[str, Any]) -> SFTTrainer:
    sft_cfg = cfg["training"]["sft"]
    dataset = load_dataset("json", data_files=sft_cfg["dataset_path"], split="train")
    dataset = dataset.map(_format_example, remove_columns=dataset.column_names)
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
