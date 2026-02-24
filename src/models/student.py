from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class StudentBundle:
    model: Any
    tokenizer: Any
    model_name: str


class StudentModelFactory:
    """Factory for loading base model and attaching LoRA adapters."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def load(self, production: bool = False) -> StudentBundle:
        models_cfg = self.cfg["models"]
        model_name = (
            models_cfg["production_student_model_name"]
            if production
            else models_cfg["student_model_name"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
        }
        if models_cfg.get("use_8bit", False):
            load_kwargs["load_in_8bit"] = True
        torch_dtype = models_cfg.get("torch_dtype")
        if torch_dtype == "float16":
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if models_cfg.get("use_8bit", False):
            model = prepare_model_for_kbit_training(model)

        lora_cfg = self.cfg["lora"]
        peft_cfg = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

        return StudentBundle(model=model, tokenizer=tokenizer, model_name=model_name)

    @staticmethod
    def save_adapter(bundle: StudentBundle, output_dir: str) -> None:
        bundle.model.save_pretrained(output_dir)
        bundle.tokenizer.save_pretrained(output_dir)

    @staticmethod
    def merge_and_save(bundle: StudentBundle, output_dir: str) -> None:
        merged = bundle.model.merge_and_unload()
        merged.save_pretrained(output_dir)
        bundle.tokenizer.save_pretrained(output_dir)
