from __future__ import annotations

from pathlib import Path

import yaml
from peft import PeftModel

from src.models.student import StudentModelFactory


def main() -> None:
    cfg = yaml.safe_load(Path("config/training_config.yaml").read_text())
    factory = StudentModelFactory(cfg)
    bundle = factory.load(production=True)

    dpo_dir = Path(cfg["training"]["dpo"]["output_dir"])
    if not dpo_dir.exists():
        raise FileNotFoundError(f"DPO output dir not found: {dpo_dir}")

    bundle.model = PeftModel.from_pretrained(bundle.model, str(dpo_dir))

    adapter_dir = Path("outputs/final/adapter")
    merged_dir = Path("outputs/final/merged")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    factory.save_adapter(bundle, str(adapter_dir))
    factory.merge_and_save(bundle, str(merged_dir))
    print({"adapter": str(adapter_dir), "merged": str(merged_dir), "loaded_adapter_from": str(dpo_dir)})


if __name__ == "__main__":
    main()
