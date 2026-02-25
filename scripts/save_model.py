from __future__ import annotations

from pathlib import Path

import yaml
from peft import PeftModel

from src.models.student import StudentModelFactory


def _latest_round_dir(dpo_root: Path) -> Path:
    round_dirs = sorted(
        [path for path in dpo_root.glob("round_*") if path.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not round_dirs:
        raise FileNotFoundError(f"No round checkpoints found under: {dpo_root}")
    return round_dirs[-1]


def main() -> None:
    cfg = yaml.safe_load(Path("config/training_config.yaml").read_text())
    factory = StudentModelFactory(cfg)

    trained_model = cfg["models"].get("trained_model", "iteration")
    is_production = trained_model == "production"
    bundle = factory.load(production=is_production)

    dpo_dir = Path(cfg["training"]["dpo"]["output_dir"])
    if not dpo_dir.exists():
        raise FileNotFoundError(f"DPO output dir not found: {dpo_dir}")

    latest_round_dir = _latest_round_dir(dpo_dir)
    bundle.model = PeftModel.from_pretrained(bundle.model, str(latest_round_dir))

    adapter_dir = Path("outputs/final/adapter")
    merged_dir = Path("outputs/final/merged")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    factory.save_adapter(bundle, str(adapter_dir))
    factory.merge_and_save(bundle, str(merged_dir))
    print(
        {
            "adapter": str(adapter_dir),
            "merged": str(merged_dir),
            "loaded_adapter_from": str(latest_round_dir),
            "trained_model": trained_model,
        }
    )


if __name__ == "__main__":
    main()
