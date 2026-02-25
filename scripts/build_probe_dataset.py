from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.embedding_extractor import HiddenStateExtractor, pooled_layer_features


def _load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    training_cfg = yaml.safe_load(Path("config/training_config.yaml").read_text())
    probe_cfg = yaml.safe_load(Path("config/probe_config.yaml").read_text())

    model_name = training_cfg["models"]["student_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    layers = training_cfg["probe"]["layers"]
    extractor = HiddenStateExtractor(model, layers)

    sentence_rows = _load_rows("data/probe_training/probe_sentences.jsonl")
    output_rows = []
    for row in sentence_rows:
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            hidden_by_layer = extractor.run(**inputs)
            features = pooled_layer_features(hidden_by_layer).squeeze(0).cpu().tolist()
        output_rows.append({"text": row["text"], "features": features, "label": row["label"]})

    output_path = Path(probe_cfg["training"]["dataset_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r) for r in output_rows) + "\n")
    print({"rows": len(output_rows), "saved_to": str(output_path), "layers": layers})


if __name__ == "__main__":
    main()
