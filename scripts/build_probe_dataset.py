from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.embedding_extractor import HiddenStateExtractor, pooled_layer_features


LABEL_TO_ID = {"causal": 0, "correlational": 1, "unrelated": 2}


def _load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _batched(rows: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
    batch_size = int(probe_cfg["training"].get("extract_batch_size", 8))
    for row_batch in _batched(sentence_rows, batch_size=batch_size):
        sentences = [row.get("sentence", row.get("text", "")) for row in row_batch]
        labels = []
        for row in row_batch:
            label_value = row["label"]
            if isinstance(label_value, str):
                labels.append(LABEL_TO_ID[label_value])
            else:
                labels.append(int(label_value))
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            hidden_by_layer = extractor.run(**inputs)
            features = pooled_layer_features(hidden_by_layer).cpu().tolist()
        for sentence, label, row_features in zip(sentences, labels, features):
            output_rows.append({"sentence": sentence, "features": row_features, "label": label})

    output_path = Path(probe_cfg["training"]["dataset_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r) for r in output_rows) + "\n")
    print({"rows": len(output_rows), "saved_to": str(output_path), "layers": layers})


if __name__ == "__main__":
    main()
