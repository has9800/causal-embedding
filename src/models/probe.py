from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class ProbeDataset(Dataset):
    def __init__(self, rows: List[Dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.rows[idx]
        x = torch.tensor(item["features"], dtype=torch.float32)
        y = torch.tensor(item["label"], dtype=torch.long)
        return x, y


class CausalProbe(nn.Module):
    """Frozen linear probe for causal/correlational/unrelated classification."""

    def __init__(self, input_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class ProbeTrainingResult:
    train_loss: float
    val_accuracy: float
    best_state_dict: Dict[str, torch.Tensor]
    epoch_metrics: List[Dict[str, float]]


def train_probe(
    model: CausalProbe,
    dataset_rows: List[Dict],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
) -> ProbeTrainingResult:
    dataset = ProbeDataset(dataset_rows)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    final_loss = 0.0
    best_val_acc = float("-inf")
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epoch_metrics: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            running_loss += final_loss
            num_batches += 1

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x).argmax(dim=-1)
                correct += int((preds == y).sum().item())
                total += int(y.numel())

        val_acc = correct / max(total, 1)
        train_loss = running_loss / max(num_batches, 1)
        epoch_metrics.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_accuracy": val_acc})
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    for p in model.parameters():
        p.requires_grad = False

    return ProbeTrainingResult(
        train_loss=final_loss,
        val_accuracy=best_val_acc,
        best_state_dict=best_state_dict,
        epoch_metrics=epoch_metrics,
    )
