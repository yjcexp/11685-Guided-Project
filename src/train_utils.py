"""Reusable training utilities for EEG classification."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data_utils import ensure_dir


@torch.no_grad()
def _to_cpu_int_list(tensor: torch.Tensor) -> List[int]:
    return tensor.detach().cpu().to(torch.int64).tolist()


def set_seed(seed: int) -> None:
    """Set seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch and return (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for eeg, labels, _, _ in tqdm(dataloader, desc="train", leave=False):
        eeg = eeg.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(eeg)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_metadata: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray, List[str], List[Dict[str, object]]]:
    """Evaluate model and optionally return row-level metadata records."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    y_true: List[int] = []
    y_pred: List[int] = []
    subject_ids: List[str] = []
    metadata_rows: List[Dict[str, object]] = []

    for eeg, labels, subj_batch, meta_batch in tqdm(dataloader, desc="eval", leave=False):
        eeg = eeg.to(device)
        labels = labels.to(device)

        logits = model(eeg)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        batch_size = labels.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_correct += int((preds == labels).sum().item())

        y_true.extend(_to_cpu_int_list(labels))
        y_pred.extend(_to_cpu_int_list(preds))
        subject_ids.extend([str(sid) for sid in subj_batch])

        if collect_metadata:
            # meta_batch is collated as dict[str, list/ tensor]
            row_indices = meta_batch["row_index"]
            session_ids = meta_batch["session_id"]
            run_ids = meta_batch["run_id"]
            trial_indices = meta_batch["trial_index"]
            image_names = meta_batch.get("image_name", [None] * batch_size)

            for i in range(batch_size):
                metadata_rows.append(
                    {
                        "row_index": int(row_indices[i]),
                        "subject_id": str(subj_batch[i]),
                        "session_id": str(session_ids[i]),
                        "run_id": str(run_ids[i]),
                        "trial_index": int(trial_indices[i]),
                        "image_name": None if image_names[i] is None else str(image_names[i]),
                    }
                )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return (
        avg_loss,
        avg_acc,
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        subject_ids,
        metadata_rows,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: Dict[str, object],
) -> None:
    """Save model checkpoint."""
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
