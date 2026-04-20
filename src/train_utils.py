"""Reusable training utilities for EEG classification."""

from __future__ import annotations

import inspect
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data_utils import ensure_dir


SUBJECT_AWARE_MODEL_NAMES = {
    "subject_aware_cnn_transformer",
    "cnn_transformer_subject_head",
    "multihead_cnn_transformer",
    "subject_embedding_cnn_transformer",
    "cnn_transformer_subject_embedding",
    "subject_conditioned_cnn_transformer",
    "eegnet_subject_embedding",
    "subject_embedding_eegnet",
    "subject_conditioned_eegnet",
    "subject_conditioned_multiscale_eegnet",
    "multiscale_eegnet_subject_conditioned",
    "subject_conditioned_multiscale_eegnet_classifier",
}


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


def _subject_sort_key(subject_id: object) -> Tuple[int, str]:
    text = str(subject_id).strip()
    match = re.fullmatch(r"sub-(\d+)", text)
    if match is not None:
        return int(match.group(1)), text
    if text.isdigit():
        return int(text), text
    return 10**9, text


def build_subject_id_mapping(subject_ids: Sequence[object]) -> Dict[str, int]:
    """Build a stable zero-based subject mapping from the actual dataset ids."""
    unique_subject_ids = sorted({str(subject_id).strip() for subject_id in subject_ids}, key=_subject_sort_key)
    return {subject_id: idx for idx, subject_id in enumerate(unique_subject_ids)}


def model_requires_subject_ids(model_name: str) -> bool:
    """Return whether a model is expected to consume subject_ids."""
    return model_name in SUBJECT_AWARE_MODEL_NAMES


def encode_subject_ids(
    subject_ids: Sequence[object],
    device: torch.device,
    subject_id_to_index: Mapping[str, int] | None = None,
) -> torch.Tensor:
    """Convert raw subject identifiers into zero-based integer subject indices."""
    encoded: List[int] = []
    for subject_id in subject_ids:
        if isinstance(subject_id, torch.Tensor):
            value = int(subject_id.item())
        else:
            text = str(subject_id).strip()
            if subject_id_to_index is not None:
                if text not in subject_id_to_index:
                    known_subjects = sorted(subject_id_to_index.keys(), key=_subject_sort_key)
                    raise ValueError(
                        f"Unknown subject_id={text!r}. Known subjects from training data: {known_subjects}"
                    )
                value = int(subject_id_to_index[text])
            elif text.isdigit():
                value = int(text)
            else:
                raise ValueError(
                    f"Unable to encode subject_id={subject_id!r}. "
                    "Expected a subject_id_to_index mapping or integer-like subject ids."
                )
        encoded.append(value)
    return torch.tensor(encoded, dtype=torch.long, device=device)


def forward_model(
    model: nn.Module,
    eeg: torch.Tensor,
    subject_ids: Sequence[object] | None = None,
    subject_id_to_index: Mapping[str, int] | None = None,
    requires_subject_ids: bool | None = None,
) -> torch.Tensor:
    """Call the model with or without subject_ids."""
    if requires_subject_ids is None:
        signature = inspect.signature(model.forward)
        requires_subject_ids = "subject_ids" in signature.parameters

    if requires_subject_ids:
        if subject_ids is None:
            raise ValueError("Model forward requires subject_ids, but no subject_ids were provided.")
        encoded_subject_ids = encode_subject_ids(
            subject_ids,
            device=eeg.device,
            subject_id_to_index=subject_id_to_index,
        )
        return model(eeg, encoded_subject_ids)
    return model(eeg)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    subject_id_to_index: Mapping[str, int] | None = None,
    requires_subject_ids: bool | None = None,
) -> Tuple[float, float]:
    """Train for one epoch and return (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for eeg, labels, subject_ids, _ in tqdm(dataloader, desc="train", leave=False):
        eeg = eeg.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(
            model,
            eeg,
            subject_ids,
            subject_id_to_index=subject_id_to_index,
            requires_subject_ids=requires_subject_ids,
        )
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
    subject_id_to_index: Mapping[str, int] | None = None,
    requires_subject_ids: bool | None = None,
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

        logits = forward_model(
            model,
            eeg,
            subj_batch,
            subject_id_to_index=subject_id_to_index,
            requires_subject_ids=requires_subject_ids,
        )
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
