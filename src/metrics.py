"""Metrics for EEG classification."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor | Sequence[int]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute batch accuracy from logits and integer labels."""
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item())


def compute_confusion_matrix(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    y_pred: np.ndarray | torch.Tensor | Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix with shape [num_classes, num_classes]."""
    true_np = _to_numpy(y_true).astype(np.int64)
    pred_np = _to_numpy(y_pred).astype(np.int64)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_np, pred_np):
        if t < 0 or t >= num_classes or p < 0 or p >= num_classes:
            raise ValueError(f"Label out of range for confusion matrix: y_true={t}, y_pred={p}")
        cm[t, p] += 1
    return cm


def compute_per_subject_accuracy(
    y_true: np.ndarray | torch.Tensor | Sequence[int],
    y_pred: np.ndarray | torch.Tensor | Sequence[int],
    subject_ids: Sequence[str],
) -> Dict[str, float]:
    """Compute per-subject classification accuracy."""
    true_np = _to_numpy(y_true).astype(np.int64)
    pred_np = _to_numpy(y_pred).astype(np.int64)

    if len(true_np) != len(subject_ids) or len(pred_np) != len(subject_ids):
        raise ValueError("Lengths of y_true, y_pred, and subject_ids must match.")

    by_subject: Dict[str, list[bool]] = {}
    for t, p, sid in zip(true_np, pred_np, subject_ids):
        by_subject.setdefault(str(sid), []).append(int(t == p))

    return {sid: float(np.mean(matches)) for sid, matches in sorted(by_subject.items())}
