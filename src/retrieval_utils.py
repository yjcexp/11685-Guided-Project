"""Utilities for EEG-caption retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.train_utils import call_model_method


@dataclass
class RetrievalBatchOutput:
    eeg_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    metadata_rows: List[Dict[str, object]]


def compute_similarity_matrix(
    eeg_embeddings: torch.Tensor | np.ndarray,
    text_embeddings: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix after L2 normalization."""
    if isinstance(eeg_embeddings, np.ndarray):
        eeg_embeddings = torch.from_numpy(eeg_embeddings)
    if isinstance(text_embeddings, np.ndarray):
        text_embeddings = torch.from_numpy(text_embeddings)
    eeg_embeddings = F.normalize(eeg_embeddings.float(), dim=-1)
    text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
    return torch.matmul(eeg_embeddings, text_embeddings.T).cpu().numpy()


def compute_recall_at_k(similarity: np.ndarray, k_values: Sequence[int]) -> Dict[str, float]:
    """Compute EEG-to-caption recall@k with diagonal positives."""
    num_queries = similarity.shape[0]
    order = np.argsort(-similarity, axis=1)
    metrics: Dict[str, float] = {}
    for k in k_values:
        hits = sum(int(i in order[i, :k]) for i in range(num_queries))
        metrics[f"recall@{k}"] = hits / max(num_queries, 1)
    return metrics


def compute_precision_at_k(similarity: np.ndarray, k_values: Sequence[int]) -> Dict[str, float]:
    """Compute EEG-to-caption precision@k with diagonal positives."""
    num_queries = similarity.shape[0]
    order = np.argsort(-similarity, axis=1)
    metrics: Dict[str, float] = {}
    for k in k_values:
        precision_sum = 0.0
        for i in range(num_queries):
            precision_sum += float(i in order[i, :k]) / float(k)
        metrics[f"precision@{k}"] = precision_sum / max(num_queries, 1)
    return metrics


def compute_match_mismatch_clipscore(
    similarity: np.ndarray,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """Compute mean similarity for matched and mismatched EEG-caption pairs."""
    rng = np.random.default_rng(rng_seed)
    num_samples = similarity.shape[0]
    matched = np.diag(similarity)
    mismatched_indices = []
    for i in range(num_samples):
        choices = np.arange(num_samples)
        choices = choices[choices != i]
        mismatched_indices.append(int(rng.choice(choices)))
    mismatched = np.asarray([similarity[i, j] for i, j in enumerate(mismatched_indices)], dtype=np.float32)
    retrieved = similarity[np.arange(num_samples), np.argmax(similarity, axis=1)]
    return {
        "clipscore_matched_mean": float(matched.mean()),
        "clipscore_matched_std": float(matched.std()),
        "clipscore_mismatched_mean": float(mismatched.mean()),
        "clipscore_mismatched_std": float(mismatched.std()),
        "clipscore_retrieved_mean": float(retrieved.mean()),
        "clipscore_retrieved_std": float(retrieved.std()),
    }


@torch.no_grad()
def extract_retrieval_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    subject_id_to_index: Mapping[str, int] | None = None,
) -> RetrievalBatchOutput:
    """Extract normalized EEG and text embeddings for a retrieval split."""
    model.eval()
    eeg_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []
    metadata_rows: List[Dict[str, object]] = []

    for eeg, captions, subject_ids, meta_batch in dataloader:
        eeg = eeg.to(device)
        eeg_batch = call_model_method(
            model,
            "encode_eeg",
            eeg,
            subject_ids=subject_ids,
            subject_id_to_index=subject_id_to_index,
        )
        text_batch = model.encode_text(list(captions), device=device)
        eeg_embeddings.append(eeg_batch.detach().cpu())
        text_embeddings.append(text_batch.detach().cpu())

        batch_size = eeg.shape[0]
        row_indices = meta_batch["row_index"]
        session_ids = meta_batch["session_id"]
        run_ids = meta_batch["run_id"]
        trial_indices = meta_batch["trial_index"]
        image_names = meta_batch.get("image_name", [None] * batch_size)
        class_labels = meta_batch.get("class_label", [None] * batch_size)
        for i in range(batch_size):
            metadata_rows.append(
                {
                    "row_index": int(row_indices[i]),
                    "subject_id": str(subject_ids[i]),
                    "session_id": str(session_ids[i]),
                    "run_id": str(run_ids[i]),
                    "trial_index": int(trial_indices[i]),
                    "image_name": None if image_names[i] is None else str(image_names[i]),
                    "class_label": None if class_labels[i] is None else int(class_labels[i]),
                    "caption": str(captions[i]),
                }
            )

    return RetrievalBatchOutput(
        eeg_embeddings=torch.cat(eeg_embeddings, dim=0),
        text_embeddings=torch.cat(text_embeddings, dim=0),
        metadata_rows=metadata_rows,
    )


def count_trainable_parameters(model: torch.nn.Module) -> Dict[str, int | float]:
    """Return trainable and total parameter counts."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_fraction": float(trainable / max(total, 1)),
    }
