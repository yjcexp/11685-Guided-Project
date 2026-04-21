"""Evaluate Task 2B EEG-to-caption retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_utils import ensure_dir
from src.datasets import EEGRetrievalDataset, build_normalization_state
from src.models import EEGTextRetrievalModel
from src.retrieval_utils import (
    compute_match_mismatch_clipscore,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_similarity_matrix,
    count_trainable_parameters,
    extract_retrieval_embeddings,
)
from src.train_utils import build_subject_id_mapping, model_requires_subject_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EEG-to-caption retrieval model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to retrieval config YAML.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to retrieval checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_num_timesteps(cfg: Dict[str, object]) -> int:
    total_timesteps = int(cfg.get("num_timesteps", 500))
    time_window_start = int(cfg.get("time_window_start", 0))
    time_window_end = cfg.get("time_window_end")
    end = total_timesteps if time_window_end is None else int(time_window_end)
    return end - time_window_start


def resolve_checkpoint_path(output_dir: Path, model_name: str, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path
    candidate_dirs = sorted(
        [path for path in (output_dir / "checkpoints").glob(f"{model_name}-*") if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate_dir in candidate_dirs:
        path = candidate_dir / "best_model.pt"
        if path.exists():
            return path
    return output_dir / "checkpoints" / "best_model.pt"


def build_retrieval_model(cfg: Dict[str, object]) -> EEGTextRetrievalModel:
    eeg_model_kwargs = {
        "num_classes": int(cfg.get("num_classes", 20)),
        "num_channels": int(cfg.get("num_channels", 122)),
        "num_timesteps": resolve_num_timesteps(cfg),
        "temporal_filters": int(cfg.get("temporal_filters", 16)),
        "depth_multiplier": int(cfg.get("depth_multiplier", 2)),
        "separable_filters": int(cfg.get("separable_filters", 32)),
        "dropout": float(cfg.get("dropout", 0.25)),
        "embedding_dim": int(cfg.get("embedding_dim", 256)),
        "projection_dim": cfg.get("projection_dim"),
        "projection_hidden_dim": cfg.get("projection_hidden_dim"),
        "projection_dropout": float(cfg.get("projection_dropout", 0.1)),
        "normalize_projected_embedding": bool(cfg.get("normalize_projected_embedding", True)),
    }
    for optional_key in [
        "temporal_kernel_size",
        "separable_kernel_size",
        "pool1_kernel_size",
        "pool2_kernel_size",
        "num_refinement_blocks",
        "refinement_kernel_size",
        "use_gated_pooling",
    ]:
        if optional_key in cfg:
            eeg_model_kwargs[optional_key] = cfg[optional_key]
    return EEGTextRetrievalModel(
        eeg_model_name=str(cfg["eeg_model_name"]),
        eeg_model_kwargs=eeg_model_kwargs,
        clip_model_name=str(cfg.get("clip_model_name", "openai/clip-vit-base-patch32")),
        clip_train_mode=str(cfg.get("clip_train_mode", "frozen")),
        partial_unfreeze_last_n_layers=int(cfg.get("partial_unfreeze_last_n_layers", 2)),
        use_text_projection=bool(cfg.get("use_text_projection", True)),
        lora_r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.1)),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(str(cfg.get("device", "cuda")))
    output_dir = Path(cfg.get("output_dir", "outputs"))
    model_name = str(cfg.get("model_name", "task2b_retrieval"))
    checkpoint_path = resolve_checkpoint_path(output_dir, model_name, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_csv = Path(cfg[f"{args.split}_split_csv"])
    train_df = pd.read_csv(cfg["train_split_csv"])
    split_df = pd.read_csv(split_csv)
    normalization = str(cfg.get("normalization", "none"))
    normalization_state = build_normalization_state(
        train_df,
        normalization=normalization,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )
    dataset = EEGRetrievalDataset(
        split_df,
        normalization=normalization,
        normalization_state=normalization_state,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    model = build_retrieval_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    subject_id_to_index = None
    if model_requires_subject_ids(str(cfg["eeg_model_name"])):
        subject_id_to_index = build_subject_id_mapping(train_df["subject_id"].astype(str).tolist())

    extracted = extract_retrieval_embeddings(
        model,
        dataloader,
        device=device,
        subject_id_to_index=subject_id_to_index,
    )
    similarity = compute_similarity_matrix(extracted.eeg_embeddings, extracted.text_embeddings)
    recall = compute_recall_at_k(similarity, k_values=[1, 5, 10])
    precision = compute_precision_at_k(similarity, k_values=[1, 5, 10])
    clipscore = compute_match_mismatch_clipscore(similarity)
    param_stats = count_trainable_parameters(model)

    run_name = checkpoint_path.parent.name if checkpoint_path.parent.name.startswith(f"{model_name}-") else f"{model_name}-eval"
    pred_dir = ensure_dir(output_dir / "predictions" / run_name)
    log_dir = ensure_dir(output_dir / "logs" / run_name)

    eeg_path = pred_dir / f"{args.split}_eeg_embeddings.pt"
    text_path = pred_dir / f"{args.split}_text_embeddings.pt"
    sim_path = pred_dir / f"{args.split}_similarity.npy"
    meta_path = pred_dir / f"{args.split}_retrieval_metadata.csv"
    torch.save(extracted.eeg_embeddings, eeg_path)
    torch.save(extracted.text_embeddings, text_path)
    pd.DataFrame(extracted.metadata_rows).to_csv(meta_path, index=False)
    import numpy as np
    np.save(sim_path, similarity)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split_csv": str(split_csv),
        **recall,
        **precision,
        **clipscore,
        **param_stats,
    }
    metrics_path = log_dir / f"{args.split}_retrieval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"[retrieval-eval] checkpoint={checkpoint_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"[retrieval-eval] Saved metrics: {metrics_path}")
    print(f"[retrieval-eval] Saved EEG embeddings: {eeg_path}")
    print(f"[retrieval-eval] Saved text embeddings: {text_path}")
    print(f"[retrieval-eval] Saved similarity matrix: {sim_path}")
    print(f"[retrieval-eval] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
