"""Extract EEG embeddings and projected embeddings for a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_utils import ensure_dir
from src.datasets import EEGClassificationDataset, build_normalization_state
from src.models import build_model
from src.train_utils import build_subject_id_mapping, call_model_method, model_requires_subject_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract EEG embeddings for a split.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to the latest matching run for the configured model.",
    )
    parser.add_argument(
        "--split_csv",
        type=Path,
        default=None,
        help="Override split CSV path. Defaults to test_split_csv from config.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_checkpoint_path(output_dir: Path, model_name: str, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path

    checkpoints_root = output_dir / "checkpoints"
    pattern = f"{model_name}-*"
    candidate_dirs = sorted(
        [path for path in checkpoints_root.glob(pattern) if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate_dir in candidate_dirs:
        candidate_checkpoint = candidate_dir / "best_model.pt"
        if candidate_checkpoint.exists():
            return candidate_checkpoint

    direct_path = checkpoints_root / "best_model.pt"
    return direct_path


def resolve_run_name(checkpoint_path: Path, model_name: str) -> str:
    checkpoint_parent = checkpoint_path.parent
    if checkpoint_parent.name.startswith(f"{model_name}-"):
        return checkpoint_parent.name
    return f"{model_name}-extract"


def resolve_num_timesteps(cfg: Dict[str, object]) -> int:
    total_timesteps = int(cfg.get("num_timesteps", 500))
    time_window_start = int(cfg.get("time_window_start", 0))
    time_window_end = cfg.get("time_window_end")
    end = total_timesteps if time_window_end is None else int(time_window_end)
    if not (0 <= time_window_start < end <= total_timesteps):
        raise ValueError(
            "Invalid time window configuration. "
            f"Expected 0 <= start < end <= {total_timesteps}, got start={time_window_start}, end={end}."
        )
    return end - time_window_start


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(str(cfg["device"]))
    output_dir = Path(cfg["output_dir"])
    model_name = str(cfg["model_name"])
    split_csv = args.split_csv or Path(cfg["test_split_csv"])
    split_name = Path(split_csv).stem
    checkpoint_path = resolve_checkpoint_path(output_dir, model_name, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_df = pd.read_csv(cfg["train_split_csv"])
    normalization = str(cfg.get("normalization", "none"))
    normalization_state = build_normalization_state(
        train_df,
        normalization=normalization,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )
    split_df = pd.read_csv(split_csv)
    dataset = EEGClassificationDataset(
        split_df,
        normalization=normalization,
        normalization_state=normalization_state,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    requires_subject_ids = model_requires_subject_ids(model_name)
    subject_id_to_index = None
    if requires_subject_ids:
        subject_id_to_index = build_subject_id_mapping(train_df["subject_id"].astype(str).tolist())

    model = build_model(
        model_name=model_name,
        num_classes=int(cfg["num_classes"]),
        num_channels=int(cfg.get("num_channels", 122)),
        num_timesteps=resolve_num_timesteps(cfg),
        num_subjects=len(subject_id_to_index) if subject_id_to_index is not None else int(cfg.get("num_subjects", 13)),
        cnn_out_channels=int(cfg.get("cnn_out_channels", 64)),
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 8)),
        num_transformer_layers=int(cfg.get("num_transformer_layers", 2)),
        dim_feedforward=int(cfg.get("dim_feedforward", 256)),
        head_hidden_dim=cfg.get("head_hidden_dim"),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        hidden_dims=cfg.get("hidden_dims", [512, 256]),
        dropout=float(cfg.get("dropout", 0.3)),
        temporal_filters=int(cfg.get("temporal_filters", 16)),
        temporal_kernel_size=int(cfg.get("temporal_kernel_size", 32)),
        separable_kernel_size=int(cfg.get("separable_kernel_size", 8)),
        pool1_kernel_size=int(cfg.get("pool1_kernel_size", 2)),
        pool2_kernel_size=int(cfg.get("pool2_kernel_size", 4)),
        depth_multiplier=int(cfg.get("depth_multiplier", 2)),
        separable_filters=int(cfg.get("separable_filters", 32)),
        use_gated_pooling=bool(cfg.get("use_gated_pooling", True)),
        branch_kernel_sizes=cfg.get("branch_kernel_sizes", [15, 31, 63]),
        num_refinement_blocks=int(cfg.get("num_refinement_blocks", 2)),
        refinement_kernel_size=int(cfg.get("refinement_kernel_size", 15)),
        stem_pool_kernel=int(cfg.get("stem_pool_kernel", 4)),
        embedding_dim=int(cfg.get("embedding_dim", 256)),
        projection_dim=cfg.get("projection_dim"),
        projection_hidden_dim=cfg.get("projection_hidden_dim"),
        projection_dropout=float(cfg.get("projection_dropout", 0.0)),
        normalize_projected_embedding=bool(cfg.get("normalize_projected_embedding", False)),
        subject_embedding_dim=int(cfg.get("subject_embedding_dim", 32)),
        classifier_hidden_dim=int(cfg.get("classifier_hidden_dim", 128)),
        fuse_mode=str(cfg.get("fuse_mode", "concat")),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {})
    checkpoint_model_name = checkpoint_cfg.get("model_name") if isinstance(checkpoint_cfg, dict) else None
    if checkpoint_model_name is not None and str(checkpoint_model_name) != model_name:
        raise RuntimeError(
            "Checkpoint model_name does not match the requested config model_name. "
            f"Requested {model_name!r}, but checkpoint at {checkpoint_path} was saved for "
            f"{checkpoint_model_name!r}."
        )
    if requires_subject_ids and isinstance(checkpoint_cfg, dict) and "subject_id_to_index" in checkpoint_cfg:
        subject_id_to_index = {
            str(subject_id): int(subject_index)
            for subject_id, subject_index in checkpoint_cfg["subject_id_to_index"].items()
        }
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    run_name = resolve_run_name(checkpoint_path, model_name)
    embeddings_dir = ensure_dir(output_dir / "embeddings" / run_name)
    logs_dir = ensure_dir(output_dir / "logs" / run_name)

    all_embeddings: list[np.ndarray] = []
    all_projected: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    projection_available = hasattr(model, "encode_projected")

    for eeg, labels, subject_ids, meta_batch in dataloader:
        eeg = eeg.to(device)
        embeddings = call_model_method(
            model,
            "encode",
            eeg,
            subject_ids=subject_ids,
            subject_id_to_index=subject_id_to_index,
        )
        all_embeddings.append(embeddings.detach().cpu().numpy())

        if projection_available:
            projected = call_model_method(
                model,
                "encode_projected",
                eeg,
                subject_ids=subject_ids,
                subject_id_to_index=subject_id_to_index,
            )
            all_projected.append(projected.detach().cpu().numpy())

        batch_size = labels.shape[0]
        row_indices = meta_batch["row_index"]
        session_ids = meta_batch["session_id"]
        run_ids = meta_batch["run_id"]
        trial_indices = meta_batch["trial_index"]
        image_names = meta_batch.get("image_name", [None] * batch_size)
        for i in range(batch_size):
            metadata_rows.append(
                {
                    "row_index": int(row_indices[i]),
                    "subject_id": str(subject_ids[i]),
                    "session_id": str(session_ids[i]),
                    "run_id": str(run_ids[i]),
                    "trial_index": int(trial_indices[i]),
                    "image_name": None if image_names[i] is None else str(image_names[i]),
                    "class_label": int(labels[i]),
                }
            )

    embedding_array = np.concatenate(all_embeddings, axis=0)
    embedding_path = embeddings_dir / f"{split_name}_eeg_embedding.npy"
    np.save(embedding_path, embedding_array)

    projected_path = None
    if all_projected:
        projected_array = np.concatenate(all_projected, axis=0)
        projected_path = embeddings_dir / f"{split_name}_projected_embedding.npy"
        np.save(projected_path, projected_array)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = embeddings_dir / f"{split_name}_embedding_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split_csv": str(split_csv),
        "embedding_path": str(embedding_path),
        "projected_embedding_path": None if projected_path is None else str(projected_path),
        "metadata_path": str(metadata_path),
        "num_samples": int(len(metadata_df)),
        "embedding_dim": int(embedding_array.shape[1]),
        "projected_embedding_dim": None if projected_path is None else int(projected_array.shape[1]),
    }
    summary_path = logs_dir / f"{split_name}_embedding_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[extract] Saved EEG embeddings: {embedding_path}")
    if projected_path is not None:
        print(f"[extract] Saved projected embeddings: {projected_path}")
    else:
        print("[extract] Projected embeddings unavailable for this model/config.")
    print(f"[extract] Saved metadata: {metadata_path}")
    print(f"[extract] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
