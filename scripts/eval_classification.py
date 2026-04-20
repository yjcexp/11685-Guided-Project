"""Evaluate Task 1 EEG classification model on a split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data_utils import ensure_dir
from src.datasets import EEGClassificationDataset
from src.metrics import compute_confusion_matrix, compute_per_subject_accuracy
from src.models import build_model
from src.train_utils import build_subject_id_mapping, evaluate, model_requires_subject_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EEG Task 1 classification baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_baseline.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to <output_dir>/checkpoints/best_model.pt",
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
    direct_path = checkpoints_root / "best_model.pt"
    if direct_path.exists():
        return direct_path

    pattern = f"{model_name}-*"
    candidate_dirs = [path for path in checkpoints_root.glob(pattern) if path.is_dir()]
    candidate_dirs = sorted(candidate_dirs, key=lambda path: path.name, reverse=True)
    for candidate_dir in candidate_dirs:
        candidate_checkpoint = candidate_dir / "best_model.pt"
        if candidate_checkpoint.exists():
            return candidate_checkpoint

    return direct_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(str(cfg["device"]))
    output_dir = Path(cfg["output_dir"])

    model_name = str(cfg["model_name"])
    requires_subject_ids = model_requires_subject_ids(model_name)
    checkpoint_path = resolve_checkpoint_path(output_dir, model_name, args.checkpoint)
    split_csv = args.split_csv or Path(cfg["test_split_csv"])

    test_df = pd.read_csv(split_csv)
    test_dataset = EEGClassificationDataset(
        test_df,
        normalization=str(cfg.get("normalization", "none")),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    subject_id_to_index = None
    if requires_subject_ids:
        train_df = pd.read_csv(cfg["train_split_csv"])
        subject_id_to_index = build_subject_id_mapping(train_df["subject_id"].astype(str).tolist())

    model = build_model(
        model_name=model_name,
        num_classes=int(cfg["num_classes"]),
        num_channels=int(cfg.get("num_channels", 122)),
        num_timesteps=int(cfg.get("num_timesteps", 500)),
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
        depth_multiplier=int(cfg.get("depth_multiplier", 2)),
        separable_filters=int(cfg.get("separable_filters", 32)),
    ).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {})
    if requires_subject_ids and isinstance(checkpoint_cfg, dict) and "subject_id_to_index" in checkpoint_cfg:
        subject_id_to_index = {
            str(subject_id): int(subject_index)
            for subject_id, subject_index in checkpoint_cfg["subject_id_to_index"].items()
        }
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred, subject_ids, metadata_rows = evaluate(
        model,
        test_loader,
        criterion,
        device,
        collect_metadata=True,
        subject_id_to_index=subject_id_to_index,
        requires_subject_ids=requires_subject_ids,
    )

    num_classes = int(cfg["num_classes"])
    conf_mat = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_subject = compute_per_subject_accuracy(y_true, y_pred, subject_ids)

    print(f"[eval] loss={test_loss:.4f} accuracy={test_acc:.4f}")
    print("[eval] Per-subject accuracy")
    for sid, acc in per_subject.items():
        print(f"  - {sid}: {acc:.4f}")

    predictions_dir = ensure_dir(output_dir / "predictions")
    figures_dir = ensure_dir(output_dir / "figures")
    logs_dir = ensure_dir(output_dir / "logs")

    pred_df = pd.DataFrame(metadata_rows)
    pred_df["true_label"] = y_true
    pred_df["pred_label"] = y_pred
    pred_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    np.save(figures_dir / "confusion_matrix.npy", conf_mat)
    pd.DataFrame(conf_mat).to_csv(figures_dir / "confusion_matrix.csv", index=False)

    metrics_payload = {
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "num_samples": int(len(y_true)),
        "checkpoint": str(checkpoint_path),
        "split_csv": str(split_csv),
        "per_subject_accuracy": {k: float(v) for k, v in per_subject.items()},
    }
    if subject_id_to_index is not None:
        metrics_payload["subject_id_to_index"] = subject_id_to_index
    with (logs_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print(f"[eval] Saved predictions: {predictions_dir / 'test_predictions.csv'}")
    print(f"[eval] Saved confusion matrix: {figures_dir / 'confusion_matrix.npy'}")
    print(f"[eval] Saved metrics JSON: {logs_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
