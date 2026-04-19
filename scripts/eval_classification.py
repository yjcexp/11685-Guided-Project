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
from src.train_utils import evaluate


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(str(cfg["device"]))
    output_dir = Path(cfg["output_dir"])

    checkpoint_path = args.checkpoint or (output_dir / "checkpoints" / "best_model.pt")
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

    model = build_model(
        model_name=str(cfg["model_name"]),
        num_classes=int(cfg["num_classes"]),
        num_channels=int(cfg.get("num_channels", 122)),
        num_timesteps=int(cfg.get("num_timesteps", 500)),
        hidden_dims=cfg.get("hidden_dims", [512, 256]),
        dropout=float(cfg.get("dropout", 0.3)),
        temporal_filters=int(cfg.get("temporal_filters", 16)),
        depth_multiplier=int(cfg.get("depth_multiplier", 2)),
        separable_filters=int(cfg.get("separable_filters", 32)),
    ).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred, subject_ids, metadata_rows = evaluate(
        model,
        test_loader,
        criterion,
        device,
        collect_metadata=True,
    )

    num_classes = int(cfg["num_classes"])
    conf_mat = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_subject = compute_per_subject_accuracy(y_true, y_pred, subject_ids)

    print(f"[eval] loss={test_loss:.4f} accuracy={test_acc:.4f}")
    print("[eval] Per-subject accuracy")
    for sid, acc in per_subject.items():
        print(f"  - {sid}: {acc:.4f}")

    # Derive a per-eval tag from the config filename so that multiple
    # evaluations (baseline + ablations sharing the same model_name) do not
    # overwrite each other's outputs.
    eval_tag = args.config.stem
    if eval_tag.startswith("classification_"):
        eval_tag = eval_tag[len("classification_") :]

    predictions_dir = ensure_dir(output_dir / "predictions" / eval_tag)
    figures_dir = ensure_dir(output_dir / "figures" / eval_tag)
    logs_dir = ensure_dir(output_dir / "logs" / eval_tag)

    pred_df = pd.DataFrame(metadata_rows)
    pred_df["true_label"] = y_true
    pred_df["pred_label"] = y_pred
    pred_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    np.save(figures_dir / "confusion_matrix.npy", conf_mat)
    pd.DataFrame(conf_mat).to_csv(figures_dir / "confusion_matrix.csv", index=False)

    metrics_payload = {
        "eval_tag": eval_tag,
        "config": str(args.config),
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "num_samples": int(len(y_true)),
        "checkpoint": str(checkpoint_path),
        "split_csv": str(split_csv),
        "per_subject_accuracy": {k: float(v) for k, v in per_subject.items()},
    }
    with (logs_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print(f"[eval] Saved predictions: {predictions_dir / 'test_predictions.csv'}")
    print(f"[eval] Saved confusion matrix: {figures_dir / 'confusion_matrix.npy'}")
    print(f"[eval] Saved metrics JSON: {logs_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
