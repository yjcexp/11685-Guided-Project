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
from src.datasets import EEGClassificationDataset, build_normalization_state
from src.metrics import compute_confusion_matrix, compute_per_subject_accuracy
from src.models import build_model
from src.train_utils import build_subject_id_mapping, evaluate, model_requires_subject_ids


EXPECTED_SUBMISSION_ROWS = 26000


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
    parser.add_argument(
        "--submission_csv",
        type=Path,
        default=None,
        help="Optional path to save Kaggle-style submission CSV.",
    )
    parser.add_argument(
        "--include_category_name",
        action="store_true",
        help="Include a third CategoryName column in the submission CSV.",
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
    candidate_dirs = [path for path in checkpoints_root.glob(pattern) if path.is_dir()]
    candidate_dirs = sorted(candidate_dirs, key=lambda path: path.name, reverse=True)
    for candidate_dir in candidate_dirs:
        candidate_checkpoint = candidate_dir / "best_model.pt"
        if candidate_checkpoint.exists():
            return candidate_checkpoint

    direct_path = checkpoints_root / "best_model.pt"
    if direct_path.exists():
        return direct_path

    return direct_path


def resolve_eval_run_name(checkpoint_path: Path, model_name: str) -> str:
    checkpoint_parent = checkpoint_path.parent
    if checkpoint_parent.name.startswith(f"{model_name}-"):
        return checkpoint_parent.name
    return f"{model_name}-eval"


def resolve_split_csv(args: argparse.Namespace, cfg: Dict[str, object]) -> Path:
    if args.split_csv is not None:
        return args.split_csv
    if args.submission_csv is not None:
        metadata_csv = Path(str(cfg.get("metadata_csv", "data/processed/metadata.csv")))
        if metadata_csv.exists():
            return metadata_csv
    return Path(str(cfg["test_split_csv"]))


def build_per_class_accuracy_df(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> pd.DataFrame:
    rows = []
    for class_label in range(num_classes):
        mask = y_true == class_label
        num_samples = int(mask.sum())
        accuracy = float((y_pred[mask] == y_true[mask]).mean()) if num_samples > 0 else float("nan")
        rows.append(
            {
                "class_label": class_label,
                "num_samples": num_samples,
                "accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def build_predicted_label_distribution_df(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> pd.DataFrame:
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    rows = []
    for class_label in range(num_classes):
        rows.append(
            {
                "class_label": class_label,
                "true_count": int(true_counts[class_label]),
                "pred_count": int(pred_counts[class_label]),
                "pred_fraction": float(pred_counts[class_label] / max(len(y_pred), 1)),
            }
        )
    return pd.DataFrame(rows)


def build_per_subject_metrics_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, subject_df in pred_df.groupby("subject_id", sort=True):
        num_samples = int(len(subject_df))
        accuracy = float((subject_df["true_label"] == subject_df["pred_label"]).mean()) if num_samples > 0 else float("nan")
        rows.append(
            {
                "subject_id": str(subject_id),
                "num_samples": num_samples,
                "accuracy": accuracy,
                "num_true_classes": int(subject_df["true_label"].nunique()),
                "num_pred_classes": int(subject_df["pred_label"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_submission_dataframe(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    include_category_name: bool = False,
) -> pd.DataFrame:
    fallback_trial_ids = (
        pred_df["subject_id"].astype(str)
        + "_"
        + pred_df["session_id"].astype(str)
        + "_"
        + pred_df["run_id"].astype(str)
        + "_"
        + pred_df["trial_index"].astype(int).astype(str)
    )

    if "Id" in pred_df.columns:
        raw_ids = pred_df["Id"]
    elif "id" in pred_df.columns:
        raw_ids = pred_df["id"]
    else:
        raw_ids = None

    if raw_ids is None:
        trial_ids = fallback_trial_ids
    else:
        normalized_ids = raw_ids.astype("string").str.strip()
        invalid_mask = normalized_ids.isna() | normalized_ids.isin({"", "None", "nan", "<NA>"})
        trial_ids = normalized_ids.mask(invalid_mask, fallback_trial_ids).astype(str)

    submission_df = pd.DataFrame(
        {
            "Id": trial_ids,
            "Category": pred_df["pred_label"].astype(int),
        }
    )
    if include_category_name:
        label_to_name_map: dict[int, str] = {}
        if {"class_label", "class_name"}.issubset(train_df.columns):
            mapping_df = (
                train_df[["class_label", "class_name"]]
                .dropna()
                .drop_duplicates()
                .sort_values("class_label")
            )
            label_to_name_map = {
                int(row["class_label"]): str(row["class_name"])
                for _, row in mapping_df.iterrows()
            }
        submission_df["CategoryName"] = submission_df["Category"].map(label_to_name_map).fillna("")
    return submission_df


def validate_submission_dataframe(submission_df: pd.DataFrame) -> None:
    required_columns = ["Id", "Category"]
    if list(submission_df.columns[:2]) != required_columns:
        raise ValueError(
            f"Submission CSV must start with columns {required_columns}, got {list(submission_df.columns)}"
        )
    if len(submission_df) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(
            "Submission row count mismatch. "
            f"Expected {EXPECTED_SUBMISSION_ROWS} rows, got {len(submission_df)}. "
            "If no official separate Kaggle test file is provided, use the full metadata CSV."
        )
    if submission_df["Id"].isna().any() or (submission_df["Id"].astype(str).str.strip() == "").any():
        raise ValueError("Submission contains empty Id values.")
    if submission_df["Id"].nunique() != len(submission_df):
        raise ValueError("Submission contains duplicated Id values.")
    categories = submission_df["Category"].astype(int)
    invalid_mask = (categories < 0) | (categories > 19)
    if invalid_mask.any():
        invalid_values = sorted(set(categories[invalid_mask].tolist()))
        raise ValueError(f"Submission contains Category values outside [0, 19]: {invalid_values}")


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(str(cfg["device"]))
    output_dir = Path(cfg["output_dir"])

    model_name = str(cfg["model_name"])
    resolved_num_timesteps = resolve_num_timesteps(cfg)
    requires_subject_ids = model_requires_subject_ids(model_name)
    checkpoint_path = resolve_checkpoint_path(output_dir, model_name, args.checkpoint)
    split_csv = resolve_split_csv(args, cfg)
    split_name = Path(split_csv).stem

    train_df = pd.read_csv(cfg["train_split_csv"])
    normalization = str(cfg.get("normalization", "none"))
    normalization_state = build_normalization_state(
        train_df,
        normalization=normalization,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )
    test_df = pd.read_csv(split_csv)
    test_dataset = EEGClassificationDataset(
        test_df,
        normalization=normalization,
        normalization_state=normalization_state,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
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
        subject_id_to_index = build_subject_id_mapping(train_df["subject_id"].astype(str).tolist())

    model = build_model(
        model_name=model_name,
        num_classes=int(cfg["num_classes"]),
        num_channels=int(cfg.get("num_channels", 122)),
        num_timesteps=resolved_num_timesteps,
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
        classifier_head_dropout=float(cfg.get("classifier_head_dropout", 0.2)),
        fuse_mode=str(cfg.get("fuse_mode", "concat")),
    ).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_cfg = checkpoint.get("config", {})
    checkpoint_model_name = None
    if isinstance(checkpoint_cfg, dict):
        checkpoint_model_name = checkpoint_cfg.get("model_name")
    if checkpoint_model_name is not None and str(checkpoint_model_name) != model_name:
        raise RuntimeError(
            "Checkpoint model_name does not match the requested config model_name. "
            f"Requested {model_name!r}, but checkpoint at {checkpoint_path} was saved for "
            f"{checkpoint_model_name!r}. Pass --checkpoint explicitly or use the matching config."
        )
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

    eval_run_name = resolve_eval_run_name(checkpoint_path, model_name)
    predictions_dir = ensure_dir(output_dir / "predictions" / eval_run_name)
    figures_dir = ensure_dir(output_dir / "figures" / eval_run_name)
    logs_dir = ensure_dir(output_dir / "logs" / eval_run_name)

    pred_df = pd.DataFrame(metadata_rows)
    pred_df["true_label"] = y_true
    pred_df["pred_label"] = y_pred
    predictions_path = predictions_dir / f"{split_name}_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    submission_path = args.submission_csv or (predictions_dir / f"{split_name}_submission.csv")
    submission_df = build_submission_dataframe(
        pred_df=pred_df,
        train_df=train_df,
        include_category_name=bool(args.include_category_name),
    )
    if args.submission_csv is not None:
        validate_submission_dataframe(submission_df)
    submission_df.to_csv(submission_path, index=False)

    per_class_accuracy_df = build_per_class_accuracy_df(y_true, y_pred, num_classes=num_classes)
    predicted_label_distribution_df = build_predicted_label_distribution_df(y_true, y_pred, num_classes=num_classes)
    per_subject_metrics_df = build_per_subject_metrics_df(pred_df)

    per_class_accuracy_path = logs_dir / f"{split_name}_per_class_accuracy.csv"
    predicted_label_distribution_path = logs_dir / f"{split_name}_predicted_label_distribution.csv"
    per_subject_metrics_path = logs_dir / f"{split_name}_per_subject_metrics.csv"
    per_class_accuracy_df.to_csv(per_class_accuracy_path, index=False)
    predicted_label_distribution_df.to_csv(predicted_label_distribution_path, index=False)
    per_subject_metrics_df.to_csv(per_subject_metrics_path, index=False)

    confusion_matrix_npy_path = figures_dir / f"{split_name}_confusion_matrix.npy"
    confusion_matrix_csv_path = figures_dir / f"{split_name}_confusion_matrix.csv"
    np.save(confusion_matrix_npy_path, conf_mat)
    pd.DataFrame(conf_mat).to_csv(confusion_matrix_csv_path, index=False)

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
    metrics_json_path = logs_dir / f"{split_name}_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print(f"[eval] Saved predictions: {predictions_path}")
    print(f"[eval] Saved submission CSV: {submission_path}")
    print(f"[eval] Saved confusion matrix: {confusion_matrix_npy_path}")
    print(f"[eval] Saved metrics JSON: {metrics_json_path}")
    print(f"[eval] Saved per-class accuracy: {per_class_accuracy_path}")
    print(f"[eval] Saved predicted-label distribution: {predicted_label_distribution_path}")
    print(f"[eval] Saved per-subject metrics: {per_subject_metrics_path}")


if __name__ == "__main__":
    main()
