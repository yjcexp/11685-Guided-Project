"""Train Task 1 EEG 20-way classification baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data_utils import ensure_dir
from src.datasets import EEGClassificationDataset
from src.models import build_model
from src.train_utils import (
    build_subject_id_mapping,
    evaluate,
    model_requires_subject_ids,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG Task 1 classification baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_baseline.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def init_wandb(cfg: Dict[str, object], output_dir: Path) -> Tuple[Optional[object], Optional[str]]:
    wandb_cfg = cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return None, None

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "wandb is enabled in config but not installed. Install with: pip install wandb"
        ) from exc

    model_name = str(cfg.get("model_name", "model"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_name = str(wandb_cfg.get("run_name") or f"{model_name}-{timestamp}")
    resolved_entity = str(wandb_cfg.get("entity", "yejc20"))
    resolved_project = str(wandb_cfg.get("project", "eeg-task1"))

    run_config = {
        "model_name": cfg.get("model_name"),
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "epochs": cfg.get("epochs"),
        "normalization": cfg.get("normalization"),
        "seed": cfg.get("seed"),
    }

    run = wandb.init(
        project=resolved_project,
        entity=resolved_entity,
        name=resolved_run_name,
        mode=str(wandb_cfg.get("mode", "online")),
        config=run_config,
        dir=str(output_dir),
    )
    return run, resolved_run_name


def save_training_plots(history_df: pd.DataFrame, figures_dir: Path) -> Dict[str, Path]:
    ensure_dir(figures_dir)

    loss_path = figures_dir / "train_val_loss.png"
    acc_path = figures_dir / "train_val_accuracy.png"

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    return {"loss_curve": loss_path, "acc_curve": acc_path}


def log_split_diagnostics(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    print("[data] Train label distribution")
    print(train_df["class_label"].value_counts().sort_index().to_string())
    print("[data] Val label distribution")
    print(val_df["class_label"].value_counts().sort_index().to_string())
    print("[data] Train samples per subject")
    print(train_df["subject_id"].value_counts().sort_index().to_string())
    print("[data] Val samples per subject")
    print(val_df["subject_id"].value_counts().sort_index().to_string())
    print("[data] Train per-subject label diversity")
    print(train_df.groupby("subject_id")["class_label"].nunique().sort_index().to_string())
    print("[data] Val per-subject label diversity")
    print(val_df.groupby("subject_id")["class_label"].nunique().sort_index().to_string())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))

    output_dir = Path(cfg["output_dir"])
    checkpoints_root = ensure_dir(output_dir / "checkpoints")
    model_name = str(cfg["model_name"])
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = ensure_dir(checkpoints_root / f"{model_name}-{run_timestamp}")
    logs_dir = ensure_dir(output_dir / "logs" / f"{model_name}-{run_timestamp}")
    figures_dir = ensure_dir(output_dir / "figures"/ f"{model_name}-{run_timestamp}")

    wandb_run, wandb_run_name = init_wandb(cfg, output_dir)

    train_df = pd.read_csv(cfg["train_split_csv"])
    val_df = pd.read_csv(cfg["val_split_csv"])
    log_split_diagnostics(train_df, val_df)

    tiny_subset_size = cfg.get("tiny_subset_size")
    if tiny_subset_size is not None:
        tiny_subset_size = int(tiny_subset_size)
        tiny_subset_seed = int(cfg.get("tiny_subset_seed", cfg["seed"]))
        train_df = train_df.sample(n=tiny_subset_size, random_state=tiny_subset_seed).reset_index(drop=True)
        print(f"[debug] Using tiny training subset: {tiny_subset_size} samples (seed={tiny_subset_seed})")

    cfg = dict(cfg)
    requires_subject_ids = model_requires_subject_ids(model_name)
    subject_id_to_index = None
    if requires_subject_ids:
        subject_id_to_index = build_subject_id_mapping(train_df["subject_id"].astype(str).tolist())
        cfg["subject_id_to_index"] = subject_id_to_index
        cfg["num_subjects"] = len(subject_id_to_index)

    train_dataset = EEGClassificationDataset(
        train_df,
        normalization=str(cfg.get("normalization", "none")),
    )
    val_dataset = EEGClassificationDataset(
        val_df,
        normalization=str(cfg.get("normalization", "none")),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
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
        num_subjects=int(cfg.get("num_subjects", 13)),
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    best_val_acc = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []

    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            subject_id_to_index=subject_id_to_index,
            requires_subject_ids=requires_subject_ids,
        )
        val_loss, val_acc, _, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            subject_id_to_index=subject_id_to_index,
            requires_subject_ids=requires_subject_ids,
        )

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)

        print(
            f"[train] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(epoch_log)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path=checkpoints_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=cfg,
            )

    # Save last checkpoint too.
    save_checkpoint(
        checkpoint_path=checkpoints_dir / "last_model.pt",
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        best_val_acc=best_val_acc,
        config=cfg,
    )

    history_df = pd.DataFrame(history)
    history_csv = logs_dir / "train_history.csv"
    history_json = logs_dir / "train_history.json"
    history_df.to_csv(history_csv, index=False)
    with history_json.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    plot_paths = save_training_plots(history_df, figures_dir)

    summary = {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "epochs": int(epochs),
        "device": str(device),
        "checkpoint_dir": str(checkpoints_dir),
        "num_train_samples": int(len(train_dataset)),
        "num_val_samples": int(len(val_dataset)),
        "best_checkpoint": str(checkpoints_dir / "best_model.pt"),
        "last_checkpoint": str(checkpoints_dir / "last_model.pt"),
        "history_csv": str(history_csv),
        "history_json": str(history_json),
        "loss_curve": str(plot_paths["loss_curve"]),
        "acc_curve": str(plot_paths["acc_curve"]),
    }
    summary_json = logs_dir / "train_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if wandb_run_name is not None:
        wandb_info_path = logs_dir / "wandb_run.json"
        with wandb_info_path.open("w", encoding="utf-8") as handle:
            json.dump({"run_name": wandb_run_name}, handle, indent=2)

    subject_mapping_path = None
    if subject_id_to_index is not None:
        summary["subject_id_to_index"] = subject_id_to_index
        subject_mapping_path = logs_dir / "subject_id_mapping.json"
        with subject_mapping_path.open("w", encoding="utf-8") as handle:
            json.dump(subject_id_to_index, handle, indent=2, sort_keys=True)

    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            wandb_run.log({
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            })
            wandb_run.log(
                {
                    "loss_curve": wandb.Image(str(plot_paths["loss_curve"])),
                    "acc_curve": wandb.Image(str(plot_paths["acc_curve"])),
                }
            )
        finally:
            wandb_run.finish()

    print(f"[train] Best val_acc={best_val_acc:.4f} at epoch={best_epoch}")
    print(f"[train] Checkpoint dir: {checkpoints_dir}")
    print(f"[train] Saved checkpoint: {checkpoints_dir / 'best_model.pt'}")
    print(f"[train] Saved last checkpoint: {checkpoints_dir / 'last_model.pt'}")
    print(f"[train] Saved history CSV: {history_csv}")
    print(f"[train] Saved history JSON: {history_json}")
    print(f"[train] Saved training summary: {summary_json}")
    if subject_mapping_path is not None:
        print(f"[train] Saved subject mapping: {subject_mapping_path}")
    print(f"[train] Saved curves: {plot_paths['loss_curve']}, {plot_paths['acc_curve']}")
    if wandb_run_name is not None:
        print(f"[train] Wandb run name: {wandb_run_name}")


if __name__ == "__main__":
    main()
