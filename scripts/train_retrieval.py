"""Train Task 2B EEG-to-caption retrieval."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_utils import ensure_dir
from src.datasets import EEGRetrievalDataset, build_normalization_state
from src.losses import ContrastiveLoss
from src.models import EEGTextRetrievalModel
from src.retrieval_utils import count_trainable_parameters
from src.train_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG-to-caption retrieval model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to retrieval config YAML.")
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
    if not (0 <= time_window_start < end <= total_timesteps):
        raise ValueError(
            "Invalid time window configuration. "
            f"Expected 0 <= start < end <= {total_timesteps}, got start={time_window_start}, end={end}."
        )
    return end - time_window_start


def init_wandb(cfg: Dict[str, object], output_dir: Path) -> Tuple[Optional[object], Optional[str]]:
    wandb_cfg = cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None, None

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("wandb is enabled but not installed. Install with `pip install wandb`.") from exc

    model_name = str(cfg.get("model_name", "retrieval_model"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(wandb_cfg.get("run_name") or f"{model_name}-{timestamp}")
    run = wandb.init(
        entity=str(wandb_cfg.get("entity", "11685-Proj")),
        project=str(wandb_cfg.get("project", "eeg-task2b")),
        name=run_name,
        mode=str(wandb_cfg.get("mode", "online")),
        dir=str(output_dir),
        config={
            "model_name": cfg.get("model_name"),
            "eeg_model_name": cfg.get("eeg_model_name"),
            "clip_train_mode": cfg.get("clip_train_mode"),
            "batch_size": cfg.get("batch_size"),
            "lr": cfg.get("lr"),
            "epochs": cfg.get("epochs"),
            "normalization": cfg.get("normalization"),
            "time_window_start": cfg.get("time_window_start"),
            "time_window_end": cfg.get("time_window_end"),
            "seed": cfg.get("seed"),
        },
    )
    return run, run_name


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
        "time_window_start": int(cfg.get("time_window_start", 0)),
        "time_window_end": cfg.get("time_window_end"),
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


def train_epoch(
    model: EEGTextRetrievalModel,
    dataloader: DataLoader,
    criterion: ContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for eeg, captions, _, _ in dataloader:
        eeg = eeg.to(device)
        optimizer.zero_grad(set_to_none=True)
        eeg_embeddings, text_embeddings = model(eeg, list(captions), device=device)
        loss = criterion(eeg_embeddings, text_embeddings)
        loss.backward()
        optimizer.step()
        batch_size = eeg.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: EEGTextRetrievalModel,
    dataloader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for eeg, captions, _, _ in dataloader:
        eeg = eeg.to(device)
        eeg_embeddings, text_embeddings = model(eeg, list(captions), device=device)
        loss = criterion(eeg_embeddings, text_embeddings)
        batch_size = eeg.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "cuda")))

    output_dir = Path(cfg.get("output_dir", "outputs"))
    model_name = str(cfg.get("model_name", "task2b_retrieval"))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints" / f"{model_name}-{run_timestamp}")
    logs_dir = ensure_dir(output_dir / "logs" / f"{model_name}-{run_timestamp}")

    wandb_run, wandb_run_name = init_wandb(cfg, output_dir)

    train_df = pd.read_csv(cfg["train_split_csv"])
    val_df = pd.read_csv(cfg["val_split_csv"])
    normalization = str(cfg.get("normalization", "none"))
    normalization_state = build_normalization_state(
        train_df,
        normalization=normalization,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )

    train_dataset = EEGRetrievalDataset(
        train_df,
        normalization=normalization,
        normalization_state=normalization_state,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )
    val_dataset = EEGRetrievalDataset(
        val_df,
        normalization=normalization,
        normalization_state=normalization_state,
        time_window_start=int(cfg.get("time_window_start", 0)),
        time_window_end=cfg.get("time_window_end"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    model = build_retrieval_model(cfg).to(device)
    checkpoint_stats = model.load_eeg_checkpoint(cfg.get("task1_checkpoint"))
    param_stats = count_trainable_parameters(model)
    print(f"[retrieval] Loaded Task 1 checkpoint keys: {checkpoint_stats}")
    print(f"[retrieval] Trainable parameters: {param_stats}")

    criterion = ContrastiveLoss(temperature=float(cfg.get("temperature", 0.07)))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    scheduler_name = str(cfg.get("scheduler", "cosine")).strip().lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(cfg.get("epochs", 10)), 1),
            eta_min=float(cfg.get("scheduler_eta_min", 1e-6)),
        )

    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(1, int(cfg.get("epochs", 10)) + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)
        print(
            f"[retrieval] epoch={epoch:03d}/{int(cfg.get('epochs', 10))} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log({**epoch_log, **param_stats})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "task1_checkpoint_stats": checkpoint_stats,
                    "parameter_stats": param_stats,
                },
                checkpoint_dir / "best_model.pt",
            )

    torch.save(
        {
            "epoch": int(cfg.get("epochs", 10)),
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "task1_checkpoint_stats": checkpoint_stats,
            "parameter_stats": param_stats,
        },
        checkpoint_dir / "last_model.pt",
    )

    summary = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "checkpoint_dir": str(checkpoint_dir),
        "wandb_run_name": wandb_run_name,
        **checkpoint_stats,
        **param_stats,
        "history": history,
    }
    with (logs_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[retrieval] Saved best checkpoint: {checkpoint_dir / 'best_model.pt'}")
    print(f"[retrieval] Saved summary: {logs_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
