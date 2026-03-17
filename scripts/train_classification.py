"""
Train classification model for EEG signals.

Usage:
    python scripts/train_classification.py --config configs/classification_baseline.yaml
"""
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path

from src.datasets import EEGClassificationDataset
from src.models import BaselineMLP, EEGCNN
from src.losses import CrossEntropyLoss
from src.train_utils import (
    train_one_epoch, validate, save_checkpoint,
    get_optimizer, get_scheduler
)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_classification(config):
    """Train classification model"""
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config['training']['seed'])

    # TODO: Load data
    # metadata = load_metadata(config['data']['metadata_path'])
    # train_df = pd.read_csv(Path(config['data']['split_dir']) / 'train.csv')
    # val_df = pd.read_csv(Path(config['data']['split_dir']) / 'val.csv')

    # TODO: Create datasets and dataloaders
    # train_dataset = EEGClassificationDataset(metadata, train_df, config['data']['metadata_path'])
    # val_dataset = EEGClassificationDataset(metadata, val_df, config['data']['metadata_path'])

    # train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # TODO: Determine model dimensions from data
    # sample_data = train_dataset[0][0]
    # input_dim = sample_data.numel()

    # Create model
    if config['model']['name'] == 'baseline_mlp':
        # TODO: Set actual input_dim and num_classes
        model = BaselineMLP(
            input_dim=1000,  # TODO: Set from data
            hidden_dims=config['model']['hidden_dims'],
            num_classes=10  # TODO: Set from data
        )
    elif config['model']['name'] == 'cnn':
        model = EEGCNN(
            in_channels=1,  # TODO: Set from data
            conv_channels=config['model']['conv_channels'],
            kernel_size=config['model']['kernel_size'],
            pool_size=config['model']['pool_size'],
            num_classes=10  # TODO: Set from data
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")

    model = model.to(device)

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = get_optimizer(
        model,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = get_scheduler(
        optimizer,
        config['training']['num_epochs']
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(model, None, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc, _, _ = validate(model, None, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if epoch % config['output']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                config['output']['checkpoint_dir']
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = Path(config['output']['checkpoint_dir']) / 'best_model.pt'
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')

    args = parser.parse_args()

    config = load_config(args.config)
    train_classification(config)


if __name__ == '__main__':
    main()
