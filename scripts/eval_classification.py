"""
Evaluate classification model.

Usage:
    python scripts/eval_classification.py --config configs/classification_baseline.yaml --checkpoint outputs/checkpoints/best_model.pt
"""
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path

from src.datasets import EEGClassificationDataset
from src.models import BaselineMLP, EEGCNN
from src.losses import CrossEntropyLoss
from src.train_utils import validate, load_checkpoint
from src.metrics import compute_accuracy, compute_confusion_matrix, plot_confusion_matrix


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_classification(config, checkpoint_path, split='test'):
    """Evaluate classification model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # TODO: Load test data
    # metadata = load_metadata(config['data']['metadata_path'])
    # test_df = pd.read_csv(Path(config['data']['split_dir']) / f'{split}.csv')
    # test_dataset = EEGClassificationDataset(metadata, test_df, config['data']['metadata_path'])
    # test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # Create model
    if config['model']['name'] == 'baseline_mlp':
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

    # Load checkpoint
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Evaluate
    criterion = CrossEntropyLoss()
    val_loss, val_acc, predictions, targets = validate(model, None, criterion, device)

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({split} split)")
    print(f"{'='*50}")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_acc:.2f}%")

    # Compute confusion matrix
    cm = compute_confusion_matrix(predictions, targets, config['model']['num_classes'])
    print(f"\nConfusion Matrix:\n{cm}")

    # Save results
    output_dir = Path(config['output']['log_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix plot
    plot_path = output_dir / f'confusion_matrix_{split}.png'
    class_names = [f'Class {i}' for i in range(config['model']['num_classes'])]
    plot_confusion_matrix(cm, class_names, save_path=str(plot_path))
    print(f"\nSaved confusion matrix to {plot_path}")

    # Save predictions
    predictions_path = Path('outputs/predictions') / f'{split}_predictions.pt'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'predictions': predictions,
        'targets': targets,
        'loss': val_loss,
        'accuracy': val_acc
    }, predictions_path)
    print(f"Saved predictions to {predictions_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate classification model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate')

    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_classification(config, args.checkpoint, args.split)


if __name__ == '__main__':
    main()
