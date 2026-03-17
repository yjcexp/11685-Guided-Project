"""
Evaluate retrieval model.

Usage:
    python scripts/eval_retrieval.py --config configs/retrieval_baseline.yaml --checkpoint outputs/checkpoints/best_model.pt
"""
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path

from src.datasets import EEGRetrievalDataset
from src.models import RetrievalEncoder
from src.losses import ContrastiveLoss
from src.train_utils import validate
from src.retrieval_utils import extract_embeddings, compute_recall_at_k, batch_retrieval
from src.metrics import compute_accuracy, compute_confusion_matrix, plot_confusion_matrix


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_retrieval(config, checkpoint_path, split='test'):
    """Evaluate retrieval model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # TODO: Load test data
    # metadata = load_metadata(config['data']['metadata_path'])
    # test_df = pd.read_csv(Path(config['data']['split_dir']) / f'{split}.csv')
    # test_dataset = EEGRetrievalDataset(metadata, test_df, config['data']['metadata_path'])
    # test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # Create model
    model = RetrievalEncoder(
        encoder_type=config['model']['encoder_type'],
        input_dim=1000,  # TODO: Set from data
        embedding_dim=config['model']['embedding_dim'],
        projection_dim=config['model']['projection_dim']
    )

    model = model.to(device)

    # Load checkpoint
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Extract embeddings
    print("Extracting embeddings...")
    test_embeddings = extract_embeddings(model, None, device)
    print(f"Extracted {test_embeddings.size(0)} embeddings")

    # Compute recall@k
    print("\nComputing Recall@K...")
    recall_results = compute_recall_at_k(
        test_embeddings,
        test_embeddings,
        k_values=[1, 5, 10]
    )

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({split} split)")
    print(f"{'='*50}")
    for metric, value in recall_results.items():
        print(f"{metric}: {value:.4f}")

    # Save results
    output_dir = Path(config['output']['log_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_path = Path('outputs/predictions') / f'{split}_embeddings.pt'
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_embeddings, embeddings_path)
    print(f"\nSaved embeddings to {embeddings_path}")

    # Save retrieval results
    results_path = Path('outputs/predictions') / f'{split}_retrieval_results.txt'
    with open(results_path, 'w') as f:
        for metric, value in recall_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Saved retrieval results to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate')

    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_retrieval(config, args.checkpoint, args.split)


if __name__ == '__main__':
    main()
