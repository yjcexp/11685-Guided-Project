"""
Generate train/val/test splits from metadata.

Usage:
    python scripts/make_splits.py --metadata_path data/processed/metadata.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

from src.data_utils import ensure_dir


def make_splits(metadata_path, split_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                split_by='subject', random_seed=42):
    """
    Generate train/val/test splits from metadata.

    Args:
        metadata_path: Path to metadata CSV file
        split_dir: Path to directory to save split files
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        split_by: How to split data ('subject' for subject-wise, 'random' for random)
        random_seed: Random seed for reproducibility

    Returns:
        dict: Dictionary containing train, val, test DataFrames
    """
    np.random.seed(random_seed)

    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(metadata)} samples")

    ensure_dir(split_dir)

    if split_by == 'subject':
        # Subject-wise split
        subjects = metadata['subject_id'].unique()
        train_subjects, temp_subjects = train_test_split(
            subjects, train_size=train_ratio, random_state=random_seed
        )
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=random_seed
        )

        train_split = metadata[metadata['subject_id'].isin(train_subjects)]
        val_split = metadata[metadata['subject_id'].isin(val_subjects)]
        test_split = metadata[metadata['subject_id'].isin(test_subjects)]

    else:  # Random split
        train_split, temp_split = train_test_split(
            metadata, train_size=train_ratio, random_state=random_seed
        )
        val_split, test_split = train_test_split(
            temp_split,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=random_seed
        )

    # Save splits
    split_dir = Path(split_dir)
    train_split.to_csv(split_dir / 'train.csv', index=False)
    val_split.to_csv(split_dir / 'val.csv', index=False)
    test_split.to_csv(split_dir / 'test.csv', index=False)

    print(f"Train split: {len(train_split)} samples")
    print(f"Val split: {len(val_split)} samples")
    print(f"Test split: {len(test_split)} samples")

    return {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }


def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test splits')
    parser.add_argument('--metadata_path', type=str, default='data/processed/metadata.csv',
                        help='Path to metadata CSV file')
    parser.add_argument('--split_dir', type=str, default='data/splits',
                        help='Path to split directory')
    parser.add_argument('--split_by', type=str, default='subject',
                        choices=['subject', 'random'],
                        help='How to split data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    make_splits(args.metadata_path, args.split_dir, split_by=args.split_by, random_seed=args.seed)


if __name__ == '__main__':
    main()
