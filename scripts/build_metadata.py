"""
Build unified metadata table from raw EEG, labels, and captions.

Usage:
    python scripts/build_metadata.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from src.data_utils import load_metadata, ensure_dir


def build_metadata_from_raw(raw_data_dir, output_dir):
    """
    Build unified metadata table from raw data.

    Args:
        raw_data_dir: Path to raw data directory
        output_dir: Path to output directory for processed data

    Returns:
        pd.DataFrame: Metadata table with columns:
            - subject_id
            - session_id
            - trial_id
            - eeg_path
            - class_label
            - caption
            - image_id
    """
    raw_dir = Path(raw_data_dir)
    ensure_dir(output_dir)

    # TODO: Implement metadata building logic
    # This is a placeholder - adapt to your actual data structure

    # Example structure:
    metadata = []

    # Iterate through raw data files
    for subject_dir in sorted(raw_dir.glob('subject_*')):
        subject_id = subject_dir.name

        for session_dir in sorted(subject_dir.glob('session_*')):
            session_id = session_dir.name

            for eeg_file in sorted(session_dir.glob('*.npy')):
                trial_id = eeg_file.stem

                # TODO: Load corresponding labels and captions
                # Adjust this based on your actual data structure
                metadata.append({
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'trial_id': trial_id,
                    'eeg_path': str(eeg_file.absolute()),
                    'class_label': None,  # Load from label file
                    'caption': None,  # Load from caption file
                    'image_id': None  # Load from mapping file
                })

    # Create DataFrame
    metadata_df = pd.DataFrame(metadata)

    # Save to CSV
    output_path = Path(output_dir) / 'metadata.csv'
    metadata_df.to_csv(output_path, index=False)

    print(f"Metadata saved to {output_path}")
    print(f"Total samples: {len(metadata_df)}")

    return metadata_df


def main():
    parser = argparse.ArgumentParser(description='Build metadata from raw data')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Path to output directory')

    args = parser.parse_args()

    build_metadata_from_raw(args.raw_dir, args.output_dir)


if __name__ == '__main__':
    main()
