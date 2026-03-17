"""Data utility functions for EEG preprocessing and loading"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_metadata(metadata_path):
    """Load metadata CSV file"""
    return pd.read_csv(metadata_path)


def normalize_eeg(eeg_signal, method='zscore'):
    """Normalize EEG signals"""
    if method == 'zscore':
        return (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8)
    elif method == 'minmax':
        return (eeg_signal - np.min(eeg_signal)) / (np.max(eeg_signal) - np.min(eeg_signal) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_eeg_file(eeg_path):
    """Load EEG numpy file"""
    return np.load(eeg_path)


def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)
