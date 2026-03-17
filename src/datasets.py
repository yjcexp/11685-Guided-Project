"""PyTorch Dataset classes for EEG data"""
import torch
from torch.utils.data import Dataset
import numpy as np


class EEGClassificationDataset(Dataset):
    """Dataset for EEG classification task"""

    def __init__(self, metadata, split_df, eeg_base_path):
        self.metadata = metadata
        self.split_df = split_df
        self.eeg_base_path = eeg_base_path

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        eeg_path = row['eeg_path']
        label = row['class_label']

        # Load EEG data
        eeg = np.load(eeg_path)

        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg)
        label_tensor = torch.LongTensor([label]).squeeze()

        return eeg_tensor, label_tensor


class EEGRetrievalDataset(Dataset):
    """Dataset for EEG retrieval task"""

    def __init__(self, metadata, split_df, eeg_base_path):
        self.metadata = metadata
        self.split_df = split_df
        self.eeg_base_path = eeg_base_path

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        eeg_path = row['eeg_path']
        caption = row['caption']

        # Load EEG data
        eeg = np.load(eeg_path)

        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg)

        return eeg_tensor, caption
