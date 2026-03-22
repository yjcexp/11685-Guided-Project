"""PyTorch datasets for EEG Task 1 classification."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_REQUIRED_COLUMNS = {
    "subject_id",
    "session_id",
    "run_id",
    "trial_index",
    "eeg_path",
    "class_label",
}


class EEGClassificationDataset(Dataset):
    """Trial-level EEG dataset for 20-way classification."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        normalization: str = "none",
        cache_size: int = 32,
    ) -> None:
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Split dataframe missing required columns: {sorted(missing)}")

        self.df = df.reset_index(drop=True)
        self.normalization = normalization
        self.cache_size = max(1, int(cache_size))
        self._run_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.df)

    def _load_run(self, eeg_path: str) -> np.ndarray:
        if eeg_path in self._run_cache:
            self._run_cache.move_to_end(eeg_path)
            return self._run_cache[eeg_path]

        array = np.load(eeg_path, mmap_mode="r")
        if array.ndim != 3:
            raise ValueError(
                f"Expected EEG run array with 3 dims [N, C, T] or [N, T, C], got {array.shape} for {eeg_path}"
            )
        if array.shape[1:] not in {(122, 500), (500, 122)}:
            raise ValueError(
                "Unsupported EEG run shape. Expected [N, 122, 500] or [N, 500, 122], "
                f"got {array.shape} for {eeg_path}"
            )

        self._run_cache[eeg_path] = array
        if len(self._run_cache) > self.cache_size:
            self._run_cache.popitem(last=False)
        return array

    @staticmethod
    def _normalize(eeg: np.ndarray, mode: str) -> np.ndarray:
        eps = 1e-6
        if mode == "none":
            return eeg
        if mode == "zscore_per_trial":
            mean = float(eeg.mean())
            std = float(eeg.std())
            return (eeg - mean) / (std + eps)
        if mode == "zscore_per_channel":
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True)
            return (eeg - mean) / (std + eps)
        raise ValueError(f"Unsupported normalization mode: {mode}")

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, int, str, Dict[str, object]]:
        row = self.df.iloc[index]
        eeg_path = str(row["eeg_path"])
        trial_index = int(row["trial_index"])

        run_array = self._load_run(eeg_path)
        if not (0 <= trial_index < run_array.shape[0]):
            raise IndexError(
                f"trial_index={trial_index} out of bounds for run {eeg_path} with {run_array.shape[0]} trials"
            )

        eeg = np.asarray(run_array[trial_index], dtype=np.float32)
        original_shape = tuple(eeg.shape)
        if original_shape == (500, 122):
            eeg = eeg.T
        elif original_shape != (122, 500):
            raise ValueError(
                "Expected per-trial EEG shape to be (122, 500) or (500, 122), "
                f"got {original_shape} for {eeg_path} trial {trial_index}"
            )

        eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        eeg = self._normalize(eeg, self.normalization)
        label = int(row["class_label"])

        metadata: Dict[str, object] = {
            "row_index": int(index),
            "subject_id": str(row["subject_id"]),
            "session_id": str(row["session_id"]),
            "run_id": str(row["run_id"]),
            "trial_index": trial_index,
            "image_name": row.get("image_name", None),
        }

        return torch.from_numpy(eeg), label, str(row["subject_id"]), metadata

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.df
