"""PyTorch datasets for EEG Task 1 classification."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

_EPS = 1e-6


def _resolve_time_window(
    total_timesteps: int,
    time_window_start: int = 0,
    time_window_end: int | None = None,
) -> tuple[int, int]:
    start = int(time_window_start)
    end = total_timesteps if time_window_end is None else int(time_window_end)
    if not (0 <= start < end <= total_timesteps):
        raise ValueError(
            "Invalid time window. "
            f"Expected 0 <= start < end <= {total_timesteps}, got start={start}, end={end}."
        )
    return start, end


def _ensure_trial_shape(eeg: np.ndarray, eeg_path: str, trial_index: int) -> np.ndarray:
    original_shape = tuple(eeg.shape)
    if original_shape == (500, 122):
        eeg = eeg.T
    elif original_shape != (122, 500):
        raise ValueError(
            "Expected per-trial EEG shape to be (122, 500) or (500, 122), "
            f"got {original_shape} for {eeg_path} trial {trial_index}"
        )
    return np.ascontiguousarray(eeg, dtype=np.float32)


def _compute_mean_std(sum_values: np.ndarray, sumsq_values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    mean = sum_values / max(count, 1)
    var = np.maximum((sumsq_values / max(count, 1)) - np.square(mean), _EPS)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def build_normalization_state(
    data: str | Path | pd.DataFrame,
    normalization: str,
    time_window_start: int = 0,
    time_window_end: int | None = None,
) -> Dict[str, Any] | None:
    """Fit normalization statistics on the training split when needed.

    Supported fitted modes:
    - `zscore_per_channel_trainset`
    - `zscore_per_subject_per_channel`

    Other modes return `None` because they normalize per sample at runtime.
    """
    if normalization not in {"zscore_per_channel_trainset", "zscore_per_subject_per_channel"}:
        return None

    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Split dataframe missing required columns: {sorted(missing)}")

    start, end = _resolve_time_window(total_timesteps=500, time_window_start=time_window_start, time_window_end=time_window_end)
    window_len = end - start

    if normalization == "zscore_per_channel_trainset":
        sum_values = np.zeros(122, dtype=np.float64)
        sumsq_values = np.zeros(122, dtype=np.float64)
        trial_count = 0

        for eeg_path, path_df in df.groupby("eeg_path", sort=False):
            run_array = np.load(str(eeg_path), mmap_mode="r")
            for trial_index in path_df["trial_index"].astype(int).tolist():
                eeg = _ensure_trial_shape(np.asarray(run_array[trial_index], dtype=np.float32), str(eeg_path), trial_index)
                eeg = eeg[:, start:end]
                sum_values += eeg.sum(axis=1)
                sumsq_values += np.square(eeg).sum(axis=1)
                trial_count += 1

        if trial_count == 0:
            raise ValueError("Cannot fit normalization state on an empty training split.")

        mean, std = _compute_mean_std(sum_values, sumsq_values, trial_count * window_len)
        return {
            "mode": normalization,
            "mean": mean[:, None],
            "std": std[:, None],
            "time_window_start": start,
            "time_window_end": end,
        }

    subject_sums: Dict[str, np.ndarray] = {}
    subject_sumsq: Dict[str, np.ndarray] = {}
    subject_counts: Dict[str, int] = {}
    for (subject_id, eeg_path), path_df in df.groupby(["subject_id", "eeg_path"], sort=False):
        run_array = np.load(str(eeg_path), mmap_mode="r")
        subject_key = str(subject_id)
        if subject_key not in subject_sums:
            subject_sums[subject_key] = np.zeros(122, dtype=np.float64)
            subject_sumsq[subject_key] = np.zeros(122, dtype=np.float64)
            subject_counts[subject_key] = 0
        for trial_index in path_df["trial_index"].astype(int).tolist():
            eeg = _ensure_trial_shape(np.asarray(run_array[trial_index], dtype=np.float32), str(eeg_path), trial_index)
            eeg = eeg[:, start:end]
            subject_sums[subject_key] += eeg.sum(axis=1)
            subject_sumsq[subject_key] += np.square(eeg).sum(axis=1)
            subject_counts[subject_key] += 1

    if not subject_counts:
        raise ValueError("Cannot fit subject-specific normalization state on an empty training split.")

    stats_by_subject: Dict[str, Dict[str, np.ndarray]] = {}
    for subject_id in sorted(subject_counts):
        mean, std = _compute_mean_std(
            subject_sums[subject_id],
            subject_sumsq[subject_id],
            subject_counts[subject_id] * window_len,
        )
        stats_by_subject[subject_id] = {
            "mean": mean[:, None],
            "std": std[:, None],
        }
    return {
        "mode": normalization,
        "stats_by_subject": stats_by_subject,
        "time_window_start": start,
        "time_window_end": end,
    }


class EEGClassificationDataset(Dataset):
    """Trial-level EEG dataset for 20-way classification."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        normalization: str = "none",
        normalization_state: Dict[str, Any] | None = None,
        cache_size: int = 32,
        time_window_start: int = 0,
        time_window_end: int | None = None,
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
        self.normalization_state = normalization_state
        self.cache_size = max(1, int(cache_size))
        self._run_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.time_window_start, self.time_window_end = _resolve_time_window(
            total_timesteps=500,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
        )

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
    def _normalize(
        eeg: np.ndarray,
        mode: str,
        subject_id: str,
        normalization_state: Dict[str, Any] | None,
    ) -> np.ndarray:
        if mode == "none":
            return eeg
        if mode == "zscore_per_trial":
            mean = float(eeg.mean())
            std = float(eeg.std())
            return (eeg - mean) / (std + _EPS)
        if mode == "zscore_per_channel":
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True)
            return (eeg - mean) / (std + _EPS)
        if mode == "zscore_per_channel_trainset":
            if normalization_state is None:
                raise ValueError("normalization_state is required for zscore_per_channel_trainset.")
            return (eeg - normalization_state["mean"]) / (normalization_state["std"] + _EPS)
        if mode == "zscore_per_subject_per_channel":
            if normalization_state is None:
                raise ValueError("normalization_state is required for zscore_per_subject_per_channel.")
            stats_by_subject = normalization_state["stats_by_subject"]
            if subject_id not in stats_by_subject:
                known_subjects = sorted(stats_by_subject)
                raise ValueError(
                    f"Missing subject-specific normalization stats for {subject_id!r}. "
                    f"Known subjects: {known_subjects}"
                )
            subject_stats = stats_by_subject[subject_id]
            return (eeg - subject_stats["mean"]) / (subject_stats["std"] + _EPS)
        if mode in {"demean_only", "baseline_correction", "demean_per_channel"}:
            mean = eeg.mean(axis=1, keepdims=True)
            return eeg - mean
        raise ValueError(f"Unsupported normalization mode: {mode}")

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, int, str, Dict[str, object]]:
        row = self.df.iloc[index]
        eeg_path = str(row["eeg_path"])
        trial_index = int(row["trial_index"])
        subject_id = str(row["subject_id"])

        run_array = self._load_run(eeg_path)
        if not (0 <= trial_index < run_array.shape[0]):
            raise IndexError(
                f"trial_index={trial_index} out of bounds for run {eeg_path} with {run_array.shape[0]} trials"
            )

        eeg = _ensure_trial_shape(np.asarray(run_array[trial_index], dtype=np.float32), eeg_path, trial_index)
        eeg = eeg[:, self.time_window_start:self.time_window_end]
        eeg = self._normalize(eeg, self.normalization, subject_id=subject_id, normalization_state=self.normalization_state)
        label = int(row["class_label"])

        metadata: Dict[str, object] = {
            "row_index": int(index),
            "subject_id": str(row["subject_id"]),
            "session_id": str(row["session_id"]),
            "run_id": str(row["run_id"]),
            "trial_index": trial_index,
            "image_name": row.get("image_name", None),
        }

        return torch.from_numpy(np.ascontiguousarray(eeg, dtype=np.float32)), label, subject_id, metadata

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.df
