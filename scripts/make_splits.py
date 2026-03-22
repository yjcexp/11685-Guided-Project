"""Create subject-wise session splits for EEG classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_utils import ensure_dir


REQUIRED_COLUMNS = {"subject_id", "session_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits from metadata CSV.")
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("data/processed/metadata.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory where train/val/test CSV files will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--randomize_sessions",
        action="store_true",
        help="Randomly assign each subject's 5 sessions into 3 train / 1 val / 1 test using --seed.",
    )
    return parser.parse_args()


def _validate_metadata(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {sorted(missing)}")

    session_counts = df.groupby("subject_id")["session_id"].nunique().sort_index()
    bad_subjects = session_counts[session_counts != 5]
    if not bad_subjects.empty:
        details = ", ".join(f"{sid}:{cnt}" for sid, cnt in bad_subjects.items())
        raise ValueError(
            "Each subject must have exactly 5 sessions for session-based split. "
            f"Violations: {details}"
        )


def _assign_subject_sessions(
    sessions: List[str],
    rng: np.random.Generator,
    randomize: bool,
) -> Dict[str, str]:
    ordered = sorted(sessions)
    if randomize:
        ordered = list(rng.permutation(ordered))

    return {
        ordered[0]: "train",
        ordered[1]: "train",
        ordered[2]: "train",
        ordered[3]: "val",
        ordered[4]: "test",
    }


def create_splits(metadata_df: pd.DataFrame, seed: int, randomize_sessions: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = metadata_df.copy()
    df["split"] = ""

    for subject_id, group in df.groupby("subject_id"):
        session_ids = sorted(group["session_id"].astype(str).unique().tolist())
        assignment = _assign_subject_sessions(session_ids, rng=rng, randomize=randomize_sessions)
        mask = df["subject_id"] == subject_id
        df.loc[mask, "split"] = df.loc[mask, "session_id"].map(assignment)

    if (df["split"] == "").any():
        raise ValueError("Found rows without split assignment.")

    return df


def main() -> None:
    args = parse_args()

    metadata_df = pd.read_csv(args.metadata_csv)
    _validate_metadata(metadata_df)

    split_df = create_splits(metadata_df, seed=args.seed, randomize_sessions=args.randomize_sessions)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    # Overlap checks by full row tuple identity index.
    all_indices = set(train_df.index) | set(val_df.index) | set(test_df.index)
    if len(all_indices) != len(split_df):
        raise ValueError("Split overlap or missing rows detected.")

    output_dir = ensure_dir(args.output_dir)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata_with_split_path = args.metadata_csv.parent / "metadata_with_split.csv"
    split_df.to_csv(metadata_with_split_path, index=False)

    print(f"[make_splits] Wrote: {train_path} ({len(train_df)} rows)")
    print(f"[make_splits] Wrote: {val_path} ({len(val_df)} rows)")
    print(f"[make_splits] Wrote: {test_path} ({len(test_df)} rows)")
    print(f"[make_splits] Wrote: {metadata_with_split_path}")

    if "class_name" in split_df.columns:
        class_counts = split_df.groupby(["split", "class_name"]).size().reset_index(name="count")
        print("[make_splits] Class distribution by split (first 20 rows):")
        print(class_counts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
