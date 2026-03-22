"""Utilities for EEG Task 1 metadata construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_IMAGE_COLUMN_CANDIDATES: Sequence[str] = (
    "FilePath",
    "filepath",
    "file_path",
    "ImagePath",
    "image_path",
    "path",
    "image",
    "image_name",
    "Image",
    "filename",
    "file_name",
    "stimulus",
    "stimulus_name",
)

_INVALID_FALLBACK_CLASS_NAMES = {
    "train",
    "val",
    "test",
    "images",
}


@dataclass(frozen=True)
class RunPair:
    """A paired low-speed EEG run and its image CSV."""

    subject_id: str
    session_id: str
    run_id: str
    eeg_path: Path
    image_csv_path: Path


class MetadataBuildError(RuntimeError):
    """Raised when metadata construction fails due to invalid data assumptions."""


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _clean_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_low_speed_file(path: Path) -> bool:
    token = _clean_token(path.as_posix())
    return "lowspeed" in token and "rsvp" not in token


def _extract_first_id(patterns: Sequence[str], text: str) -> Optional[int]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_subject_session_run(path: Path) -> Tuple[str, str, str]:
    """Parse subject/session/run IDs from a file path."""
    joined = path.as_posix()

    subject = _extract_first_id(
        (r"(?:sub(?:ject)?)\D*(\d{1,3})",),
        joined,
    )
    session = _extract_first_id(
        (r"(?:ses(?:sion)?)\D*(\d{1,3})",),
        joined,
    )
    run = _extract_first_id((r"run\D*(\d{1,3})",), joined)

    if subject is None or session is None or run is None:
        raise MetadataBuildError(
            f"Could not parse subject/session/run from path: {path}"
        )

    return f"sub-{subject:02d}", f"ses-{session:02d}", f"run-{run:02d}"


def find_low_speed_runs(dataset_root: Path) -> List[Path]:
    """Find low-speed EEG run .npy files recursively."""
    candidates = [p for p in dataset_root.rglob("*.npy") if _is_low_speed_file(p)]
    eeg_runs = [p for p in candidates if "1000hz" in p.name.lower()]
    if not eeg_runs:
        eeg_runs = candidates
    if not eeg_runs:
        raise MetadataBuildError(
            f"No low-speed EEG .npy files found under dataset root: {dataset_root}"
        )
    return sorted(eeg_runs)


def _find_low_speed_image_csvs(dataset_root: Path) -> List[Path]:
    csvs = [p for p in dataset_root.rglob("*.csv") if _is_low_speed_file(p)]
    csvs = [p for p in csvs if "image" in p.name.lower()]
    return sorted(csvs)


def pair_eeg_and_csv(dataset_root: Path) -> Tuple[List[RunPair], List[Path]]:
    """Pair low-speed EEG run files with corresponding image CSV files."""
    eeg_paths = find_low_speed_runs(dataset_root)
    csv_paths = _find_low_speed_image_csvs(dataset_root)

    if not csv_paths:
        raise MetadataBuildError(
            f"No low-speed image CSV files found under dataset root: {dataset_root}"
        )

    csv_by_key: Dict[Tuple[str, str, str], Path] = {}
    for csv_path in csv_paths:
        key = parse_subject_session_run(csv_path)
        if key in csv_by_key:
            raise MetadataBuildError(
                f"Duplicate image CSV for key {key}: {csv_by_key[key]} and {csv_path}"
            )
        csv_by_key[key] = csv_path

    run_pairs: List[RunPair] = []
    missing_pairs: List[Path] = []

    for eeg_path in eeg_paths:
        key = parse_subject_session_run(eeg_path)
        csv_path = csv_by_key.get(key)

        if csv_path is None:
            local_candidates = [
                eeg_path.with_name(eeg_path.name.replace("_1000Hz.npy", "_image.csv")),
                eeg_path.with_name(eeg_path.name.replace(".npy", "_image.csv")),
                eeg_path.with_name(eeg_path.name.replace("_eeg.npy", "_image.csv")),
            ]
            csv_path = next((c for c in local_candidates if c.exists()), None)

        if csv_path is None:
            missing_pairs.append(eeg_path)
            continue

        run_pairs.append(
            RunPair(
                subject_id=key[0],
                session_id=key[1],
                run_id=key[2],
                eeg_path=eeg_path,
                image_csv_path=csv_path,
            )
        )

    if missing_pairs:
        example = "\n".join(str(p) for p in missing_pairs[:5])
        raise MetadataBuildError(
            "Failed to pair EEG runs with image CSV files. "
            f"Missing pairs: {len(missing_pairs)}\n"
            f"Examples:\n{example}"
        )

    matched_csvs = {pair.image_csv_path for pair in run_pairs}
    unmatched_csvs = [csv_path for csv_path in csv_paths if csv_path not in matched_csvs]
    return run_pairs, unmatched_csvs


def _find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    existing = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in existing:
            return existing[name.lower()]
    return None


def _strip_resized_suffix(image_name: str) -> str:
    """Strip a trailing '_resized' suffix from an image stem."""
    if image_name.lower().endswith("_resized"):
        return image_name[: -len("_resized")]
    return image_name


def _get_row_file_path_value(row: pd.Series) -> Optional[str]:
    """Return the first non-empty file path-like value from candidate columns."""
    for col in _IMAGE_COLUMN_CANDIDATES:
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip()
            if not raw:
                continue
            return raw
    return None


def resolve_image_name(file_path_value: str) -> str:
    """Resolve image stem from FilePath using PureWindowsPath semantics."""
    image_name_raw = PureWindowsPath(file_path_value).stem
    if not image_name_raw:
        raise MetadataBuildError(f"Failed to parse image name from FilePath: {file_path_value!r}")
    return image_name_raw


def resolve_image_path(row: pd.Series, csv_path: Path, dataset_root: Path) -> Optional[str]:
    """Resolve an absolute image path if path-like value exists in the row."""
    for col in _IMAGE_COLUMN_CANDIDATES:
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip()
            if not raw:
                continue
            candidate = Path(raw)
            if candidate.is_absolute():
                return str(candidate)
            local = (csv_path.parent / candidate).resolve()
            if local.exists():
                return str(local)
            from_root = (dataset_root / candidate).resolve()
            if from_root.exists():
                return str(from_root)
            return str(local)
    return None


def infer_fallback_class_name(file_path_value: Optional[str]) -> Optional[str]:
    """Infer fallback class name from FilePath parent folder."""
    if not file_path_value:
        return None

    candidate = PureWindowsPath(file_path_value)
    if len(candidate.parts) < 2:
        return None

    parent = candidate.parent.name.strip()
    if not parent:
        return None

    parent_lower = parent.lower()
    if parent_lower in _INVALID_FALLBACK_CLASS_NAMES:
        return None
    if re.fullmatch(r"sub-\d+", parent_lower):
        return None
    if re.fullmatch(r"ses-\d+", parent_lower):
        return None
    if re.fullmatch(r"run-\d+", parent_lower):
        return None

    return parent


def load_captions_if_available(dataset_root: Path) -> pd.DataFrame:
    """Load captions metadata from captions.txt; return empty DataFrame if unavailable."""
    candidates = sorted(dataset_root.rglob("captions.txt"))
    if not candidates:
        return pd.DataFrame(columns=["image_name", "category", "abstracted"])

    caption_file = candidates[0]
    captions_df = pd.read_csv(caption_file, sep="\t", dtype=str)
    if captions_df.empty:
        return pd.DataFrame(columns=["image_name", "category", "abstracted"])

    # Normalize column names and whitespace.
    captions_df.columns = [str(c).strip() for c in captions_df.columns]
    for col in captions_df.columns:
        captions_df[col] = captions_df[col].astype(str).str.strip()

    required = {"image_name", "category", "abstracted"}
    if not required.issubset(set(captions_df.columns)):
        raise MetadataBuildError(
            "captions.txt was found but missing required columns. "
            f"Expected columns {sorted(required)}, got columns: {list(captions_df.columns)}"
        )

    parsed = captions_df[["image_name", "category", "abstracted"]].copy()
    parsed["image_name"] = parsed["image_name"].astype(str).str.strip()
    parsed["category"] = parsed["category"].astype(str).str.strip()
    parsed["abstracted"] = parsed["abstracted"].astype(str).str.strip()
    parsed = parsed[parsed["image_name"].str.len() > 0]
    parsed = parsed.drop_duplicates(subset=["image_name"], keep="first")
    return parsed


def build_trial_metadata(
    run_pair: RunPair,
    dataset_root: Path,
) -> List[Dict[str, object]]:
    """Build row-per-trial metadata for one paired run."""
    eeg = np.load(run_pair.eeg_path, mmap_mode="r")
    if eeg.ndim != 3:
        raise MetadataBuildError(
            f"Expected EEG array with 3 dims [trials, channels, time], got shape {eeg.shape} "
            f"for {run_pair.eeg_path}"
        )

    run_df = pd.read_csv(run_pair.image_csv_path)
    n_trials = int(eeg.shape[0])
    if len(run_df) != n_trials:
        raise MetadataBuildError(
            "EEG/image trial count mismatch for run "
            f"{run_pair.eeg_path}: EEG trials={n_trials}, image rows={len(run_df)} "
            f"(CSV: {run_pair.image_csv_path})"
        )

    rows: List[Dict[str, object]] = []
    for trial_index, csv_row in run_df.iterrows():
        file_path_value = _get_row_file_path_value(csv_row)
        if file_path_value is None:
            raise MetadataBuildError(
                "Missing image identifier in image CSV row. Tried columns: "
                f"{list(_IMAGE_COLUMN_CANDIDATES)}. CSV: {run_pair.image_csv_path}, "
                f"row index: {trial_index}"
            )

        image_name = resolve_image_name(file_path_value)
        image_path = resolve_image_path(csv_row, run_pair.image_csv_path, dataset_root)
        image_name_noresize = _strip_resized_suffix(image_name)
        class_from_path = infer_fallback_class_name(file_path_value)

        rows.append(
            {
                "subject_id": run_pair.subject_id,
                "session_id": run_pair.session_id,
                "run_id": run_pair.run_id,
                "trial_index": int(trial_index),
                "eeg_path": str(run_pair.eeg_path.resolve()),
                "image_csv_path": str(run_pair.image_csv_path.resolve()),
                "FilePath": file_path_value,
                "image_name_raw": image_name,
                "image_name_noresize": image_name_noresize,
                "image_path": image_path,
                "class_from_path": class_from_path,
                "class_name": pd.NA,
                "class_label": -1,
                "caption": pd.NA,
                "split": "",
            }
        )

    return rows


def build_metadata_table(dataset_root: Path) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """Build full trial-level metadata table for low-speed EEG classification."""
    run_pairs, unmatched_csvs = pair_eeg_and_csv(dataset_root)
    captions_df = load_captions_if_available(dataset_root)

    all_rows: List[Dict[str, object]] = []
    for pair in run_pairs:
        all_rows.extend(build_trial_metadata(pair, dataset_root=dataset_root))

    if not all_rows:
        raise MetadataBuildError("No metadata rows were produced from discovered run pairs.")

    metadata_df = pd.DataFrame(all_rows)
    metadata_df["image_name_noresize"] = metadata_df["image_name_noresize"].astype(str).str.strip()
    print(f"[build_metadata] metadata columns: {list(metadata_df.columns)}")

    if not captions_df.empty:
        captions_df["image_name"] = captions_df["image_name"].astype(str).str.strip()
        metadata_df = metadata_df.merge(
            captions_df,
            how="left",
            left_on="image_name_noresize",
            right_on="image_name",
            suffixes=("_meta", "_cap"),
        )
    else:
        metadata_df["category"] = pd.NA
        metadata_df["abstracted"] = pd.NA
        metadata_df["image_name"] = pd.NA

    print(f"[build_metadata] merged columns: {list(metadata_df.columns)}")

    # Resolve potential suffix variants from merge.
    category_col = (
        "category"
        if "category" in metadata_df.columns
        else ("category_cap" if "category_cap" in metadata_df.columns else None)
    )
    if category_col is None and "category_y" in metadata_df.columns:
        category_col = "category_y"

    abstracted_col = (
        "abstracted"
        if "abstracted" in metadata_df.columns
        else ("abstracted_cap" if "abstracted_cap" in metadata_df.columns else None)
    )
    if abstracted_col is None and "abstracted_y" in metadata_df.columns:
        abstracted_col = "abstracted_y"

    image_name_col = (
        "image_name"
        if "image_name" in metadata_df.columns
        else ("image_name_cap" if "image_name_cap" in metadata_df.columns else None)
    )
    if image_name_col is None and "image_name_y" in metadata_df.columns:
        image_name_col = "image_name_y"

    if category_col is None:
        raise MetadataBuildError(
            "After merging captions, no category column found "
            "(checked: category, category_cap, category_y)."
        )
    if abstracted_col is None:
        raise MetadataBuildError(
            "After merging captions, no abstracted column found "
            "(checked: abstracted, abstracted_cap, abstracted_y)."
        )
    if image_name_col is None:
        raise MetadataBuildError(
            "After merging captions, no image_name column found "
            "(checked: image_name, image_name_cap, image_name_y)."
        )

    # Canonical merged columns for downstream use and debug printing.
    metadata_df["category"] = metadata_df[category_col]
    metadata_df["abstracted"] = metadata_df[abstracted_col]
    metadata_df["image_name"] = metadata_df[image_name_col]

    metadata_df["category"] = metadata_df["category"].astype("string").str.strip()
    metadata_df["abstracted"] = metadata_df["abstracted"].astype("string").str.strip()
    metadata_df["image_name"] = metadata_df["image_name"].astype("string").str.strip()

    # Primary: captions.txt category
    metadata_df["class_name"] = metadata_df["category"]
    metadata_df["caption"] = metadata_df["abstracted"]
    # Fallback: FilePath parent folder class (with blacklist + pattern rejection)
    fallback_mask = metadata_df["class_name"].isna() | (metadata_df["class_name"].astype(str).str.len() == 0)
    metadata_df.loc[fallback_mask, "class_name"] = metadata_df.loc[fallback_mask, "class_from_path"]

    print(f"[build_metadata] rows with non-null category: {int(metadata_df['category'].notna().sum())}")
    print(f"[build_metadata] rows with non-null class_name: {int(metadata_df['class_name'].notna().sum())}")
    print(
        "[build_metadata] unique class_name (first 30): "
        f"{metadata_df['class_name'].dropna().astype(str).sort_values().unique()[:30].tolist()}"
    )
    sample_cols = ["FilePath", "image_name_noresize", "image_name", "category", "class_from_path", "class_name"]
    sample_cols = [c for c in sample_cols if c in metadata_df.columns]
    print("[build_metadata] sample merged rows (first 10):")
    print(metadata_df[sample_cols].head(10).to_string(index=False))

    missing_class = int(metadata_df["class_name"].isna().sum())
    if missing_class > 0:
        raise MetadataBuildError(f"Missing class assignments in {missing_class} metadata rows.")

    unique_classes = sorted(metadata_df["class_name"].astype(str).unique().tolist())
    if len(unique_classes) != 20:
        preview = ", ".join(unique_classes[:30])
        raise MetadataBuildError(
            f"Expected exactly 20 classes, found {len(unique_classes)} classes. "
            f"Classes: {preview}"
        )

    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    metadata_df["class_label"] = metadata_df["class_name"].map(class_to_idx).astype(int)

    summary = {
        "num_subjects": int(metadata_df["subject_id"].nunique()),
        "num_sessions": int(metadata_df[["subject_id", "session_id"]].drop_duplicates().shape[0]),
        "num_runs": int(metadata_df[["subject_id", "session_id", "run_id", "eeg_path"]].drop_duplicates().shape[0]),
        "total_trials": int(len(metadata_df)),
        "num_classes": int(len(unique_classes)),
        "missing_captions": int(metadata_df["caption"].isna().sum()),
        "missing_class_assignments": int(missing_class),
        "unmatched_image_csv_files": int(len(unmatched_csvs)),
    }
    metadata_df = metadata_df.drop(columns=["class_from_path"], errors="ignore")
    return metadata_df, class_to_idx, summary


def save_class_mapping(mapping: Dict[str, int], output_path: Path) -> None:
    """Save class-name to class-label mapping as JSON."""
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)
