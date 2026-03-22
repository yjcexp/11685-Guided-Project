"""Build trial-level metadata for low-speed EEG classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_utils import (
    MetadataBuildError,
    build_metadata_table,
    ensure_dir,
    save_class_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build low-speed EEG metadata CSV.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/ocean/projects/cis250019p/gandotra/11785-gp-eeg"),
        help="Root directory of the PSC EEG dataset.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/processed/metadata.csv"),
        help="Output metadata CSV path.",
    )
    parser.add_argument(
        "--class_mapping_json",
        type=Path,
        default=Path("data/processed/class_mapping.json"),
        help="Output class-name to class-label mapping JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        metadata_df, class_mapping, summary = build_metadata_table(args.dataset_root)
    except MetadataBuildError as exc:
        raise SystemExit(f"[build_metadata] ERROR: {exc}") from exc

    ensure_dir(args.output_csv.parent)
    metadata_df.to_csv(args.output_csv, index=False)
    save_class_mapping(class_mapping, args.class_mapping_json)

    print(f"[build_metadata] Wrote metadata: {args.output_csv}")
    print(f"[build_metadata] Wrote class mapping: {args.class_mapping_json}")
    print("[build_metadata] Summary")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    print("[build_metadata] Class mapping")
    for class_name, class_id in sorted(class_mapping.items(), key=lambda kv: kv[1]):
        print(f"  - {class_id:02d}: {class_name}")


if __name__ == "__main__":
    main()
