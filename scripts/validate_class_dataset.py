"""
Validate class student dataset integrity before feature extraction and training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


class CheckState:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def assert_true(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"[PASS] {label}")
        else:
            self.failed += 1
            print(f"[FAIL] {label}")
            if detail:
                print(f"       {detail}")


def _normalize(path_value: str) -> str:
    return str(path_value or "").strip().replace("\\", "/").lstrip("./")


def validate_metadata(df: pd.DataFrame, checks: CheckState) -> None:
    required = [
        "filepath",
        "speaker_id",
        "native_language",
        "age_group",
        "utterance_type",
        "size_bytes",
    ]
    missing = [col for col in required if col not in df.columns]
    checks.assert_true("Required metadata columns", len(missing) == 0, f"Missing: {missing}")

    if missing:
        return

    checks.assert_true("Non-empty speaker IDs", df["speaker_id"].astype(str).str.strip().ne("").all())
    checks.assert_true("Non-empty native_language", df["native_language"].astype(str).str.strip().ne("").all())
    checks.assert_true("No duplicate filepath rows", not df["filepath"].duplicated().any())


def validate_audio_files(df: pd.DataFrame, audio_root: Path, checks: CheckState) -> None:
    missing = []
    zero_or_negative = 0

    for _, row in df.iterrows():
        rel = _normalize(row.get("filepath", ""))
        path = audio_root / rel

        if not rel or not path.exists():
            missing.append(rel)
            continue

        if path.stat().st_size <= 0:
            zero_or_negative += 1

    checks.assert_true("Referenced audio files exist", len(missing) == 0, f"Missing count: {len(missing)}")
    checks.assert_true("No zero-byte audio files", zero_or_negative == 0, f"Zero-byte files: {zero_or_negative}")


def validate_splits(split_dir: Path, checks: CheckState) -> None:
    train_path = split_dir / "train.csv"
    dev_path = split_dir / "dev.csv"
    test_path = split_dir / "test.csv"

    available = all(p.exists() for p in [train_path, dev_path, test_path])
    checks.assert_true("Split files available", available, f"Expected: {train_path}, {dev_path}, {test_path}")

    if not available:
        return

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    train_speakers = set(train_df["speaker_id"].astype(str))
    dev_speakers = set(dev_df["speaker_id"].astype(str))
    test_speakers = set(test_df["speaker_id"].astype(str))

    checks.assert_true("No train/dev speaker overlap", len(train_speakers & dev_speakers) == 0)
    checks.assert_true("No train/test speaker overlap", len(train_speakers & test_speakers) == 0)
    checks.assert_true("No dev/test speaker overlap", len(dev_speakers & test_speakers) == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate class student dataset")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/collections/class_students/organized_by_state_clean/metadata_training_ready.csv",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="data/collections/class_students/organized_by_state_clean",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="data/splits/class_students",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    audio_root = Path(args.audio_root)
    split_dir = Path(args.split_dir)

    checks = CheckState()

    checks.assert_true("Metadata file exists", metadata_path.exists(), str(metadata_path))
    checks.assert_true("Audio root exists", audio_root.exists(), str(audio_root))

    if not metadata_path.exists() or not audio_root.exists():
        print(f"\nSummary: passed={checks.passed}, failed={checks.failed}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)

    validate_metadata(df, checks)
    validate_audio_files(df, audio_root, checks)
    validate_splits(split_dir, checks)

    print("\n" + "=" * 80)
    print(f"Validation summary: passed={checks.passed}, failed={checks.failed}")
    print("=" * 80)

    if checks.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
