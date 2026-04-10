"""
Prepare class student metadata for training by normalizing labels and filepaths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LANG_FIXES = {
    "telengana": "telangana",
    "telangana_telgu": "telangana_telugu",
    "andhrapradesh": "andhra_pradesh",
}


def slugify(value: str) -> str:
    clean = str(value or "").strip().lower().replace(" ", "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def normalize_language(native_language: str, native_language_other: str) -> str:
    lang = slugify(native_language)
    other = slugify(native_language_other)

    if lang == "other" and other:
        lang = other

    return LANG_FIXES.get(lang, lang or "unknown")


def normalize_relative_path(raw_path: str, audio_root: Path) -> str:
    path_value = str(raw_path or "").replace("\\", "/").strip().lstrip("./")

    candidates = []
    if path_value:
        candidates.append(path_value)
        if path_value.startswith("organized_by_state/"):
            candidates.append(path_value[len("organized_by_state/"):])

    for relative in candidates:
        if (audio_root / relative).exists():
            return relative

    return candidates[-1] if candidates else ""


def prepare_metadata(
    metadata_path: Path,
    audio_root: Path,
    output_path: Path,
    min_bytes: int,
    drop_tiny: bool,
    strict: bool,
) -> int:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    df = pd.read_csv(metadata_path)

    required_cols = [
        "filepath",
        "speaker_id",
        "native_language",
        "native_language_other",
        "size_bytes",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["size_bytes"] = pd.to_numeric(df["size_bytes"], errors="coerce").fillna(0).astype(int)

    if drop_tiny:
        if "is_tiny" in df.columns:
            df = df[df["is_tiny"].fillna(0).astype(int) == 0]
        df = df[df["size_bytes"] >= int(min_bytes)]

    df["speaker_id"] = df["speaker_id"].astype(str).str.strip().str.lower()
    df["native_language"] = df.apply(
        lambda row: normalize_language(row.get("native_language", ""), row.get("native_language_other", "")),
        axis=1,
    )

    df["filepath"] = df["filepath"].apply(lambda value: normalize_relative_path(value, audio_root))

    if "age_group" not in df.columns:
        df["age_group"] = "adult"
    else:
        df["age_group"] = df["age_group"].fillna("adult")

    if "utterance_type" not in df.columns:
        df["utterance_type"] = "read"
    else:
        df["utterance_type"] = df["utterance_type"].fillna("read")

    if "split" not in df.columns:
        df["split"] = "unassigned"
    else:
        df["split"] = "unassigned"

    missing_files = []
    for rel in df["filepath"].tolist():
        if not rel or not (audio_root / rel).exists():
            missing_files.append(rel)

    if strict and missing_files:
        sample = missing_files[:5]
        raise RuntimeError(
            f"Found {len(missing_files)} missing audio files. Sample: {sample}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("=" * 80)
    print("Prepared class metadata for training")
    print("=" * 80)
    print(f"Input rows: {len(pd.read_csv(metadata_path))}")
    print(f"Output rows: {len(df)}")
    print(f"Unique speakers: {df['speaker_id'].nunique()}")
    print(f"Unique languages: {df['native_language'].nunique()}")
    print(f"Audio root: {audio_root}")
    print(f"Output: {output_path}")
    print(f"Missing files: {len(missing_files)}")

    if missing_files:
        missing_path = output_path.with_name("missing_audio_paths.txt")
        missing_path.write_text("\n".join(str(v) for v in missing_files), encoding="utf-8")
        print(f"Missing paths list: {missing_path}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare class student metadata for training")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/collections/class_students/organized_by_state_clean/metadata_all.csv",
        help="Input metadata CSV",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="data/collections/class_students/organized_by_state_clean",
        help="Root directory containing class audio files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/collections/class_students/organized_by_state_clean/metadata_training_ready.csv",
        help="Output metadata CSV",
    )
    parser.add_argument(
        "--min_bytes",
        type=int,
        default=1000,
        help="Minimum recording size to keep",
    )
    parser.add_argument(
        "--keep_tiny",
        action="store_true",
        help="Keep tiny files instead of filtering them",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any referenced audio file is missing",
    )

    args = parser.parse_args()

    prepare_metadata(
        metadata_path=Path(args.metadata),
        audio_root=Path(args.audio_root),
        output_path=Path(args.output),
        min_bytes=int(args.min_bytes),
        drop_tiny=not args.keep_tiny,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
