"""
Downloader for DynamicSuperb/AccentClassification_AccentdbExtended (Accent DB Extended)

This dataset only exposes a 'test' split on Hugging Face. We:
  1. Load the test split with audio decoding disabled (avoids torchcodec / FFmpeg issues).
  2. Copy cached WAV files into a local folder structure grouped by label.
  3. Perform an internal stratified re-split into train / val / test for modeling.
  4. Emit a single metadata CSV with columns expected by existing pipeline:
       filepath, filename, speaker_id, native_language, age_group, utterance_type,
       text, sampling_rate, split, duration

Notes:
- Labels (original accent classes): american, australian, bangla, british, indian,
  malayalam, odiya, telugu, welsh.
- We treat 'label' directly as native_language; you can later map/merge as desired.
- Age group not provided; we default to 'adult'.
- Speaker id not provided; we synthesize: <label>_spk_<idx>.
- Duration is approximated by reading WAV header using soundfile where possible;
  if that fails we set duration to 0.

Quick start:
    python src/data/download_accentdb_extended.py --output_dir data/raw/accentdb --max_samples 120

After download you can run feature extraction referencing the generated metadata CSV.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import random
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Audio
import soundfile as sf


def load_raw_dataset(cache_dir: str | None = None):
    """Load dataset with audio decoding disabled."""
    ds = load_dataset("DynamicSuperb/AccentClassification_AccentdbExtended", split="test", cache_dir=cache_dir)
    # Disable decoding to bypass torchcodec/FFmpeg dependency
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def stratified_split(indices, labels, train_ratio, val_ratio, seed=42):
    """Stratified split of indices by labels into train/val/test.

    Returns dict: {"train": [...], "val": [...], "test": [...]} of indices.
    """
    random.seed(seed)
    by_label = {}
    for idx, lab in zip(indices, labels):
        by_label.setdefault(lab, []).append(idx)

    train, val, test = [], [], []
    for lab, idxs in by_label.items():
        random.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, n - n_train - n_val)
        # Adjust if rounding caused overflow
        if n_train + n_val + n_test > n:
            overflow = n_train + n_val + n_test - n
            # reduce test size first
            n_test = max(1, n_test - overflow)
        split_train = idxs[:n_train]
        split_val = idxs[n_train:n_train + n_val]
        split_test = idxs[n_train + n_val:n_train + n_val + n_test]
        train.extend(split_train)
        val.extend(split_val)
        test.extend(split_test)
    return {"train": train, "val": val, "test": test}


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"  ⚠ Failed to copy {src} -> {dst}: {e}")
        return False


def estimate_duration(wav_path: Path):
    try:
        info = sf.info(str(wav_path))
        if info.samplerate > 0:
            return info.frames / info.samplerate, info.samplerate
        return 0.0, info.samplerate
    except Exception:
        return 0.0, 16000  # Fallback samplerate


def run(output_dir: str, max_samples: int | None, train_ratio: float, val_ratio: float, seed: int, cache_dir: str | None):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading Accent DB Extended (decode disabled)...")
    ds = load_raw_dataset(cache_dir)
    print(f"✓ Loaded dataset with {len(ds)} samples (test split only)")

    # Basic fields present
    sample0 = ds[0]
    print(f"Sample keys: {list(sample0.keys())}")

    # Limit samples if requested
    all_indices = list(range(len(ds)))
    if max_samples and max_samples < len(all_indices):
        all_indices = all_indices[:max_samples]
        print(f"⚡ Limiting to first {len(all_indices)} samples for quick run")

    labels = [ds[i]["label"] for i in all_indices]

    # Stratified internal split
    splits = stratified_split(all_indices, labels, train_ratio, val_ratio, seed)
    print("Split sizes:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")

    metadata_rows = []

    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_indices)} samples)...")
        for local_idx in tqdm(split_indices, desc=f"{split_name} samples"):
            try:
                record = ds[local_idx]
                label = record["label"]
                # Audio field when decode=False contains 'bytes' and optionally 'path'
                audio_field = record["audio"]
                
                # Strategy: write bytes directly to file if available; else use cached path
                filename = f"{label}_{local_idx:05d}.wav"
                dest_dir = out_root / label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / filename
                
                # Check if audio bytes are available (primary method)
                if isinstance(audio_field, dict) and audio_field.get("bytes"):
                    # Write raw WAV bytes to file
                    with open(dest_path, 'wb') as f:
                        f.write(audio_field["bytes"])
                elif isinstance(audio_field, dict) and audio_field.get("path"):
                    # Fallback: copy from cached path
                    audio_path = Path(audio_field["path"])
                    if audio_path.exists():
                        copied = safe_copy(audio_path, dest_path)
                        if not copied:
                            continue
                    else:
                        print(f"  ⚠ Missing file path for index {local_idx}, skipping")
                        continue
                else:
                    print(f"  ⚠ No audio bytes or path for index {local_idx}, skipping")
                    continue
                
                duration, sr = estimate_duration(dest_path)
                metadata_rows.append({
                    "filepath": str(dest_path.relative_to(out_root)),
                    "filename": filename,
                    "speaker_id": f"{label}_spk_{local_idx}",
                    "native_language": label,
                    "age_group": "adult",
                    "utterance_type": "sentence",
                    "text": "",  # not provided
                    "sampling_rate": sr,
                    "split": split_name,
                    "duration": duration
                })
            except Exception as e:
                print(f"  ⚠ Error processing index {local_idx}: {e}")
                continue

    if not metadata_rows:
        print("✗ No samples processed. Exiting.")
        return

    df = pd.DataFrame(metadata_rows)
    meta_path = out_root / "metadata_accentdb_extended.csv"
    df.to_csv(meta_path, index=False)

    print("\n" + "=" * 80)
    print("Download + organization complete!")
    print("=" * 80)
    print(f"✓ Total samples saved: {len(df)}")
    print(f"✓ Class distribution: {df['native_language'].value_counts().to_dict()}")
    print(f"✓ Split counts: {df['split'].value_counts().to_dict()}")
    print(f"✓ Metadata: {meta_path}")
    print(f"✓ Audio root: {out_root}")
    print("Next: run feature extraction on this metadata CSV.")


def parse_args():
    p = argparse.ArgumentParser(description="Download Accent DB Extended and create stratified splits")
    p.add_argument("--output_dir", type=str, default="data/raw/accentdb_extended", help="Where to place copied audio + metadata")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (quick test)")
    p.add_argument("--train_ratio", type=float, default=0.7, help="Train set ratio")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio; remainder becomes test")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--cache_dir", type=str, default=None, help="Optional HuggingFace cache directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
