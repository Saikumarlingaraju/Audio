"""
Download a small multilingual subset from Mozilla Common Voice (Hindi, Tamil, Telugu, Malayalam, Kannada)
Creates a unified metadata.csv and stores audio clips.
Requires: datasets, soundfile
"""
import argparse
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from itertools import islice

LANG_MAP = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
}


def download_commonvoice_subset(output_dir: str, split: str = "train", samples_per_lang: int = 100):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root = out_dir / "commonvoice"
    audio_root.mkdir(exist_ok=True)

    records = []

    versions = [
        "mozilla-foundation/common_voice_17_0",
        "mozilla-foundation/common_voice_13_0",
        # Older ones below may not exist anymore; kept for completeness
        "mozilla-foundation/common_voice_11_0",
        "mozilla-foundation/common_voice_7_0",
    ]

    for lang_code, lang_name in LANG_MAP.items():
        print(f"\n=== Language: {lang_name} ({lang_code}) ===")
        ds = None
        last_err = None
        for ver in versions:
            try:
                # First try regular (non-streaming) load
                ds = load_dataset(ver, lang_code, split=split, trust_remote_code=True)
                print(f"✓ Loaded {ver}")
                break
            except Exception as e:
                last_err = e
                print(f"⚠ {ver} not available (regular load): {e}")
                # Try streaming mode as a fallback (doesn't require local data files in repo)
                try:
                    ds = load_dataset(ver, lang_code, split=split, streaming=True, trust_remote_code=True)
                    print(f"✓ Loaded {ver} in streaming mode")
                    break
                except Exception as e2:
                    last_err = e2
                    print(f"⚠ {ver} not available (streaming): {e2}")
        if ds is None:
            print(f"❌ Failed to load Common Voice for {lang_name}: {last_err}")
            continue

        # Determine iteration strategy based on streaming or not
        is_streaming = getattr(ds, "_format_columns", None) is None and not hasattr(ds, "__len__")
        if is_streaming:
            n = samples_per_lang
            print(f"Selecting {n} samples (streaming)")
            iterator = islice(ds, n)
        else:
            n = min(samples_per_lang, len(ds))
            print(f"Selecting {n} samples out of {len(ds)}")
            iterator = ds.select(range(n))

        lang_dir = audio_root / lang_name
        lang_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(tqdm(iterator, total=n, desc=f"Processing {lang_name}")):
            audio = sample.get("audio")
            if audio is None:
                continue
            array = audio["array"]
            sr = audio["sampling_rate"]
            filename = f"{split}_{lang_code}_{i:05d}.wav"
            filepath = lang_dir / filename
            try:
                sf.write(str(filepath), array, sr)
            except Exception as e:
                print(f"  ⚠ Write failed for {filepath}: {e}")
                continue
            records.append({
                "filepath": str(filepath.relative_to(out_dir)),
                "filename": filename,
                "split": split,
                "native_language": lang_name,
                "speaker_id": sample.get("client_id", "unknown"),
                "text": sample.get("sentence", ""),
                "age_group": sample.get("age", "adult"),
                "gender": sample.get("gender", "unknown"),
                "accents": sample.get("accents", ""),
                "duration": len(array) / sr if array is not None else 0,
                "sampling_rate": sr
            })

    if not records:
        print("❌ No samples downloaded.")
        return

    df = pd.DataFrame(records)
    meta_path = out_dir / "metadata_commonvoice.csv"
    df.to_csv(meta_path, index=False)
    print("\n================ SUMMARY ================")
    print(f"Total samples: {len(df)}")
    print("Per language counts:")
    print(df["native_language"].value_counts())
    print(f"Metadata saved to: {meta_path}")
    print(f"Audio root: {audio_root}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download Common Voice subset for accent modeling")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "validation"], help="Dataset split")
    parser.add_argument("--samples_per_lang", type=int, default=100, help="Samples per language")
    args = parser.parse_args()
    download_commonvoice_subset(args.output_dir, args.split, args.samples_per_lang)


if __name__ == "__main__":
    main()
