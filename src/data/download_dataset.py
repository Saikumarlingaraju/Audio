"""
Download IndicAccentDb dataset from HuggingFace and organize by adult/child subsets.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf


def download_and_organize_dataset(output_dir, subset="all", cache_dir=None, max_samples=None):
    """
    Download IndicAccentDb from HuggingFace and organize files.
    
    Args:
        output_dir: Directory to save downloaded audio files
        subset: Which subset to download ("all", "train", "test")
        cache_dir: Cache directory for HuggingFace datasets
        max_samples: Maximum number of samples to download (for quick testing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading IndicAccentDb from HuggingFace...")
    print("=" * 80)
    
    try:
        # Load dataset from HuggingFace with audio disabled to avoid encoding issues
        if subset == "all":
            dataset = load_dataset("DarshanaS/IndicAccentDb", cache_dir=cache_dir)
        else:
            dataset = load_dataset("DarshanaS/IndicAccentDb", split=subset, cache_dir=cache_dir)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Dataset splits: {dataset.keys() if hasattr(dataset, 'keys') else subset}")
        
        # Create subdirectories
        adult_dir = output_dir / "adults"
        child_dir = output_dir / "children"
        adult_dir.mkdir(exist_ok=True)
        child_dir.mkdir(exist_ok=True)
        
        metadata_list = []
        
        # Process each split
        splits = dataset.keys() if hasattr(dataset, 'keys') else [subset]
        
        for split_name in splits:
            split_data = dataset[split_name] if hasattr(dataset, 'keys') else dataset
            print(f"\nProcessing split: {split_name}")
            print(f"  Total samples: {len(split_data)}")
            
            # Limit samples if max_samples is specified
            num_samples = min(len(split_data), max_samples) if max_samples else len(split_data)
            if max_samples:
                print(f"  Limited to: {num_samples} samples (quick mode)")
            
            # Process each audio sample
            for idx, sample in enumerate(tqdm(list(split_data)[:num_samples], desc=f"Processing {split_name}")):
                try:
                    # Extract metadata
                    audio_data = sample.get('audio', {})
                    audio_array = audio_data.get('array')
                    sampling_rate = audio_data.get('sampling_rate', 16000)
                    
                    # Get metadata fields
                    speaker_id = sample.get('speaker_id', f'speaker_{idx}')
                    native_language = sample.get('native_language', 'unknown')
                    age_group = sample.get('age_group', 'adult')  # 'adult' or 'child'
                    utterance_type = sample.get('utterance_type', 'sentence')  # 'word' or 'sentence'
                    utterance_text = sample.get('text', '')
                    
                    # Determine output directory based on age group
                    if age_group and 'child' in age_group.lower():
                        target_dir = child_dir
                    else:
                        target_dir = adult_dir
                    
                    # Create language subdirectory
                    lang_dir = target_dir / native_language
                    lang_dir.mkdir(exist_ok=True)
                    
                    # Save audio file
                    filename = f"{split_name}_{speaker_id}_{idx:05d}.wav"
                    filepath = lang_dir / filename
                    
                    if audio_array is not None:
                        sf.write(str(filepath), audio_array, sampling_rate)
                    
                    # Store metadata
                    metadata_list.append({
                        'filepath': str(filepath.relative_to(output_dir)),
                        'filename': filename,
                        'speaker_id': speaker_id,
                        'native_language': native_language,
                        'age_group': age_group,
                        'utterance_type': utterance_type,
                        'text': utterance_text,
                        'sampling_rate': sampling_rate,
                        'split': split_name,
                        'duration': len(audio_array) / sampling_rate if audio_array is not None else 0
                    })
                    
                except Exception as e:
                    print(f"  ⚠ Error processing sample {idx}: {str(e)}")
                    continue
        
        # Save metadata to CSV
        metadata_df = pd.DataFrame(metadata_list)
        metadata_path = output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        print("\n" + "=" * 80)
        print("Download and organization complete!")
        print("=" * 80)
        print(f"✓ Total files downloaded: {len(metadata_df)}")
        print(f"✓ Adult samples: {len(metadata_df[metadata_df['age_group'] != 'child'])}")
        print(f"✓ Child samples: {len(metadata_df[metadata_df['age_group'] == 'child'])}")
        print(f"✓ Languages: {metadata_df['native_language'].nunique()}")
        print(f"  {metadata_df['native_language'].value_counts().to_dict()}")
        print(f"✓ Metadata saved to: {metadata_path}")
        print(f"✓ Audio files saved to: {output_dir}")
        
        return metadata_df
        
    except Exception as e:
        # Fallback: attempt manual file download via huggingface_hub to bypass datasets audio encoding
        print(f"✗ Primary download failed: {e}")
        print("\n🔄 Attempting FALLBACK download using huggingface_hub (raw file streaming)...")
        try:
            from huggingface_hub import list_repo_files, hf_hub_download
        except ImportError:
            print("❌ huggingface_hub not installed. Run: pip install huggingface-hub")
            return pd.DataFrame([])

        repo_id = "DarshanaS/IndicAccentDb"
        try:
            files = list_repo_files(repo_id)
        except Exception as le:
            print(f"❌ Failed to list repository files: {le}")
            return pd.DataFrame([])

        # Filter audio files (common extensions)
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = [f for f in files if os.path.splitext(f)[1].lower() in audio_exts]

        if not audio_files:
            print("❌ No audio files found in repository file list. Fallback aborted.")
            return pd.DataFrame([])

        if max_samples:
            audio_files = audio_files[:max_samples]
            print(f"⚡ Limiting to {len(audio_files)} files (quick mode)")
        else:
            print(f"🔢 Total audio files discovered: {len(audio_files)}")

        # Prepare directories
        adult_dir = output_dir / "adults"
        adult_dir.mkdir(exist_ok=True)

        fallback_metadata = []
        print("\n⬇ Downloading raw audio files (fallback)...")
        for idx, rel_path in enumerate(tqdm(audio_files, desc="Fallback downloading")):
            try:
                # Derive language heuristically from path components
                parts = rel_path.split('/')
                lang_guess = 'unknown'
                for p in parts:
                    lower = p.lower()
                    if lower in ["hindi", "tamil", "telugu", "malayalam", "kannada"]:
                        lang_guess = p
                        break

                lang_dir = adult_dir / lang_guess
                lang_dir.mkdir(exist_ok=True)

                local_filename = f"fallback_{idx:05d}_{os.path.basename(rel_path)}"
                local_path = lang_dir / local_filename

                # Download file to local path
                try:
                    downloaded_path = hf_hub_download(repo_id=repo_id, filename=rel_path, local_dir=str(lang_dir))
                    # If hf_hub_download saved under different name, rename/move if needed
                    if Path(downloaded_path) != local_path:
                        if not local_path.exists():
                            os.replace(downloaded_path, local_path)
                except Exception as dl_e:
                    print(f"  ⚠ Skip {rel_path}: {dl_e}")
                    continue

                fallback_metadata.append({
                    'filepath': str(local_path.relative_to(output_dir)),
                    'filename': local_filename,
                    'speaker_id': f"unknown_{idx}",
                    'native_language': lang_guess,
                    'age_group': 'adult',
                    'utterance_type': 'unknown',
                    'text': '',
                    'sampling_rate': '',
                    'split': 'train',  # assign all to train for now
                    'duration': ''
                })
            except Exception as inner_e:
                print(f"  ⚠ Error in fallback processing {rel_path}: {inner_e}")
                continue

        if not fallback_metadata:
            print("❌ Fallback produced no downloadable samples.")
            return pd.DataFrame([])

        fallback_df = pd.DataFrame(fallback_metadata)
        metadata_path = output_dir / "metadata_fallback.csv"
        fallback_df.to_csv(metadata_path, index=False)

        print("\n" + "=" * 80)
        print("✅ Fallback download complete")
        print("=" * 80)
        print(f"✓ Files downloaded (fallback): {len(fallback_df)}")
        print(f"✓ Languages (heuristic): {fallback_df['native_language'].nunique()}")
        print(f"  {fallback_df['native_language'].value_counts().to_dict()}")
        print(f"✓ Metadata saved to: {metadata_path}")
        print(f"✓ Audio root: {output_dir}")
        print("⚠ All samples assigned to 'train' split. Create proper splits with create_splits.py after verification.")

        return fallback_df


def main():
    parser = argparse.ArgumentParser(
        description="Download IndicAccentDb dataset from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded audio files"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "train", "test", "validation"],
        help="Which subset to download"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to download per split (for quick testing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("IndicAccentDb Dataset Downloader")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Subset: {args.subset}")
    print(f"Cache directory: {args.cache_dir or 'default'}")
    if args.max_samples:
        print(f"Max samples per split: {args.max_samples} (QUICK MODE)")
    print("=" * 80 + "\n")
    
    metadata_df = download_and_organize_dataset(
        output_dir=args.output_dir,
        subset=args.subset,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples
    )
    
    print("\n✅ Dataset download completed successfully!")


if __name__ == "__main__":
    main()
