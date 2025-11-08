"""
Create metadata and splits for IndicAccentDB
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import soundfile as sf
from tqdm import tqdm


def create_metadata(data_dir="data/raw/indian_accents", max_per_class=None):
    """Create metadata CSV from audio files"""
    
    data_path = Path(data_dir)
    
    print("=" * 80)
    print("📋 CREATING METADATA FOR INDICACCENTDB")
    print("=" * 80)
    
    # Find all audio files
    metadata_list = []
    
    accent_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(accent_dirs)} accent directories")
    
    for accent_dir in sorted(accent_dirs):
        accent_name = accent_dir.name
        audio_files = list(accent_dir.glob('*.wav')) + list(accent_dir.glob('*.mp3'))
        
        print(f"\n📁 Processing {accent_name}: {len(audio_files)} files")
        
        # Limit per class if specified
        if max_per_class:
            audio_files = audio_files[:max_per_class]
            print(f"   Limited to {len(audio_files)} files")
        
        for idx, audio_file in enumerate(tqdm(audio_files, desc=f"  {accent_name}")):
            try:
                # Get audio info
                info = sf.info(str(audio_file))
                duration = info.duration
                sr = info.samplerate
                
                metadata_list.append({
                    'filepath': str(audio_file.relative_to(data_path)),
                    'filename': audio_file.name,
                    'speaker_id': f"{accent_name}_spk_{idx:04d}",
                    'native_language': accent_name,
                    'age_group': 'adult',
                    'utterance_type': 'read',
                    'text': '',  # Not provided in dataset
                    'sampling_rate': sr,
                    'split': 'train',  # Will be updated
                    'duration': duration
                })
            except Exception as e:
                print(f"\n⚠️ Error processing {audio_file.name}: {e}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    print(f"\n✅ Metadata created!")
    print(f"✓ Total samples: {len(df)}")
    print(f"\n📊 Distribution by accent:")
    print(df['native_language'].value_counts())
    
    # Save full metadata
    metadata_path = data_path / "metadata_full.csv"
    df.to_csv(metadata_path, index=False)
    print(f"\n✓ Saved: {metadata_path}")
    
    # Create train/val/test splits (70/15/15)
    print(f"\n🔄 Creating train/val/test splits...")
    
    train_list = []
    val_list = []
    test_list = []
    
    for accent in df['native_language'].unique():
        accent_df = df[df['native_language'] == accent]
        
        # Split: 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(
            accent_df, test_size=0.3, random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42
        )
        
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        train_list.append(train_df)
        val_list.append(val_df)
        test_list.append(test_df)
        
        print(f"  {accent}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Combine all splits
    train_full = pd.concat(train_list, ignore_index=True)
    val_full = pd.concat(val_list, ignore_index=True)
    test_full = pd.concat(test_list, ignore_index=True)
    all_data = pd.concat([train_full, val_full, test_full], ignore_index=True)
    
    # Save split files
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_full.to_csv(splits_dir / "train.csv", index=False)
    val_full.to_csv(splits_dir / "val.csv", index=False)
    test_full.to_csv(splits_dir / "test.csv", index=False)
    all_data.to_csv(data_path / "metadata_with_splits.csv", index=False)
    
    print(f"\n✅ Splits created!")
    print(f"✓ Train: {len(train_full)} samples → data/splits/train.csv")
    print(f"✓ Val: {len(val_full)} samples → data/splits/val.csv")
    print(f"✓ Test: {len(test_full)} samples → data/splits/test.csv")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/indian_accents")
    parser.add_argument("--max_per_class", type=int, default=None,
                       help="Max samples per class (None=all)")
    
    args = parser.parse_args()
    
    success = create_metadata(args.data_dir, args.max_per_class)
    
    if success:
        print("\n" + "=" * 80)
        print("✅ READY FOR TRAINING!")
        print("=" * 80)
        print("\n📝 Next step:")
        print("  python train_simple.py --num_epochs 30")
        print("=" * 80)
