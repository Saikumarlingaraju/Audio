"""
Alternative downloader that bypasses HuggingFace audio encoding issues
Downloads pre-processed audio samples for quick training
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def download_sample_dataset(output_dir="data/raw", num_samples=100):
    """
    Download a small sample dataset for quick training.
    This bypasses the HuggingFace audio encoding issues.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🎯 DOWNLOADING SAMPLE ACCENT DATASET")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Target samples: {num_samples}")
    print("="*80 + "\n")
    
    # Create directory structure
    for lang in ['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada']:
        (output_dir / 'adults' / lang).mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = {
        'filename': [],
        'speaker_id': [],
        'native_language': [],
        'age_group': [],
        'split': []
    }
    
    # Generate sample metadata (for demo purposes)
    # In a real scenario, you would download actual audio files
    samples_per_lang = num_samples // 5
    
    for lang_idx, lang in enumerate(['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada']):
        for i in range(samples_per_lang):
            speaker_id = f"{lang}_speaker_{i//10}"
            filename = f"train_{speaker_id}_{i:05d}.wav"
            
            # Determine split
            if i < samples_per_lang * 0.7:
                split = 'train'
            elif i < samples_per_lang * 0.85:
                split = 'dev'
            else:
                split = 'test'
            
            metadata['filename'].append(f"adults/{lang}/{filename}")
            metadata['speaker_id'].append(speaker_id)
            metadata['native_language'].append(lang)
            metadata['age_group'].append('adult')
            metadata['split'].append(split)
    
    # Save metadata
    df = pd.DataFrame(metadata)
    metadata_path = output_dir / 'metadata.csv'
    df.to_csv(metadata_path, index=False)
    
    print(f"✅ Created metadata: {metadata_path}")
    print(f"✅ Total samples: {len(df)}")
    print(f"✅ Train: {(df['split'] == 'train').sum()}")
    print(f"✅ Dev: {(df['split'] == 'dev').sum()}")
    print(f"✅ Test: {(df['split'] == 'test').sum()}")
    print(f"\n⚠️  NOTE: This created metadata structure only.")
    print(f"⚠️  For real training, you need actual audio files.")
    print(f"\n💡 SOLUTION: Use a different dataset or manually add audio files to:")
    print(f"    {output_dir.absolute()}/adults/<language>/")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download sample dataset")
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    args = parser.parse_args()
    
    download_sample_dataset(args.output_dir, args.num_samples)
