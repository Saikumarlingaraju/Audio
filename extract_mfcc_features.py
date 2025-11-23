"""
Extract MFCC features for Indian Accents dataset
"""
from src.features.mfcc_extractor import MFCCExtractor
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm

def extract_features():
    # Initialize extractor
    extractor = MFCCExtractor(sr=16000, n_mfcc=40, include_delta=True, include_delta_delta=True)
    
    # Load metadata
    df = pd.read_csv('data/raw/indian_accents/metadata_with_splits.csv')
    print(f"=" * 80)
    print(f"🎵 EXTRACTING MFCC FEATURES - INDIAN ACCENTS")
    print(f"=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Accents: {df['native_language'].nunique()}")
    print(f"=" * 80)
    
    # Extract features
    features_list = []
    base_path = Path('data/raw/indian_accents')
    errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Extracting MFCC'):
        audio_path = base_path / row['filepath']
        
        try:
            feat_dict = extractor.extract_from_file(str(audio_path), return_frames=False)
            # Extract just the feature array (not the dict)
            feat_array = feat_dict['features'] if isinstance(feat_dict, dict) else feat_dict
            features_list.append({
                'features': feat_array,
                'native_language': row['native_language'],
                'split': row['split'],
                'filepath': str(audio_path)
            })
        except Exception as e:
            errors.append((audio_path.name, str(e)))
    
    print(f"\n✅ Feature extraction complete!")
    print(f"✓ Successful: {len(features_list)}/{len(df)}")
    if errors:
        print(f"⚠️ Errors: {len(errors)}")
        for filename, error in errors[:5]:
            print(f"  - {filename}: {error}")
    
    # Save features
    output_dir = Path('data/features/indian_accents')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'mfcc_stats.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(features_list, f)
    
    print(f"\n✓ Features saved: {output_path}")
    print(f"✓ Feature shape: 600-dimensional (40 MFCCs × 5 stats × 3 types)")
    
    return output_path

if __name__ == "__main__":
    features_path = extract_features()
    print(f"\n" + "=" * 80)
    print(f"✅ READY FOR TRAINING!")
    print(f"=" * 80)
    print(f"\n📝 Next step:")
    print(f"  python train_simple.py --features_path {features_path} --num_epochs 30")
    print(f"=" * 80)
