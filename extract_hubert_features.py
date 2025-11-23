"""
Extract HuBERT features for Indian Accents dataset (PRIMARY)
"""
from src.features.hubert_extractor import HuBERTExtractor
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import torch

def extract_hubert_features():
    """Extract HuBERT layer 12 pooled features (mean + std)"""
    
    print("=" * 80)
    print("🎵 EXTRACTING HuBERT FEATURES - INDIAN ACCENTS (PRIMARY)")
    print("=" * 80)
    
    # Initialize HuBERT extractor
    print("\n📦 Loading HuBERT model...")
    extractor = HuBERTExtractor(model_name="facebook/hubert-base-ls960")
    
    # Load metadata
    metadata_path = 'data/raw/indian_accents/metadata_with_splits.csv'
    df = pd.read_csv(metadata_path)
    
    print(f"\n📊 Dataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Accents: {df['native_language'].nunique()}")
    print(f"  States: {', '.join(df['native_language'].unique())}")
    print(f"  Splits: Train={len(df[df['split']=='train'])}, Val={len(df[df['split']=='val'])}, Test={len(df[df['split']=='test'])}")
    print("=" * 80)
    
    # Extract features
    features_list = []
    base_path = Path('data/raw/indian_accents')
    errors = []
    
    print(f"\n🔄 Extracting HuBERT layer 12 embeddings (this takes ~30-45 minutes)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Extracting HuBERT'):
        audio_path = base_path / row['filepath']
        
        if not audio_path.exists():
            errors.append((audio_path.name, "File not found"))
            continue
        
        try:
            # Extract layer 12 with mean pooling
            result = extractor.extract_from_file(
                str(audio_path), 
                extract_layer=12,  # Layer 12 (best for accent features)
                pooling='mean'
            )
            
            if result is None:
                errors.append((audio_path.name, "Extraction failed"))
                continue
            
            # Get embeddings
            embeddings = result['embeddings']
            layer_12_mean = embeddings['layer_12']  # Shape: (768,)
            
            # Also compute std pooling for richer representation
            audio, sr = extractor.load_audio(str(audio_path))
            if audio is not None:
                # Extract frame-level embeddings (no pooling)
                frame_embeddings = extractor.extract_embeddings(
                    audio, 
                    sr=sr, 
                    extract_layer=12,
                    pooling=None  # Get all frames
                )
                
                if frame_embeddings and 'layer_12' in frame_embeddings:
                    frames = frame_embeddings['layer_12']  # Shape: (T, 768)
                    layer_12_std = np.std(frames, axis=0)  # Shape: (768,)
                    
                    # Concatenate mean + std = 1536-dim feature
                    features = np.concatenate([layer_12_mean, layer_12_std])
                else:
                    # Fallback: just use mean (768-dim)
                    features = layer_12_mean
            else:
                # Fallback: just use mean (768-dim)
                features = layer_12_mean
            
            features_list.append({
                'features': features,
                'native_language': row['native_language'],
                'split': row['split'],
                'filepath': str(audio_path),
                'speaker_id': row.get('speaker_id', 'unknown'),
                'age_group': row.get('age_group', 'unknown')
            })
            
        except Exception as e:
            errors.append((audio_path.name, str(e)))
    
    print(f"\n✅ Feature extraction complete!")
    print(f"✓ Successful: {len(features_list)}/{len(df)}")
    
    if errors:
        print(f"⚠️ Errors: {len(errors)}")
        print(f"First 5 errors:")
        for filename, error in errors[:5]:
            print(f"  - {filename}: {error}")
    
    if len(features_list) == 0:
        print("❌ No features extracted! Check errors above.")
        return None
    
    # Check feature dimensions
    feature_dim = features_list[0]['features'].shape[0]
    print(f"\n📐 Feature shape: {feature_dim}-dimensional")
    if feature_dim == 1536:
        print("   ✓ HuBERT layer 12 pooled (768 mean + 768 std) ✅ OPTIMAL")
    elif feature_dim == 768:
        print("   ⚠️ HuBERT layer 12 mean only (no std)")
    
    # Save features
    output_dir = Path('data/features/indian_accents')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'hubert_layer12_mean.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(features_list, f)
    
    print(f"\n✓ Features saved: {output_path}")
    print(f"✓ Total size: {len(features_list)} samples × {feature_dim} dims")
    
    # Save feature info
    info = {
        'n_samples': len(features_list),
        'feature_dim': feature_dim,
        'model': 'facebook/hubert-base-ls960',
        'layer': 12,
        'pooling': 'mean+std' if feature_dim == 1536 else 'mean',
        'errors': len(errors)
    }
    
    info_path = output_dir / 'hubert_info.pkl'
    with open(info_path, 'wb') as f:
        pickle.dump(info, f)
    
    print(f"✓ Feature info saved: {info_path}")
    
    return output_path

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HuBERT FEATURE EXTRACTION - PRIMARY FEATURES")
    print("=" * 80)
    print("\n⏱️ This will take 30-45 minutes to process 2,798 audio files")
    print("💡 Go grab a coffee! ☕\n")
    
    features_path = extract_hubert_features()
    
    if features_path:
        print(f"\n" + "=" * 80)
        print(f"✅ HUBERT FEATURES READY!")
        print(f"=" * 80)
        print(f"\n📝 Next steps:")
        print(f"  1. Train HuBERT-based model (PRIMARY):")
        print(f"     python train_simple.py --num_epochs 30")
        print(f"\n  2. Or train all models:")
        print(f"     python scripts\\train_all_models.py")
        print(f"\n  3. Or run full pipeline:")
        print(f"     python run_all_experiments.py")
        print(f"=" * 80)
    else:
        print("\n❌ Feature extraction failed! Check errors above.")
