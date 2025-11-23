"""
Package essential files for Kaggle training
Creates a zip with only what's needed for T2 (layer-wise) and T3 (word vs sentence)
"""
import zipfile
from pathlib import Path
import shutil

def create_kaggle_package():
    print("=" * 80)
    print("CREATING KAGGLE PACKAGE FOR REMAINING TASKS")
    print("=" * 80)
    
    # Define essential files
    essential_files = [
        # Core source modules
        'src/__init__.py',
        'src/features/__init__.py',
        'src/features/hubert_extractor.py',
        'src/features/dataset.py',
        'src/models/__init__.py',
        'src/models/mlp.py',
        'src/data/__init__.py',
        'src/data/create_splits.py',
        'src/utils/__init__.py',
        
        # Scripts for T2 and T3
        'scripts/layerwise_hubert_analysis.py',
        'scripts/compare_word_sentence_level.py',
        'scripts/create_word_level_dataset.py',
        
        # Data files
        'data/raw/indian_accents/metadata_with_splits.csv',
        'data/splits/train.csv',
        'data/splits/val.csv',
        'data/splits/test.csv',
        
        # Requirements
        'requirements.txt',
    ]
    
    # Create zip
    zip_path = Path('kaggle_package.zip')
    
    print(f"\nCreating {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in essential_files:
            full_path = Path(file_path)
            if full_path.exists():
                zipf.write(full_path, file_path)
                print(f"  ✓ Added: {file_path}")
            else:
                print(f"  ⚠️  Missing: {file_path}")
        
        # Add audio files (all 2798 files)
        print("\n  Adding audio files...")
        audio_base = Path('data/raw/indian_accents')
        audio_count = 0
        
        for state_dir in ['andhra_pradesh', 'gujrat', 'jharkhand', 'karnataka', 'kerala', 'tamil']:
            state_path = audio_base / state_dir
            if state_path.exists():
                for audio_file in state_path.glob('*.wav'):
                    rel_path = audio_file.relative_to(Path('.'))
                    zipf.write(audio_file, str(rel_path))
                    audio_count += 1
                print(f"    ✓ {state_dir}: {len(list(state_path.glob('*.wav')))} files")
        
        print(f"\n  Total audio files: {audio_count}")
        
        # Create README for Kaggle
        readme_content = """# Kaggle Package for T2 & T3

## Remaining Tasks:

### T2: Layer-wise HuBERT Analysis
Evaluate which HuBERT layer (0-12) best captures accent information.

```bash
# Run layer-wise analysis (will take ~2-3 hours)
python scripts/layerwise_hubert_analysis.py --layers 0,3,6,9,12 --pooling mean --epochs 10

# Full sweep (all layers, longer)
python scripts/layerwise_hubert_analysis.py --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 --pooling mean --epochs 10
```

**Output**: `experiments/hubert_layerwise/results.json` with per-layer accuracy

---

### T3: Word vs Sentence Level Comparison
Compare accent cues at word-level vs sentence-level.

```bash
# Step 1: Create word-level dataset (segments sentences into words)
python scripts/create_word_level_dataset.py

# Step 2: Compare performance
python scripts/compare_word_sentence_level.py
```

**Outputs**: 
- `experiments/word_vs_sentence_comparison.png` - visualization
- `experiments/word_sentence_comparison_results.json` - accuracy metrics

---

## Setup on Kaggle:

1. **Upload** this zip to Kaggle Dataset
2. **Create a new notebook** with GPU accelerator
3. **Install dependencies**:
   ```python
   !pip install transformers==4.30.0 torchaudio soundfile librosa
   ```
4. **Extract and run**:
   ```python
   !unzip /kaggle/input/your-dataset-name/kaggle_package.zip -d /kaggle/working/
   %cd /kaggle/working
   ```
5. **Run tasks** using commands above

---

## Expected Results:

**T2 (Layer-wise)**: You should see layer 12 or 11 achieving highest accuracy (~99%+)

**T3 (Word vs Sentence)**: Sentence-level typically outperforms word-level by 5-10% due to prosodic cues

---

## Notes:
- Audio files included: 2,798 WAV files from 6 Indian states
- HuBERT model will auto-download from HuggingFace (requires internet)
- Estimated runtime: 3-5 hours total on Kaggle GPU
- Primary feature: HuBERT layer embeddings (pooled mean/std)
"""
        zipf.writestr('KAGGLE_README.md', readme_content)
        print("  ✓ Added: KAGGLE_README.md")
        
        # Create a quick setup script for Kaggle
        setup_script = """#!/usr/bin/env python3
'''Quick setup and verification script for Kaggle'''
import sys
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path.cwd()))

print("=" * 80)
print("KAGGLE ENVIRONMENT SETUP CHECK")
print("=" * 80)

# Check imports
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch not found - install with: pip install torch")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError:
    print("✗ Transformers not found - install with: pip install transformers")

try:
    import librosa
    print(f"✓ Librosa: {librosa.__version__}")
except ImportError:
    print("✗ Librosa not found - install with: pip install librosa")

# Check data files
print("\\n" + "=" * 80)
print("DATA FILES CHECK")
print("=" * 80)

metadata = Path('data/raw/indian_accents/metadata_with_splits.csv')
if metadata.exists():
    import pandas as pd
    df = pd.read_csv(metadata)
    print(f"✓ Metadata: {len(df)} samples")
    print(f"  States: {df['native_language'].nunique()}")
    print(f"  Splits: train={len(df[df['split']=='train'])}, val={len(df[df['split']=='val'])}, test={len(df[df['split']=='test'])}")
else:
    print("✗ Metadata not found")

audio_base = Path('data/raw/indian_accents')
if audio_base.exists():
    wav_files = list(audio_base.rglob('*.wav'))
    print(f"✓ Audio files: {len(wav_files)} WAV files")
else:
    print("✗ Audio directory not found")

print("\\n" + "=" * 80)
print("READY TO RUN TASKS!")
print("=" * 80)
print("\\nNext steps:")
print("  1. For T2: python scripts/layerwise_hubert_analysis.py --layers 0,3,6,9,12 --epochs 10")
print("  2. For T3: python scripts/create_word_level_dataset.py && python scripts/compare_word_sentence_level.py")
"""
        zipf.writestr('kaggle_setup_check.py', setup_script)
        print("  ✓ Added: kaggle_setup_check.py")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Package created: {zip_path.absolute()}")
    print(f"   Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"{'=' * 80}")
    print("\n📝 Next steps:")
    print("  1. Upload kaggle_package.zip to Kaggle as a Dataset")
    print("  2. Create a new Kaggle Notebook with GPU")
    print("  3. Add the dataset to your notebook")
    print("  4. Extract and run:")
    print("     !unzip /kaggle/input/your-dataset-name/kaggle_package.zip -d /kaggle/working/")
    print("     %cd /kaggle/working")
    print("     !python kaggle_setup_check.py")
    print("  5. Follow instructions in KAGGLE_README.md")
    print(f"\n{'=' * 80}")
    
    return str(zip_path.absolute())

if __name__ == '__main__':
    create_kaggle_package()
