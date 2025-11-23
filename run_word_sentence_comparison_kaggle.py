"""
Run Word vs Sentence Level Comparison on Kaggle
Streamlined version optimized for Kaggle GPU environment

Instructions for Kaggle:
1. Upload this script to your Kaggle notebook
2. Enable GPU: Settings → Accelerator → GPU T4 x2
3. Enable Internet: Settings → Internet → On (for downloading HuBERT model)
4. Run: !python run_word_sentence_comparison_kaggle.py
"""

import sys
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path.cwd()))

# Import project modules
from src.features.hubert_extractor import HuBERTExtractor
from src.models.mlp import MLPClassifier


class SimpleFeatureDataset(Dataset):
    """Simple dataset for pre-extracted features"""
    def __init__(self, features_dict, filepaths, label_encoder=None):
        self.features = []
        self.labels = []
        self.label_encoder = label_encoder if label_encoder else LabelEncoder()
        
        # Convert list format to dict if needed
        if isinstance(features_dict, list):
            temp_dict = {}
            for item in features_dict:
                if isinstance(item, dict) and 'filepath' in item:
                    fp = item['filepath'].replace('\\', '/')
                    
                    # Remove 'data/raw/indian_accents/' prefix for sentence-level
                    # Keep 'word_level/' prefix for word-level
                    if fp.startswith('data/raw/indian_accents/'):
                        fp = fp.replace('data/raw/indian_accents/', '')
                    
                    # Handle different key names for label
                    if 'label' not in item and 'native_language' in item:
                        item['label'] = item['native_language']
                    temp_dict[fp] = item
            features_dict = temp_dict
        
        labels_list = []
        matched = 0
        for fp in filepaths:
            feature_item = None
            
            # Try exact match first
            if fp in features_dict:
                feature_item = features_dict[fp]
            # Try with path normalization
            elif fp.replace('\\', '/') in features_dict:
                normalized = fp.replace('\\', '/')
                feature_item = features_dict[normalized]
            
            if feature_item:
                self.features.append(feature_item['features'])
                # Handle different label key names
                label = feature_item.get('label') or feature_item.get('native_language')
                labels_list.append(label)
                matched += 1
        
        if matched == 0 and len(filepaths) > 0:
            print(f"\n⚠️  WARNING: No paths matched!")
            print(f"  Requested {len(filepaths)} files, matched {matched}")
            if len(filepaths) > 0:
                print(f"  Sample requested path: {filepaths[0]}")
            if features_dict:
                print(f"  Sample feature key: {list(features_dict.keys())[0]}")
        
        if label_encoder is None:
            self.labels = self.label_encoder.fit_transform(labels_list)
        else:
            self.labels = self.label_encoder.transform(labels_list)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]


def create_word_level_dataset_simple():
    """Simplified word-level dataset creation using VAD"""
    import librosa
    import soundfile as sf
    
    print("\n" + "="*80)
    print("STEP 1: Creating Word-Level Dataset")
    print("="*80 + "\n")
    
    # Paths
    metadata_path = Path('data/raw/indian_accents/metadata_with_splits.csv')
    output_dir = Path('data/word_level')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = pd.read_csv(metadata_path)
    
    # Create output directories
    states = metadata['native_language'].unique()
    for state in states:
        (output_dir / state).mkdir(parents=True, exist_ok=True)
    
    word_level_data = []
    word_count = 0
    
    print(f"Processing {len(metadata)} audio files to extract words...")
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting words"):
        audio_path = Path('data/raw/indian_accents') / row['filepath']
        
        if not audio_path.exists():
            continue
        
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            
            # Simple energy-based word detection
            frame_length = int(0.025 * sr)  # 25ms
            hop_length = int(0.010 * sr)    # 10ms
            
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            energy_db = librosa.amplitude_to_db(energy, ref=np.max)
            
            # Find speech segments
            is_speech = energy_db > -40  # dB threshold
            
            # Get word boundaries
            word_segments = []
            in_word = False
            word_start = 0
            
            for i, speech in enumerate(is_speech):
                sample_idx = i * hop_length
                
                if speech and not in_word:
                    word_start = sample_idx
                    in_word = True
                elif not speech and in_word:
                    word_end = sample_idx
                    duration = (word_end - word_start) / sr
                    
                    if duration > 0.2:  # At least 200ms
                        word_segments.append((word_start, word_end))
                    
                    in_word = False
            
            if in_word:
                word_segments.append((word_start, len(audio)))
            
            # Limit to 10 words per utterance
            word_segments = word_segments[:10]
            
            # Skip if too few words
            if len(word_segments) < 3:
                continue
            
            # Save word segments
            for word_idx, (start, end) in enumerate(word_segments):
                word_audio = audio[start:end]
                
                if len(word_audio) < sr * 0.2:
                    continue
                
                output_filename = f"{audio_path.stem}_word{word_idx:02d}.wav"
                output_filepath = output_dir / row['native_language'] / output_filename
                
                sf.write(str(output_filepath), word_audio, sr)
                
                word_level_data.append({
                    'filepath': f"word_level/{row['native_language']}/{output_filename}",
                    'native_language': row['native_language'],
                    'split': row['split'],
                    'source_file': row['filepath'],
                    'word_index': word_idx
                })
                
                word_count += 1
        
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue
    
    # Save metadata
    word_df = pd.DataFrame(word_level_data)
    word_df.to_csv(output_dir / 'word_level_metadata.csv', index=False)
    
    print(f"\n✓ Word-level dataset created")
    print(f"  Total word segments: {word_count}")
    print(f"  Metadata: {output_dir / 'word_level_metadata.csv'}")
    
    print("\nDistribution by state:")
    print(word_df.groupby('native_language').size())
    
    print("\nDistribution by split:")
    print(word_df.groupby('split').size())
    
    return word_df


def extract_features_for_level(level='sentence', layer=3):
    """Extract HuBERT features for specified level"""
    print(f"\n{'='*80}")
    print(f"STEP 2: Extracting {level.upper()}-Level Features (Layer {layer})")
    print(f"{'='*80}\n")
    
    if level == 'sentence':
        metadata_path = Path('data/raw/indian_accents/metadata_with_splits.csv')
        audio_base = Path('data/raw/indian_accents')
        output_file = Path(f'data/features/indian_accents/hubert_layer{layer}_mean.pkl')
    else:
        metadata_path = Path('data/word_level/word_level_metadata.csv')
        audio_base = Path('data')
        output_file = Path(f'data/features/word_level/hubert_layer{layer}_mean.pkl')
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    if output_file.exists():
        print(f"✓ Features already exist at {output_file}")
        with open(output_file, 'rb') as f:
            features_data = pickle.load(f)
        
        # Check format and report
        num_samples = 0
        if isinstance(features_data, dict):
            num_samples = len(features_data)
            print(f"  Loaded {num_samples} feature samples (dict format)")
        elif isinstance(features_data, list):
            num_samples = len(features_data)
            print(f"  Loaded {num_samples} feature samples (list format)")
        else:
            print(f"  Loaded features (unknown format: {type(features_data)})")
        
        # If empty, delete and re-extract
        if num_samples == 0:
            print(f"  ⚠️  Feature file is empty, re-extracting...")
            output_file.unlink()
        else:
            return output_file
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print(f"Processing {len(metadata)} audio files...")
    
    # Initialize extractor
    extractor = HuBERTExtractor()
    
    features_dict = {}
    errors = 0
    missing_files = 0
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Extracting layer {layer}"):
        # Construct full audio path
        if level == 'sentence':
            audio_path = audio_base / row['filepath']
        else:
            # For word-level, filepath already includes 'word_level/' prefix
            audio_path = Path('data') / row['filepath']
        
        if not audio_path.exists():
            missing_files += 1
            if missing_files <= 3:  # Show first 3 missing files
                print(f"\n  Missing: {audio_path}")
            continue
        
        try:
            # Extract mean pooled features
            result = extractor.extract_from_file(str(audio_path), extract_layer=layer, pooling='mean')
            features = result['embeddings'][f'layer_{layer}']
            
            # Convert to numpy if it's a torch tensor
            if hasattr(features, 'numpy'):
                features = features.numpy()
            
            # Store with normalized path
            norm_path = row['filepath'].replace('\\', '/')
            features_dict[norm_path] = {
                'features': features,
                'label': row['native_language'],
                'split': row['split']
            }
        
        except Exception as e:
            errors += 1
            if errors <= 3:  # Show first 3 errors
                print(f"\n  Error processing {audio_path}: {e}")
            continue
    
    # Save features
    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"\n✓ Extracted {len(features_dict)} samples")
    if missing_files > 0:
        print(f"  Missing files: {missing_files}")
    if errors > 0:
        print(f"  Errors: {errors}")
    print(f"  Saved to: {output_file}")
    
    return output_file


def train_model(features_path, metadata_path, output_dir, level_name, num_epochs=20):
    """Train MLP classifier on features"""
    print(f"\n{'='*80}")
    print(f"STEP 3: Training on {level_name} Data")
    print(f"{'='*80}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load features
    with open(features_path, 'rb') as f:
        features_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(features_dict)} feature samples")
    
    # Load metadata and split
    metadata = pd.read_csv(metadata_path)
    
    # FIX PATH FORMAT: Convert Windows backslashes to Linux forward slashes
    if 'filepath' in metadata.columns:
        metadata['filepath'] = metadata['filepath'].str.replace('\\', '/', regex=False)
        print(f"✓ Normalized file paths to Linux format")
    
    train_df = metadata[metadata['split'] == 'train']
    val_df = metadata[metadata['split'] == 'val']
    test_df = metadata[metadata['split'] == 'test']
    
    print(f"✓ Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Convert features_dict from list to dict if needed
    if isinstance(features_dict, list):
        print(f"✓ Converting feature list to dictionary format...")
        # Debug: Check what keys are in the first item
        if len(features_dict) > 0:
            print(f"  Sample item keys: {list(features_dict[0].keys())}")
        
        new_dict = {}
        for item in features_dict:
            if isinstance(item, dict) and 'filepath' in item:
                fp = item['filepath'].replace('\\', '/')
                
                # Remove 'data/raw/indian_accents/' prefix for sentence-level
                # Keep 'word_level/' prefix for word-level
                if fp.startswith('data/raw/indian_accents/'):
                    fp = fp.replace('data/raw/indian_accents/', '')
                
                # Handle different possible key names
                label_key = 'label' if 'label' in item else 'native_language' if 'native_language' in item else None
                
                if label_key:
                    new_dict[fp] = {
                        'features': item['features'],
                        'label': item[label_key],
                        'split': item.get('split', 'train')
                    }
        features_dict = new_dict
        print(f"  Converted {len(features_dict)} feature samples")
    
    # Debug: Check path format matching
    if features_dict:
        sample_feature_path = list(features_dict.keys())[0]
        sample_metadata_path = train_df['filepath'].iloc[0] if len(train_df) > 0 else "N/A"
        print(f"  Sample feature key: {sample_feature_path}")
        print(f"  Sample metadata path: {sample_metadata_path}")
    
    # Create datasets
    train_dataset = SimpleFeatureDataset(features_dict, train_df['filepath'].tolist())
    label_encoder = train_dataset.label_encoder
    val_dataset = SimpleFeatureDataset(features_dict, val_df['filepath'].tolist(), label_encoder=label_encoder)
    test_dataset = SimpleFeatureDataset(features_dict, test_df['filepath'].tolist(), label_encoder=label_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model setup
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]
    num_classes = len(label_encoder.classes_)
    
    print(f"✓ Dataset created with {len(train_dataset)} samples")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {list(label_encoder.classes_)}")
    print(f"  Feature dimension: {input_dim}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = MLPClassifier(input_dim=input_dim, hidden_dims=[256, 128], 
                         num_classes=num_classes, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    history = {'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_epoch = 0
    
    print(f"\nTraining for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_preds, train_labels = [], []
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}: train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_path / 'best_model.pt')
    
    # Test
    model.load_state_dict(torch.load(output_path / 'best_model.pt'))
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\n{level_name}: best_val={best_val_acc*100:.2f}% (epoch {best_epoch}) test={test_acc*100:.2f}%")
    
    results = {
        'level': level_name,
        'train_acc': history['train_acc'][-1],
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'history': history
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_comparison_plot(results):
    """Create visualization comparing word vs sentence performance"""
    print(f"\n{'='*80}")
    print("STEP 4: Creating Comparison Plots")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Accuracy comparison
    levels = ['Sentence-Level', 'Word-Level']
    train_accs = [results['sentence']['train_acc'] * 100, results['word']['train_acc'] * 100]
    val_accs = [results['sentence']['val_acc'] * 100, results['word']['val_acc'] * 100]
    test_accs = [results['sentence']['test_acc'] * 100, results['word']['test_acc'] * 100]
    
    x = range(len(levels))
    width = 0.25
    
    axes[0].bar([i - width for i in x], train_accs, width, label='Train', alpha=0.8, color='#2ecc71')
    axes[0].bar(x, val_accs, width, label='Validation', alpha=0.8, color='#3498db')
    axes[0].bar([i + width for i in x], test_accs, width, label='Test', alpha=0.8, color='#e74c3c')
    
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy Comparison: Word vs Sentence Level', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(levels)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([90, 100])
    
    # Add value labels on bars
    for i in x:
        axes[0].text(i - width, train_accs[i] + 0.5, f'{train_accs[i]:.1f}%', ha='center', fontsize=9)
        axes[0].text(i, val_accs[i] + 0.5, f'{val_accs[i]:.1f}%', ha='center', fontsize=9)
        axes[0].text(i + width, test_accs[i] + 0.5, f'{test_accs[i]:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Training curves
    epochs_s = range(1, len(results['sentence']['history']['val_acc']) + 1)
    epochs_w = range(1, len(results['word']['history']['val_acc']) + 1)
    
    axes[1].plot(epochs_s, [acc * 100 for acc in results['sentence']['history']['val_acc']], 
                label='Sentence-Level', marker='o', markersize=4, linewidth=2, color='#3498db')
    axes[1].plot(epochs_w, [acc * 100 for acc in results['word']['history']['val_acc']], 
                label='Word-Level', marker='s', markersize=4, linewidth=2, color='#e74c3c')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Training Curves: Word vs Sentence Level', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([90, 100])
    
    plt.tight_layout()
    output_file = 'experiments/word_vs_sentence_comparison.png'
    Path('experiments').mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {output_file}")
    
    return output_file


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("WORD VS SENTENCE LEVEL COMPARISON")
    print("="*80 + "\n")
    
    # Configuration
    LAYER = 3  # Use Layer 3 (best from layer-wise analysis)
    EPOCHS = 20
    
    results = {}
    
    # ========== SENTENCE LEVEL ==========
    # Check if sentence-level results already exist
    sentence_results_path = Path(f'experiments/sentence_level_layer{LAYER}/results.json')
    
    if sentence_results_path.exists():
        print("✓ Loading existing sentence-level results...")
        with open(sentence_results_path, 'r') as f:
            results['sentence'] = json.load(f)
        print(f"  Test Accuracy: {results['sentence']['test_acc']*100:.2f}%")
    else:
        # Extract features for sentence-level
        sentence_features = extract_features_for_level('sentence', layer=LAYER)
        
        # Train on sentence-level
        results['sentence'] = train_model(
            features_path=sentence_features,
            metadata_path='data/raw/indian_accents/metadata_with_splits.csv',
            output_dir=f'experiments/sentence_level_layer{LAYER}',
            level_name='Sentence-Level',
            num_epochs=EPOCHS
        )
    
    # ========== WORD LEVEL ==========
    # Create word-level dataset if needed
    word_metadata_path = Path('data/word_level/word_level_metadata.csv')
    if not word_metadata_path.exists():
        create_word_level_dataset_simple()
    else:
        print("\n✓ Word-level dataset already exists")
        word_df = pd.read_csv(word_metadata_path)
        print(f"  Total word segments: {len(word_df)}")
    
    # Extract features for word-level
    word_features = extract_features_for_level('word', layer=LAYER)
    
    # Train on word-level
    results['word'] = train_model(
        features_path=word_features,
        metadata_path=word_metadata_path,
        output_dir=f'experiments/word_level_layer{LAYER}',
        level_name='Word-Level',
        num_epochs=EPOCHS
    )
    
    # ========== COMPARISON ==========
    # Create comparison plot
    create_comparison_plot(results)
    
    # Save comparison results
    comparison_file = 'experiments/word_sentence_comparison_results.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Level':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 56)
    print(f"{'Sentence-Level':<20} {results['sentence']['train_acc']*100:>9.2f}% "
          f"{results['sentence']['val_acc']*100:>10.2f}% {results['sentence']['test_acc']*100:>10.2f}%")
    print(f"{'Word-Level':<20} {results['word']['train_acc']*100:>9.2f}% "
          f"{results['word']['val_acc']*100:>10.2f}% {results['word']['test_acc']*100:>10.2f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    diff = results['sentence']['test_acc'] - results['word']['test_acc']
    
    if diff > 0.01:
        print(f"✓ Sentence-level performs BETTER by {abs(diff)*100:.2f}%")
        print("  → Longer context provides better accent discrimination")
        print("  → Prosodic and rhythmic patterns are important cues")
        print("  → Sentence-level captures more holistic accent characteristics")
    elif diff < -0.01:
        print(f"✓ Word-level performs BETTER by {abs(diff)*100:.2f}%")
        print("  → Phonetic details at word level are more discriminative")
        print("  → Shorter segments may reduce noise and variability")
        print("  → Word-level features capture fine-grained accent markers")
    else:
        print(f"✓ Both levels perform SIMILARLY (difference: {abs(diff)*100:.2f}%)")
        print("  → Accent information is preserved across granularities")
        print("  → HuBERT captures robust accent features at multiple scales")
    
    print(f"\n✓ Results saved to {comparison_file}")
    print(f"✓ Plot saved to experiments/word_vs_sentence_comparison.png")
    
    print("\n" + "="*80)
    print("WORD VS SENTENCE COMPARISON COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
