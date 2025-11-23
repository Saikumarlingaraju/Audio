"""
Cross-Age Generalization Analysis: Adult → Child
Train on adult speech, test on children's speech

Run this on Kaggle with GPU enabled

Instructions:
1. Upload kaggle_package.zip + child_audios folder to Kaggle
2. Enable GPU: Settings → GPU T4 x2
3. Enable Internet: Settings → Internet → On
4. Run: !python cross_age_generalization_kaggle.py
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
sys.path.insert(0, str(Path.cwd()))

# Import project modules
from src.features.hubert_extractor import HuBERTExtractor
from src.features.mfcc_extractor import MFCCExtractor
from src.models.mlp import MLPClassifier


class SimpleFeatureDataset(Dataset):
    """Dataset for pre-extracted features"""
    def __init__(self, features_dict, filepaths, label_encoder=None):
        self.features = []
        self.labels = []
        self.label_encoder = label_encoder if label_encoder else LabelEncoder()
        
        # Handle list format
        if isinstance(features_dict, list):
            temp_dict = {}
            for item in features_dict:
                if isinstance(item, dict) and 'filepath' in item:
                    fp = item['filepath'].replace('\\', '/')
                    if fp.startswith('data/raw/indian_accents/'):
                        fp = fp.replace('data/raw/indian_accents/', '')
                    if 'label' not in item and 'native_language' in item:
                        item['label'] = item['native_language']
                    temp_dict[fp] = item
            features_dict = temp_dict
        
        labels_list = []
        for fp in filepaths:
            feature_item = None
            
            if fp in features_dict:
                feature_item = features_dict[fp]
            elif fp.replace('\\', '/') in features_dict:
                feature_item = features_dict[fp.replace('\\', '/')]
            
            if feature_item:
                self.features.append(feature_item['features'])
                label = feature_item.get('label') or feature_item.get('native_language')
                labels_list.append(label)
        
        if label_encoder is None:
            self.labels = self.label_encoder.fit_transform(labels_list)
        else:
            self.labels = self.label_encoder.transform(labels_list)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]


def extract_features(audio_paths, metadata_df, feature_type='hubert', layer=3):
    """Extract features from audio files"""
    print(f"\n{'='*80}")
    print(f"Extracting {feature_type.upper()} Features")
    print(f"{'='*80}\n")
    
    if feature_type == 'hubert':
        extractor = HuBERTExtractor()
        feature_dim = 768
    else:  # mfcc
        extractor = MFCCExtractor()
        feature_dim = 600
    
    features_dict = {}
    errors = 0
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting"):
        audio_path = Path('data/raw') / row['filepath']
        
        if not audio_path.exists():
            errors += 1
            continue
        
        try:
            if feature_type == 'hubert':
                result = extractor.extract_from_file(str(audio_path), extract_layer=layer, pooling='mean')
                features = result['embeddings'][f'layer_{layer}']
                
                # Convert to numpy if needed
                if hasattr(features, 'numpy'):
                    features = features.numpy()
            else:  # mfcc
                features = extractor.extract_from_file(str(audio_path))
            
            features_dict[row['filepath']] = {
                'features': features,
                'label': row['native_language'],
                'age_group': row.get('age_group', 'adult')
            }
        
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  Error: {audio_path}: {e}")
            continue
    
    print(f"\n✓ Extracted {len(features_dict)} samples (errors: {errors})")
    return features_dict, feature_dim


def train_model(train_features, train_metadata, feature_dim, num_classes, epochs=20):
    """Train classifier on adult data"""
    print(f"\n{'='*80}")
    print("Training on Adult Speech")
    print(f"{'='*80}\n")
    
    # Create dataset
    train_dataset = SimpleFeatureDataset(train_features, train_metadata['filepath'].tolist())
    label_encoder = train_dataset.label_encoder
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"✓ Training dataset: {len(train_dataset)} samples")
    print(f"  Classes: {list(label_encoder.classes_)}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = MLPClassifier(input_dim=feature_dim, hidden_dims=[256, 128], 
                         num_classes=num_classes, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1:02d}: train_acc={train_acc*100:.2f}%")
    
    return model, label_encoder, device


def evaluate_model(model, test_features, test_metadata, label_encoder, device, age_group='child'):
    """Evaluate model on test data"""
    print(f"\n{'='*80}")
    print(f"Evaluating on {age_group.capitalize()} Speech")
    print(f"{'='*80}\n")
    
    test_dataset = SimpleFeatureDataset(test_features, test_metadata['filepath'].tolist(), 
                                       label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\n{age_group.capitalize()} Speech Accuracy: {test_acc*100:.2f}%")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                               target_names=label_encoder.classes_, 
                               zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    return test_acc, test_preds, test_labels, cm


def plot_results(results, output_file='experiments/cross_age_comparison.png'):
    """Create comparison visualizations"""
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy comparison
    feature_types = ['MFCC', 'HuBERT']
    adult_accs = [results['mfcc']['adult_acc']*100, results['hubert']['adult_acc']*100]
    child_accs = [results['mfcc']['child_acc']*100, results['hubert']['child_acc']*100]
    gap = [adult_accs[i] - child_accs[i] for i in range(len(feature_types))]
    
    x = np.arange(len(feature_types))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, adult_accs, width, label='Adult (Train)', alpha=0.8, color='#2ecc71')
    axes[0, 0].bar(x + width/2, child_accs, width, label='Child (Test)', alpha=0.8, color='#e74c3c')
    
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Cross-Age Generalization: Adult → Child', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(feature_types)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i in x:
        axes[0, 0].text(i - width/2, adult_accs[i] + 1, f'{adult_accs[i]:.1f}%', 
                       ha='center', fontsize=10)
        axes[0, 0].text(i + width/2, child_accs[i] + 1, f'{child_accs[i]:.1f}%', 
                       ha='center', fontsize=10)
        axes[0, 0].text(i, max(adult_accs[i], child_accs[i]) + 5, f'Gap: {gap[i]:.1f}%', 
                       ha='center', fontsize=9, color='red', fontweight='bold')
    
    # Plot 2: Generalization gap
    axes[0, 1].bar(feature_types, gap, alpha=0.8, color='#3498db')
    axes[0, 1].set_ylabel('Accuracy Drop (%)', fontsize=12)
    axes[0, 1].set_title('Age Generalization Gap', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(gap):
        axes[0, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 3: MFCC confusion matrix
    cm_mfcc = results['mfcc']['confusion_matrix']
    sns.heatmap(cm_mfcc, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=results['mfcc']['classes'], 
                yticklabels=results['mfcc']['classes'])
    axes[1, 0].set_title(f'MFCC Confusion Matrix (Child Test: {results["mfcc"]["child_acc"]*100:.1f}%)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Plot 4: HuBERT confusion matrix
    cm_hubert = results['hubert']['confusion_matrix']
    sns.heatmap(cm_hubert, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                xticklabels=results['hubert']['classes'],
                yticklabels=results['hubert']['classes'])
    axes[1, 1].set_title(f'HuBERT Confusion Matrix (Child Test: {results["hubert"]["child_acc"]*100:.1f}%)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    Path('experiments').mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {output_file}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CROSS-AGE GENERALIZATION ANALYSIS: ADULT → CHILD")
    print("="*80 + "\n")
    
    # Load metadata
    print("Loading metadata...")
    adult_metadata = pd.read_csv('data/raw/indian_accents/metadata_with_splits.csv')
    adult_metadata['filepath'] = adult_metadata['filepath'].str.replace('\\', '/', regex=False)
    adult_metadata['age_group'] = 'adult'
    
    child_metadata = pd.read_csv('data/raw/child_audios/child_metadata.csv')
    child_metadata['filepath'] = child_metadata['filepath'].str.replace('\\', '/', regex=False)
    child_metadata['age_group'] = 'child'
    
    print(f"✓ Adult samples: {len(adult_metadata)}")
    print(f"✓ Child samples: {len(child_metadata)}")
    
    print("\nAdult distribution:")
    print(adult_metadata['native_language'].value_counts())
    
    print("\nChild distribution:")
    print(child_metadata['native_language'].value_counts())
    
    # Get number of classes
    num_classes = len(adult_metadata['native_language'].unique())
    
    results = {}
    
    # ========== EXPERIMENT 1: MFCC ==========
    print("\n" + "="*80)
    print("EXPERIMENT 1: MFCC Features")
    print("="*80)
    
    # Extract MFCC features
    adult_mfcc, mfcc_dim = extract_features(
        adult_metadata['filepath'].tolist(),
        adult_metadata,
        feature_type='mfcc'
    )
    
    child_mfcc, _ = extract_features(
        child_metadata['filepath'].tolist(),
        child_metadata,
        feature_type='mfcc'
    )
    
    # Train on adult
    mfcc_model, mfcc_encoder, device = train_model(
        adult_mfcc, adult_metadata, mfcc_dim, num_classes, epochs=20
    )
    
    # Evaluate on adult (sanity check)
    adult_acc_mfcc, _, _, _ = evaluate_model(
        mfcc_model, adult_mfcc, adult_metadata, mfcc_encoder, device, 'adult'
    )
    
    # Evaluate on child
    child_acc_mfcc, child_preds_mfcc, child_labels_mfcc, cm_mfcc = evaluate_model(
        mfcc_model, child_mfcc, child_metadata, mfcc_encoder, device, 'child'
    )
    
    results['mfcc'] = {
        'adult_acc': adult_acc_mfcc,
        'child_acc': child_acc_mfcc,
        'confusion_matrix': cm_mfcc,
        'classes': mfcc_encoder.classes_,
        'gap': adult_acc_mfcc - child_acc_mfcc
    }
    
    # Save MFCC model
    Path('experiments/cross_age_mfcc').mkdir(parents=True, exist_ok=True)
    torch.save(mfcc_model.state_dict(), 'experiments/cross_age_mfcc/model.pt')
    
    # ========== EXPERIMENT 2: HuBERT ==========
    print("\n" + "="*80)
    print("EXPERIMENT 2: HuBERT Layer 3 Features")
    print("="*80)
    
    # Extract HuBERT features
    adult_hubert, hubert_dim = extract_features(
        adult_metadata['filepath'].tolist(),
        adult_metadata,
        feature_type='hubert',
        layer=3
    )
    
    child_hubert, _ = extract_features(
        child_metadata['filepath'].tolist(),
        child_metadata,
        feature_type='hubert',
        layer=3
    )
    
    # Train on adult
    hubert_model, hubert_encoder, device = train_model(
        adult_hubert, adult_metadata, hubert_dim, num_classes, epochs=20
    )
    
    # Evaluate on adult (sanity check)
    adult_acc_hubert, _, _, _ = evaluate_model(
        hubert_model, adult_hubert, adult_metadata, hubert_encoder, device, 'adult'
    )
    
    # Evaluate on child
    child_acc_hubert, child_preds_hubert, child_labels_hubert, cm_hubert = evaluate_model(
        hubert_model, child_hubert, child_metadata, hubert_encoder, device, 'child'
    )
    
    results['hubert'] = {
        'adult_acc': adult_acc_hubert,
        'child_acc': child_acc_hubert,
        'confusion_matrix': cm_hubert,
        'classes': hubert_encoder.classes_,
        'gap': adult_acc_hubert - child_acc_hubert
    }
    
    # Save HuBERT model
    Path('experiments/cross_age_hubert').mkdir(parents=True, exist_ok=True)
    torch.save(hubert_model.state_dict(), 'experiments/cross_age_hubert/model.pt')
    
    # ========== COMPARISON ==========
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80 + "\n")
    
    print(f"{'Feature':<15} {'Adult Acc':<12} {'Child Acc':<12} {'Gap':<12} {'Winner'}")
    print("-" * 65)
    print(f"{'MFCC':<15} {results['mfcc']['adult_acc']*100:>9.2f}% "
          f"{results['mfcc']['child_acc']*100:>10.2f}% {results['mfcc']['gap']*100:>10.2f}%")
    print(f"{'HuBERT Layer 3':<15} {results['hubert']['adult_acc']*100:>9.2f}% "
          f"{results['hubert']['child_acc']*100:>10.2f}% {results['hubert']['gap']*100:>10.2f}%")
    
    # Determine winner
    if results['hubert']['child_acc'] > results['mfcc']['child_acc']:
        winner = 'HuBERT'
        diff = (results['hubert']['child_acc'] - results['mfcc']['child_acc']) * 100
    else:
        winner = 'MFCC'
        diff = (results['mfcc']['child_acc'] - results['hubert']['child_acc']) * 100
    
    print(f"\n✓ {winner} generalizes better to children by {diff:.2f}%")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if results['mfcc']['gap'] < results['hubert']['gap']:
        print("\n✓ MFCC shows better age robustness")
        print("  → Handcrafted features may capture age-invariant accent characteristics")
        print("  → Statistical summaries are less sensitive to pitch/formant shifts")
    else:
        print("\n✓ HuBERT shows better age robustness")
        print("  → Learned representations adapt better to age-related variations")
        print("  → Deep features capture accent patterns beyond acoustic changes")
    
    print(f"\nGeneralization gaps:")
    print(f"  MFCC: {results['mfcc']['gap']*100:.2f}%")
    print(f"  HuBERT: {results['hubert']['gap']*100:.2f}%")
    
    # Create visualizations
    plot_results(results)
    
    # Save results
    results_file = 'experiments/cross_age_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'mfcc': {
                'adult_acc': float(results['mfcc']['adult_acc']),
                'child_acc': float(results['mfcc']['child_acc']),
                'gap': float(results['mfcc']['gap'])
            },
            'hubert': {
                'adult_acc': float(results['hubert']['adult_acc']),
                'child_acc': float(results['hubert']['child_acc']),
                'gap': float(results['hubert']['gap'])
            },
            'winner': winner,
            'difference': float(diff)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    print("\n" + "="*80)
    print("CROSS-AGE GENERALIZATION ANALYSIS COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
