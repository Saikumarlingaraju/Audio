"""
Cross-Age Generalization Analysis - Standalone Version for Kaggle
Train on adult speech, test on children's speech
Compare MFCC vs HuBERT robustness across age groups
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, HubertModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Feature Extractors
# ============================================================================

class MFCCExtractor:
    """Extract MFCC features from audio"""
    
    def __init__(self, sr=16000, n_mfcc=40):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract(self, audio_path):
        """Extract MFCC features with delta and delta-delta"""
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate and compute statistics
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Mean and std across time
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        
        # Concatenate mean and std
        final_features = np.concatenate([mean, std])
        
        return final_features


class HuBERTExtractor:
    """Extract HuBERT features from audio"""
    
    def __init__(self, layer=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.layer = layer
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
        self.model.eval()
    
    def extract(self, audio_path):
        """Extract HuBERT features from specific layer"""
        audio, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer]
            features = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        
        return features


# ============================================================================
# Dataset
# ============================================================================

class AccentDataset(Dataset):
    """Dataset for accent classification"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# Model
# ============================================================================

class MLPClassifier(nn.Module):
    """Multi-layer perceptron for classification"""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, device='cuda'):
    """Train the model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_acc


def evaluate_model(model, data_loader, device='cuda'):
    """Evaluate model on a dataset"""
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, all_preds, all_labels, cm


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("CROSS-AGE GENERALIZATION ANALYSIS")
    print("Train on Adult Speech → Test on Children's Speech")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # State mapping
    state_to_idx = {
        'andhra_pradesh': 0,
        'gujrat': 1,
        'jharkhand': 2,
        'karnataka': 3,
        'kerala': 4,
        'tamil': 5
    }
    idx_to_state = {v: k for k, v in state_to_idx.items()}
    
    # Create output directory
    output_dir = Path('experiments/cross_age_generalization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Load Metadata
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING METADATA")
    print("="*80)
    
    # Adult data
    train_df = pd.read_csv('data/splits/train.csv')
    val_df = pd.read_csv('data/splits/val.csv')
    test_df = pd.read_csv('data/splits/test.csv')
    
    # Normalize paths
    train_df['filepath'] = train_df['filepath'].str.replace('\\', '/', regex=False)
    val_df['filepath'] = val_df['filepath'].str.replace('\\', '/', regex=False)
    test_df['filepath'] = test_df['filepath'].str.replace('\\', '/', regex=False)
    
    print(f"Adult data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Child data
    child_df = pd.read_csv('data/raw/child_audios/child_metadata.csv')
    child_df['filepath'] = child_df['filepath'].str.replace('\\', '/', regex=False)
    print(f"Child data: {len(child_df)} samples")
    print(f"Child states: {child_df['native_language'].value_counts().to_dict()}")
    
    # ========================================================================
    # Extract Features - MFCC
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXTRACTING MFCC FEATURES")
    print("="*80)
    
    mfcc_extractor = MFCCExtractor()
    
    def extract_mfcc_features(df, desc):
        features = []
        labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            try:
                audio_path = f"data/raw/{row['filepath']}"
                feat = mfcc_extractor.extract(audio_path)
                features.append(feat)
                labels.append(state_to_idx[row['native_language']])
            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")
        return np.array(features), np.array(labels)
    
    print("Extracting adult MFCC features...")
    mfcc_train_X, mfcc_train_y = extract_mfcc_features(train_df, "Train")
    mfcc_val_X, mfcc_val_y = extract_mfcc_features(val_df, "Val")
    mfcc_test_X, mfcc_test_y = extract_mfcc_features(test_df, "Test")
    
    print("Extracting child MFCC features...")
    mfcc_child_X, mfcc_child_y = extract_mfcc_features(child_df, "Child")
    
    print(f"MFCC feature shape: {mfcc_train_X.shape[1]}")
    
    # ========================================================================
    # Extract Features - HuBERT Layer 3
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXTRACTING HUBERT FEATURES (Layer 3)")
    print("="*80)
    
    hubert_extractor = HuBERTExtractor(layer=3, device=device)
    
    def extract_hubert_features(df, desc):
        features = []
        labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            try:
                audio_path = f"data/raw/{row['filepath']}"
                feat = hubert_extractor.extract(audio_path)
                features.append(feat)
                labels.append(state_to_idx[row['native_language']])
            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")
        return np.array(features), np.array(labels)
    
    print("Extracting adult HuBERT features...")
    hubert_train_X, hubert_train_y = extract_hubert_features(train_df, "Train")
    hubert_val_X, hubert_val_y = extract_hubert_features(val_df, "Val")
    hubert_test_X, hubert_test_y = extract_hubert_features(test_df, "Test")
    
    print("Extracting child HuBERT features...")
    hubert_child_X, hubert_child_y = extract_hubert_features(child_df, "Child")
    
    print(f"HuBERT feature shape: {hubert_train_X.shape[1]}")
    
    # ========================================================================
    # Train MFCC Model
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRAINING MFCC MODEL (on Adult Data)")
    print("="*80)
    
    mfcc_train_loader = DataLoader(AccentDataset(mfcc_train_X, mfcc_train_y), batch_size=32, shuffle=True)
    mfcc_val_loader = DataLoader(AccentDataset(mfcc_val_X, mfcc_val_y), batch_size=32)
    mfcc_test_loader = DataLoader(AccentDataset(mfcc_test_X, mfcc_test_y), batch_size=32)
    mfcc_child_loader = DataLoader(AccentDataset(mfcc_child_X, mfcc_child_y), batch_size=32)
    
    mfcc_model = MLPClassifier(input_dim=mfcc_train_X.shape[1], num_classes=6).to(device)
    mfcc_model, mfcc_val_acc = train_model(mfcc_model, mfcc_train_loader, mfcc_val_loader, device=device)
    
    print(f"\nMFCC Best Val Accuracy: {mfcc_val_acc:.4f}")
    
    # Evaluate on adult test
    mfcc_adult_acc, mfcc_adult_preds, mfcc_adult_labels, mfcc_adult_cm = evaluate_model(
        mfcc_model, mfcc_test_loader, device
    )
    print(f"MFCC Adult Test Accuracy: {mfcc_adult_acc:.4f}")
    
    # Evaluate on child test
    mfcc_child_acc, mfcc_child_preds, mfcc_child_labels, mfcc_child_cm = evaluate_model(
        mfcc_model, mfcc_child_loader, device
    )
    print(f"MFCC Child Test Accuracy: {mfcc_child_acc:.4f}")
    
    mfcc_gap = mfcc_adult_acc - mfcc_child_acc
    print(f"MFCC Generalization Gap: {mfcc_gap:.4f}")
    
    # Save model
    torch.save(mfcc_model.state_dict(), output_dir / 'mfcc_adult_model.pt')
    
    # ========================================================================
    # Train HuBERT Model
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRAINING HUBERT MODEL (on Adult Data)")
    print("="*80)
    
    hubert_train_loader = DataLoader(AccentDataset(hubert_train_X, hubert_train_y), batch_size=32, shuffle=True)
    hubert_val_loader = DataLoader(AccentDataset(hubert_val_X, hubert_val_y), batch_size=32)
    hubert_test_loader = DataLoader(AccentDataset(hubert_test_X, hubert_test_y), batch_size=32)
    hubert_child_loader = DataLoader(AccentDataset(hubert_child_X, hubert_child_y), batch_size=32)
    
    hubert_model = MLPClassifier(input_dim=hubert_train_X.shape[1], num_classes=6).to(device)
    hubert_model, hubert_val_acc = train_model(hubert_model, hubert_train_loader, hubert_val_loader, device=device)
    
    print(f"\nHuBERT Best Val Accuracy: {hubert_val_acc:.4f}")
    
    # Evaluate on adult test
    hubert_adult_acc, hubert_adult_preds, hubert_adult_labels, hubert_adult_cm = evaluate_model(
        hubert_model, hubert_test_loader, device
    )
    print(f"HuBERT Adult Test Accuracy: {hubert_adult_acc:.4f}")
    
    # Evaluate on child test
    hubert_child_acc, hubert_child_preds, hubert_child_labels, hubert_child_cm = evaluate_model(
        hubert_model, hubert_child_loader, device
    )
    print(f"HuBERT Child Test Accuracy: {hubert_child_acc:.4f}")
    
    hubert_gap = hubert_adult_acc - hubert_child_acc
    print(f"HuBERT Generalization Gap: {hubert_gap:.4f}")
    
    # Save model
    torch.save(hubert_model.state_dict(), output_dir / 'hubert_adult_model.pt')
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy Comparison
    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.35
    
    adult_accs = [mfcc_adult_acc * 100, hubert_adult_acc * 100]
    child_accs = [mfcc_child_acc * 100, hubert_child_acc * 100]
    
    ax.bar(x - width/2, adult_accs, width, label='Adult Test', color='skyblue')
    ax.bar(x + width/2, child_accs, width, label='Child Test', color='salmon')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Age Generalization: Adult vs Child Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['MFCC', 'HuBERT (Layer 3)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (adult, child) in enumerate(zip(adult_accs, child_accs)):
        ax.text(i - width/2, adult + 1, f'{adult:.1f}%', ha='center', fontsize=10)
        ax.text(i + width/2, child + 1, f'{child:.1f}%', ha='center', fontsize=10)
    
    # Plot 2: Generalization Gap
    ax = axes[0, 1]
    gaps = [mfcc_gap * 100, hubert_gap * 100]
    colors = ['green' if g < 5 else 'orange' if g < 10 else 'red' for g in gaps]
    
    bars = ax.bar(['MFCC', 'HuBERT (Layer 3)'], gaps, color=colors, alpha=0.7)
    ax.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax.set_title('Generalization Gap (Adult Acc - Child Acc)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{gap:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: MFCC Confusion Matrix (Child)
    ax = axes[1, 0]
    sns.heatmap(mfcc_child_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[idx_to_state[i] for i in range(6)],
                yticklabels=[idx_to_state[i] for i in range(6)])
    ax.set_title('MFCC: Child Test Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Plot 4: HuBERT Confusion Matrix (Child)
    ax = axes[1, 1]
    sns.heatmap(hubert_child_cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=[idx_to_state[i] for i in range(6)],
                yticklabels=[idx_to_state[i] for i in range(6)])
    ax.set_title('HuBERT: Child Test Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_age_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'cross_age_comparison.png'}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    results = {
        'mfcc': {
            'adult_test_accuracy': float(mfcc_adult_acc),
            'child_test_accuracy': float(mfcc_child_acc),
            'generalization_gap': float(mfcc_gap),
            'validation_accuracy': float(mfcc_val_acc)
        },
        'hubert_layer3': {
            'adult_test_accuracy': float(hubert_adult_acc),
            'child_test_accuracy': float(hubert_child_acc),
            'generalization_gap': float(hubert_gap),
            'validation_accuracy': float(hubert_val_acc)
        },
        'winner': 'MFCC' if mfcc_gap < hubert_gap else 'HuBERT Layer 3',
        'interpretation': {
            'mfcc_gap': 'Small' if mfcc_gap < 0.05 else 'Medium' if mfcc_gap < 0.10 else 'Large',
            'hubert_gap': 'Small' if hubert_gap < 0.05 else 'Medium' if hubert_gap < 0.10 else 'Large'
        }
    }
    
    with open(output_dir / 'cross_age_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results: {output_dir / 'cross_age_results.json'}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nMFCC:")
    print(f"  Adult Test Acc:  {mfcc_adult_acc*100:.2f}%")
    print(f"  Child Test Acc:  {mfcc_child_acc*100:.2f}%")
    print(f"  Gap:             {mfcc_gap*100:.2f}%")
    
    print(f"\nHuBERT Layer 3:")
    print(f"  Adult Test Acc:  {hubert_adult_acc*100:.2f}%")
    print(f"  Child Test Acc:  {hubert_child_acc*100:.2f}%")
    print(f"  Gap:             {hubert_gap*100:.2f}%")
    
    print(f"\n🏆 Winner (Better Generalization): {results['winner']}")
    print("="*80)


if __name__ == "__main__":
    main()
