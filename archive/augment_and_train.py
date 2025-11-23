"""
Train model with data augmentation to improve generalization
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

from src.models.mlp import MLPClassifier
from src.features.mfcc_extractor import MFCCExtractor

# Data augmentation functions
def add_noise(audio, noise_factor=0.005):
    """Add random noise"""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented.astype(audio.dtype)

def shift_pitch(audio, sr, n_steps=2):
    """Shift pitch by n semitones"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=1.0):
    """Time stretch audio"""
    return librosa.effects.time_stretch(audio, rate=rate)

def change_volume(audio, factor=1.0):
    """Change volume"""
    return audio * factor

def apply_random_augmentation(audio, sr):
    """Apply random augmentation"""
    aug_type = np.random.choice(['noise', 'pitch', 'stretch', 'volume', 'none'], p=[0.25, 0.20, 0.20, 0.25, 0.10])
    
    if aug_type == 'noise':
        return add_noise(audio, noise_factor=np.random.uniform(0.002, 0.008))
    elif aug_type == 'pitch':
        n_steps = np.random.choice([-2, -1, 1, 2])
        return shift_pitch(audio, sr, n_steps)
    elif aug_type == 'stretch':
        rate = np.random.uniform(0.9, 1.1)
        return time_stretch(audio, rate)
    elif aug_type == 'volume':
        factor = np.random.uniform(0.7, 1.3)
        return change_volume(audio, factor)
    else:
        return audio

class AugmentedAudioDataset(Dataset):
    """Dataset with on-the-fly augmentation"""
    
    def __init__(self, metadata_df, feature_extractor, augment=False):
        self.metadata = metadata_df
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.metadata = self.metadata.reset_index(drop=True)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load audio (ensure absolute path)
        filepath = row['filepath']
        if not Path(filepath).is_absolute():
            filepath = Path('data/raw/indian_accents') / filepath
        audio, sr = librosa.load(str(filepath), sr=16000)
        
        # Apply augmentation for training
        if self.augment and np.random.rand() > 0.3:  # 70% chance of augmentation
            audio = apply_random_augmentation(audio, sr)
        
        # Extract features using MFCC extractor methods
        mfcc_features = self.feature_extractor.extract_mfcc(audio)
        features = self.feature_extractor.compute_statistics(mfcc_features)
        
        return torch.FloatTensor(features), torch.LongTensor([row['label']])[0]

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc='Validation'):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    print("="*80)
    print("🚀 TRAINING WITH DATA AUGMENTATION")
    print("="*80)
    
    # Setup
    device = torch.device('cpu')
    
    # Load metadata
    metadata_path = Path('data/raw/indian_accents/metadata_with_splits.csv')
    metadata = pd.read_csv(metadata_path)
    
    print(f"\n✓ Loaded {len(metadata)} samples")
    print(f"✓ Classes: {metadata['native_language'].unique()}")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    metadata['label'] = label_encoder.fit_transform(metadata['native_language'])
    
    # Split data
    train_df = metadata[metadata['split'] == 'train'].reset_index(drop=True)
    val_df = metadata[metadata['split'] == 'val'].reset_index(drop=True)
    test_df = metadata[metadata['split'] == 'test'].reset_index(drop=True)
    
    print(f"\n✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize feature extractor
    feature_extractor = MFCCExtractor(
        sr=16000,
        n_mfcc=40,
        include_delta=True,
        include_delta_delta=True
    )
    
    # Create datasets with augmentation
    print("\n✓ Creating datasets with augmentation...")
    train_dataset = AugmentedAudioDataset(train_df, feature_extractor, augment=True)
    val_dataset = AugmentedAudioDataset(val_df, feature_extractor, augment=False)
    test_dataset = AugmentedAudioDataset(test_df, feature_extractor, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model with stronger regularization
    num_classes = len(label_encoder.classes_)
    model = MLPClassifier(
        input_dim=600,
        hidden_dims=[256, 128],
        num_classes=num_classes,
        dropout=0.5  # Increased dropout
    ).to(device)
    
    print(f"\n✓ Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR, added L2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "="*80)
    print("🏋️ TRAINING START")
    print("="*80)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(50):
        print(f"\n📍 Epoch {epoch+1}/50")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_dir = Path('experiments/indian_accents_mlp_augmented')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }
            
            torch.save(checkpoint, save_dir / 'best_model.pt')
            with open(save_dir / 'label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
            
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n⏹️ Early stopping after {epoch+1} epochs")
                break
    
    # Test evaluation
    print("\n" + "="*80)
    print("🧪 FINAL TEST EVALUATION")
    print("="*80)
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    
    # Save final test accuracy
    checkpoint['test_acc'] = test_acc
    torch.save(checkpoint, save_dir / 'best_model.pt')
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"📊 Best Val Acc: {best_val_acc:.2f}%")
    print(f"📊 Final Test Acc: {test_acc:.2f}%")
    print(f"💾 Model saved to: {save_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
