"""
Fast training with pre-extracted features + simple augmentation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from src.models.mlp import MLPClassifier

# Simple feature-level augmentation
def add_noise_to_features(features, noise_factor=0.1):
    """Add Gaussian noise to features"""
    noise = np.random.randn(*features.shape) * noise_factor
    return features + noise

def scale_features(features, scale_factor=1.0):
    """Scale features"""
    return features * scale_factor

def apply_feature_augmentation(features):
    """Apply random augmentation to extracted features"""
    aug_type = np.random.choice(['noise', 'scale', 'none'], p=[0.4, 0.4, 0.2])
    
    if aug_type == 'noise':
        return add_noise_to_features(features, noise_factor=np.random.uniform(0.05, 0.15))
    elif aug_type == 'scale':
        return scale_features(features, scale_factor=np.random.uniform(0.8, 1.2))
    else:
        return features

class FastAugmentedDataset(Dataset):
    """Fast dataset using pre-extracted features with augmentation"""
    
    def __init__(self, features_list, label_encoder, augment=False):
        self.features_list = features_list
        self.label_encoder = label_encoder
        self.augment = augment
        
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        item = self.features_list[idx]
        features = item['features'].copy()  # Copy to avoid modifying original
        
        # Apply augmentation
        if self.augment and np.random.rand() > 0.3:  # 70% chance
            features = apply_feature_augmentation(features)
        
        # Encode label
        label = self.label_encoder.transform([item['native_language']])[0]
        
        return torch.FloatTensor(features), torch.LongTensor([label])[0]

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
    print("🚀 FAST TRAINING WITH FEATURE AUGMENTATION")
    print("="*80)
    
    # Setup
    device = torch.device('cpu')
    
    # Load pre-extracted features
    features_path = Path('data/features/indian_accents/mfcc_stats.pkl')
    print(f"\n✓ Loading features from: {features_path}")
    
    with open(features_path, 'rb') as f:
        features_list = pickle.load(f)
    
    print(f"✓ Loaded {len(features_list)} samples")
    
    # Create label encoder
    labels = [item['native_language'] for item in features_list]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    print(f"✓ Classes: {label_encoder.classes_}")
    
    # Split by 'split' field
    train_features = [item for item in features_list if item['split'] == 'train']
    val_features = [item for item in features_list if item['split'] == 'val']
    test_features = [item for item in features_list if item['split'] == 'test']
    
    print(f"\n✓ Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
    
    # Create datasets
    print("\n✓ Creating datasets with feature augmentation...")
    train_dataset = FastAugmentedDataset(train_features, label_encoder, augment=True)
    val_dataset = FastAugmentedDataset(val_features, label_encoder, augment=False)
    test_dataset = FastAugmentedDataset(test_features, label_encoder, augment=False)
    
    # Create dataloaders with larger batch size for speed
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Initialize model with stronger regularization
    num_classes = len(label_encoder.classes_)
    model = MLPClassifier(
        input_dim=600,
        hidden_dims=[256, 128],
        num_classes=num_classes,
        dropout=0.5  # Increased dropout
    ).to(device)
    
    print(f"\n✓ Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ Dropout: 0.5 (stronger regularization)")
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
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
    
    # Load best model for testing
    print("\n" + "="*80)
    print("🧪 LOADING BEST MODEL FOR FINAL TEST")
    print("="*80)
    checkpoint = torch.load(save_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
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
    print("\n💡 Next step: Run 'python app\\app_robust.py' to use the improved model!")
    print("="*80)

if __name__ == "__main__":
    main()
