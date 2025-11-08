"""
Train model WITHOUT feature augmentation for consistency
This ensures training and inference use identical feature extraction
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
from sklearn.metrics import accuracy_score, confusion_matrix

from src.models.mlp import MLPClassifier

class SimpleDataset(Dataset):
    """Dataset WITHOUT augmentation for consistency"""
    
    def __init__(self, features_list, label_encoder):
        self.features_list = features_list
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        item = self.features_list[idx]
        features = item['features']
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
    all_preds = []
    all_labels = []
    
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
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), 100. * correct / total, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Save confusion matrix as text (matplotlib removed to avoid import issues)"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Save as text file
    with open(save_path.replace('.png', '.txt'), 'w') as f:
        f.write("Confusion Matrix\n")
        f.write("="*80 + "\n\n")
        f.write("True\\Pred  " + "  ".join([f"{c:>10}" for c in classes]) + "\n")
        f.write("-"*80 + "\n")
        for i, row in enumerate(cm):
            f.write(f"{classes[i]:>10}  " + "  ".join([f"{val:>10}" for val in row]) + "\n")
        f.write("\n" + "="*80 + "\n")
        
        # Per-class accuracy
        f.write("\nPer-class Accuracy:\n")
        for i, cls in enumerate(classes):
            acc = 100 * cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            f.write(f"  {cls}: {acc:.1f}%\n")
    
    print(f"✓ Confusion matrix saved: {save_path.replace('.png', '.txt')}")

def main():
    print("="*80)
    print("🚀 TRAINING WITHOUT AUGMENTATION (Clean Model)")
    print("="*80)
    print("This model uses the SAME feature extraction as inference")
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
    
    # Create datasets WITHOUT augmentation
    print("\n✓ Creating clean datasets (no augmentation)...")
    train_dataset = SimpleDataset(train_features, label_encoder)
    val_dataset = SimpleDataset(val_features, label_encoder)
    test_dataset = SimpleDataset(test_features, label_encoder)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model with moderate regularization
    num_classes = len(label_encoder.classes_)
    model = MLPClassifier(
        input_dim=600,
        hidden_dims=[256, 128],
        num_classes=num_classes,
        dropout=0.3  # Moderate dropout
    ).to(device)
    
    print(f"\n✓ Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ Dropout: 0.3 (moderate regularization)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "="*80)
    print("🏋️ TRAINING START")
    print("="*80)
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(30):
        print(f"\n📍 Epoch {epoch+1}/30")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_dir = Path('experiments/indian_accents_mlp_clean')
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
    
    # Test evaluation with confusion matrix
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, 
        test_preds, 
        [c.replace('_', ' ').title() for c in label_encoder.classes_],
        save_dir / 'confusion_matrix.png'
    )
    
    # Save final test accuracy
    checkpoint['test_acc'] = test_acc
    torch.save(checkpoint, save_dir / 'best_model.pt')
    
    # Save training log
    with open(save_dir / 'training_log.txt', 'w') as f:
        f.write(f"Clean Model Training (No Augmentation)\n")
        f.write(f"="*80 + "\n")
        f.write(f"Best Val Acc: {best_val_acc:.2f}%\n")
        f.write(f"Final Test Acc: {test_acc:.2f}%\n")
        f.write(f"Classes: {', '.join(label_encoder.classes_)}\n")
        f.write(f"Dropout: 0.3\n")
        f.write(f"Feature augmentation: None\n")
        f.write(f"Inference compatible: Yes\n")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"📊 Best Val Acc: {best_val_acc:.2f}%")
    print(f"📊 Final Test Acc: {test_acc:.2f}%")
    print(f"💾 Model saved to: {save_dir}")
    print(f"\n💡 This model should work consistently on both training and external audio!")
    print(f"💡 Update app_robust.py to use: experiments/indian_accents_mlp_clean")
    print("="*80)

if __name__ == "__main__":
    main()
