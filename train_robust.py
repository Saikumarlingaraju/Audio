"""
Robust training script for accent classification with improved generalization.
Incorporates data augmentation, robust models, and advanced training techniques.
"""

import argparse
import pickle
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.features.dataset import load_features_and_create_dataset
from src.features.augmentation import create_augmented_dataset
from src.features.robust_extraction import FeatureStatisticsComputer, create_robust_feature_extractor
from src.models.robust_mlp import RobustMLPClassifier, AdaptiveMLPClassifier, create_robust_model
from src.utils.metrics import compute_metrics


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def compute_feature_statistics(features_path, splits_dir, output_path):
    """Compute and save feature statistics for normalization."""
    print("Computing feature statistics for normalization...")
    
    # Create feature extractor
    extractor = create_robust_feature_extractor(normalize=False)
    
    # Initialize statistics computer
    stats_computer = FeatureStatisticsComputer(extractor)
    
    # Load training data paths
    train_csv = splits_dir / 'train.csv'
    if not train_csv.exists():
        print("Warning: No train.csv found, using all data for statistics")
        return None
    
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    
    # Add features from training set only
    print("Extracting features for statistics computation...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        audio_path = Path('data/raw/indian_accents') / row['filepath']
        if audio_path.exists():
            stats_computer.add_features_from_file(str(audio_path))
    
    # Compute and save statistics
    stats = stats_computer.compute_and_save_statistics(str(output_path))
    return stats


def train_epoch_robust(model, loader, criterion, optimizer, device, epoch):
    """Enhanced training epoch with progress tracking."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (features, labels, _) in enumerate(pbar):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Handle different model types
        if hasattr(model, 'forward') and 'return_individual_heads' in model.forward.__code__.co_varnames:
            # Adaptive MLP
            outputs = model(features)
        elif hasattr(model, 'forward') and 'return_uncertainty' in model.forward.__code__.co_varnames:
            # Uncertainty-aware MLP
            outputs = model(features)
        else:
            # Standard MLP
            outputs = model(features)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def evaluate_robust(model, loader, criterion, device):
    """Enhanced evaluation with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    uncertainties = []
    
    with torch.no_grad():
        for features, labels, _ in loader:
            features, labels = features.to(device), labels.to(device)
            
            # Handle different model types
            if hasattr(model, 'forward') and 'return_uncertainty' in model.forward.__code__.co_varnames:
                try:
                    outputs, uncertainty = model(features, return_uncertainty=True)
                    uncertainties.extend(uncertainty.cpu().numpy())
                except:
                    outputs = model(features)
            else:
                outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.0
    
    return total_loss / len(loader), accuracy, all_preds, all_labels, avg_uncertainty


def plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Robust Accent Classification Training')
    
    # Data arguments
    parser.add_argument('--features_path', type=str, 
                       default='data/features/indian_accents/hubert_layer12_mean.pkl',
                       help='Path to features file')
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/splits/val.csv')
    parser.add_argument('--test_csv', type=str, default='data/splits/test.csv')
    parser.add_argument('--output_dir', type=str, default='experiments/indian_accents_robust')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='robust_mlp',
                       choices=['robust_mlp', 'adaptive_mlp', 'uncertainty_mlp'],
                       help='Type of robust model to use')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_batch_norm', action='store_true', default=True)
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'gelu', 'swish', 'leaky_relu'])
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    
    # Augmentation arguments
    parser.add_argument('--use_augmentation', action='store_true', default=True)
    parser.add_argument('--aug_prob', type=float, default=0.5)
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compute_stats', action='store_true', 
                       help='Compute feature statistics for normalization')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 ROBUST ACCENT CLASSIFICATION TRAINING")
    print("=" * 80)
    print(f"Features: {args.features_path}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Augmentation: {args.use_augmentation} (prob={args.aug_prob})")
    print("=" * 80)
    
    # Compute feature statistics if requested
    if args.compute_stats:
        stats_path = output_dir / 'feature_stats.pt'
        compute_feature_statistics(
            args.features_path, 
            Path(args.train_csv).parent,
            stats_path
        )
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, label_encoder = load_features_and_create_dataset(
        args.features_path, args.train_csv, label_encoder=None
    )
    val_dataset, _ = load_features_and_create_dataset(
        args.features_path, args.val_csv, label_encoder=label_encoder
    )
    test_dataset, _ = load_features_and_create_dataset(
        args.features_path, args.test_csv, label_encoder=label_encoder
    )
    
    # Apply augmentation to training dataset
    if args.use_augmentation:
        print("Applying data augmentation to training set...")
        train_dataset = create_augmented_dataset(
            train_dataset, 
            augment_prob=args.aug_prob
        )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # Get model dimensions
    input_dim = train_dataset[0][0].shape[0] if hasattr(train_dataset, '__getitem__') else train_dataset.base_dataset[0][0].shape[0]
    num_classes = len(label_encoder.classes_)
    
    print(f"\n✓ Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Classes ({num_classes}): {list(label_encoder.classes_)}")
    print(f"  Input dim: {input_dim}")
    
    # Create robust model
    model = create_robust_model(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type=args.model_type,
        hidden_dims=args.hidden_sizes,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation
    ).to(args.device)
    
    print(f"\n✓ Model created: {args.model_type}")
    print(f"  Architecture: {args.hidden_sizes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for robustness
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, restore_best_weights=True)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print("\n" + "=" * 80)
    print("🏋️ TRAINING")
    print("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch_robust(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        
        # Validate
        val_loss, val_acc, _, _, val_uncertainty = evaluate_robust(
            model, val_loader, criterion, args.device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_uncertainty > 0:
            print(f"         | Val Uncertainty: {val_uncertainty:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'model_type': args.model_type,
                'input_dim': input_dim,
                'num_classes': num_classes,
                'hidden_dims': args.hidden_sizes,
                'dropout': args.dropout,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
        
        # Early stopping
        if early_stopping(val_acc, model):
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
    
    print("\n" + "=" * 80)
    print(f"✅ TRAINING COMPLETE!")
    print(f"📊 Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Load best model for testing
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    print("\n" + "=" * 80)
    print("🧪 TESTING")
    print("=" * 80)
    
    test_loss, test_acc, test_preds, test_labels, test_uncertainty = evaluate_robust(
        model, test_loader, criterion, args.device
    )
    
    print(f"✓ Test Accuracy: {test_acc:.2f}%")
    print(f"✓ Test Loss: {test_loss:.4f}")
    if test_uncertainty > 0:
        print(f"✓ Test Uncertainty: {test_uncertainty:.4f}")
    
    # Update checkpoint with test results
    checkpoint['test_acc'] = test_acc
    checkpoint['test_loss'] = test_loss
    torch.save(checkpoint, output_dir / 'best_model.pt')
    
    # Detailed evaluation
    print("\n" + "=" * 80)
    print("📈 DETAILED EVALUATION")
    print("=" * 80)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                               target_names=label_encoder.classes_, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, label_encoder.classes_, output_dir)
    print(f"✓ Confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")
    
    # Training history plot
    plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir)
    print(f"✓ Training history saved to: {output_dir / 'training_history.png'}")
    
    # Save label encoder
    with open(output_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved to: {output_dir / 'label_encoder.pkl'}")
    
    # Save training config
    config = {
        'args': vars(args),
        'model_architecture': str(model),
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }
    
    import json
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"✓ Training config saved to: {output_dir / 'training_config.json'}")
    
    print("\n" + "=" * 80)
    print("🎉 ROBUST TRAINING COMPLETE!")
    print("=" * 80)
    print("\n📋 Summary:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Final Test Accuracy: {test_acc:.2f}%")
    print(f"  Generalization Gap: {best_val_acc - test_acc:.2f}%")
    
    if best_val_acc - test_acc < 5:
        print("  ✅ Good generalization!")
    elif best_val_acc - test_acc < 10:
        print("  ⚠️ Moderate overfitting")
    else:
        print("  ❌ High overfitting - consider more regularization")
    
    print(f"\n💾 Model and artifacts saved to: {output_dir}")
    print("\n🚀 Ready for deployment with robust predictions!")


if __name__ == '__main__':
    main()