"""
Simple training script for Accent DB Extended dataset
Uses pre-extracted MFCC features and metadata splits
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.dataset import load_features_and_create_dataset
from src.models.mlp import MLPClassifier


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels, _ in loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, _ in loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    # Default now points to HuBERT pooled features (last layer mean); MFCC only for comparison
    parser.add_argument('--features_path', type=str, default='data/features/indian_accents/hubert_layer12_mean.pkl')
    parser.add_argument('--mfcc_compare_path', type=str, default='data/features/indian_accents/mfcc_stats.pkl', help='Optional MFCC stats file for side-by-side evaluation')
    parser.add_argument('--train_csv', type=str, default='data/splits/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/splits/val.csv')
    parser.add_argument('--test_csv', type=str, default='data/splits/test.csv')
    parser.add_argument('--output_dir', type=str, default='experiments/indian_accents_hubert')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ACCENT CLASSIFICATION TRAINING")
    print("=" * 80)
    print(f"Features: {args.features_path}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets (HuBERT primary)...")
    train_dataset, label_encoder = load_features_and_create_dataset(
        args.features_path, args.train_csv, label_encoder=None
    )
    val_dataset, _ = load_features_and_create_dataset(
        args.features_path, args.val_csv, label_encoder=label_encoder
    )
    test_dataset, _ = load_features_and_create_dataset(
        args.features_path, args.test_csv, label_encoder=label_encoder
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get input dimension from first sample
    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(label_encoder.classes_)
    
    print(f"\n✓ Feature type: HuBERT pooled layer (mean) -> {Path(args.features_path).name}")
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    print(f"✓ Classes ({num_classes}): {list(label_encoder.classes_)}")
    print(f"✓ Input dim: {input_dim}")
    
    # Create model
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_sizes,
        num_classes=num_classes,
        dropout=0.3
    ).to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    print(f"\n✓ Model: MLP{args.hidden_sizes} (HuBERT primary)")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, args.device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:3d}/{args.num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / 'best_model.pt')
    
    print("\n" + "=" * 80)
    print(f"✓ Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, args.device)
    
    print(f"✓ Test Accuracy (HuBERT): {test_acc:.2f}%")
    print("=" * 80)
    
    # Classification report
    print("\nCLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_, digits=3))

    # Optional MFCC comparison if file exists
    if args.mfcc_compare_path and Path(args.mfcc_compare_path).exists():
        print("\n" + "=" * 80)
        print("MFCC COMPARISON RUN (optional)")
        print("=" * 80)
        mfcc_train, _ = load_features_and_create_dataset(args.mfcc_compare_path, args.train_csv, label_encoder=label_encoder)
        mfcc_val, _ = load_features_and_create_dataset(args.mfcc_compare_path, args.val_csv, label_encoder=label_encoder)
        mfcc_test, _ = load_features_and_create_dataset(args.mfcc_compare_path, args.test_csv, label_encoder=label_encoder)
        mfcc_train_loader = DataLoader(mfcc_train, batch_size=args.batch_size, shuffle=True)
        mfcc_test_loader = DataLoader(mfcc_test, batch_size=args.batch_size, shuffle=False)
        # New model for MFCC (dimension adapts automatically)
        mfcc_input_dim = mfcc_train[0][0].shape[0]
        mfcc_model = MLPClassifier(input_dim=mfcc_input_dim, hidden_dims=args.hidden_sizes, num_classes=len(label_encoder.classes_), dropout=0.3).to(args.device)
        mfcc_optimizer = torch.optim.Adam(mfcc_model.parameters(), lr=args.lr)
        mfcc_criterion = nn.CrossEntropyLoss()
        # Train a few epochs (quick) on MFCC
        for mfcc_epoch in range(1, 6):
            mfcc_model.train()
            for feats, labs, _ in mfcc_train_loader:
                feats, labs = feats.to(args.device), labs.to(args.device)
                mfcc_optimizer.zero_grad()
                out = mfcc_model(feats)
                loss = mfcc_criterion(out, labs)
                loss.backward()
                mfcc_optimizer.step()
        _, mfcc_acc, mfcc_preds, mfcc_labels = evaluate(mfcc_model, mfcc_test_loader, mfcc_criterion, args.device)
        print(f"MFCC quick test accuracy (5 epochs): {mfcc_acc:.2f}%")
        # High-level comparison statement
        print("\nComparison Summary:")
        print(f"  HuBERT Test Accuracy: {test_acc:.2f}%")
        print(f"  MFCC (quick 5-epoch) Accuracy: {mfcc_acc:.2f}%")
        print("  → HuBERT generally captures richer phonetic + prosody cues; MFCC limited to spectral envelope stats.")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, label_encoder.classes_, output_dir / 'confusion_matrix.png')
    print(f"\n✓ Confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")
    
    # Save label encoder
    with open(output_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"✓ Label encoder saved to: {output_dir / 'label_encoder.pkl'}")
    print(f"✓ Best model saved to: {output_dir / 'best_model.pt'}")
    print("\n🎉 Training complete!")


if __name__ == '__main__':
    main()
