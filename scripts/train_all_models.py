"""
Train all model architectures (MLP, CNN, BiLSTM, Transformer) for comparison
"""
import sys
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import numpy as np

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from src.features.dataset import load_features_and_create_dataset
from src.models.mlp import MLPClassifier
from src.models.cnn import CNNClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.transformer import TransformerClassifier

FEATURES_PATH = root / 'data' / 'features' / 'indian_accents' / 'hubert_layer12_mean.pkl'
OUTPUT_BASE = root / 'experiments'

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1, all_preds, all_labels

def plot_confusion_matrix(cm, classes, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model_type='mlp', num_epochs=30, batch_size=32, lr=0.001):
    """Train a single model architecture"""
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading features...")
    train_csv = str(root / 'data' / 'splits' / 'train.csv')
    val_csv = str(root / 'data' / 'splits' / 'val.csv')
    test_csv = str(root / 'data' / 'splits' / 'test.csv')
    
    train_dataset, label_encoder = load_features_and_create_dataset(str(FEATURES_PATH), train_csv)
    val_dataset, _ = load_features_and_create_dataset(str(FEATURES_PATH), val_csv, label_encoder=label_encoder)
    test_dataset, _ = load_features_and_create_dataset(str(FEATURES_PATH), test_csv, label_encoder=label_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get feature dimension
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]
    num_classes = len(label_encoder.classes_)
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if model_type == 'mlp':
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=[256, 128],
            num_classes=num_classes,
            dropout=0.3
        )
    elif model_type == 'cnn':
        model = CNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            num_channels=[64, 128, 256],
            dropout=0.3
        )
    elif model_type == 'bilstm':
        model = BiLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        )
    elif model_type == 'transformer':
        model = TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Create output directory
    output_dir = OUTPUT_BASE / f'indian_accents_{model_type}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'epoch': epoch,
                'model_type': model_type,
                'input_dim': input_dim,
                'num_classes': num_classes
            }, output_dir / 'best_model.pt')
            print(f"✓ Saved new best model (val_acc: {val_acc*100:.2f}%)")
    
    # Load best model and evaluate on test set
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Set")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=label_encoder.classes_, 
                                   output_dict=True, zero_division=0)
    print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, label_encoder.classes_, output_dir / 'confusion_matrix.png')
    print(f"✓ Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    # Save results
    results = {
        'model_type': model_type,
        'best_val_acc': best_val_acc,
        'best_val_f1': history['val_f1'][np.argmax(history['val_acc'])],
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'history': history,
        'classification_report': report,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'input_dim': input_dim,
        'num_classes': num_classes
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save label encoder
    with open(output_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n✓ Results saved to {output_dir}")
    
    return results

def main():
    """Train all models and compare"""
    models = ['mlp', 'cnn', 'bilstm', 'transformer']
    all_results = {}
    
    for model_type in models:
        try:
            results = train_model(model_type=model_type, num_epochs=30, batch_size=32, lr=0.001)
            all_results[model_type] = results
        except Exception as e:
            print(f"\n❌ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare results
    print(f"\n{'='*80}")
    print("Model Comparison")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<15} {'Val Acc':<10} {'Val F1':<10} {'Test Acc':<10} {'Test F1':<10}")
    print("-" * 55)
    for model_type, results in all_results.items():
        print(f"{model_type.upper():<15} {results['best_val_acc']*100:>7.2f}% {results['best_val_f1']:>8.4f} {results['test_acc']*100:>9.2f}% {results['test_f1']:>8.4f}")
    
    # Save comparison
    comparison_file = OUTPUT_BASE / 'model_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Model comparison saved to {comparison_file}")

if __name__ == '__main__':
    main()
