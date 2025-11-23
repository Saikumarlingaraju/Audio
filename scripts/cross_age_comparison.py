"""
Cross-Age Generalization Analysis
Compares model performance when training on adults vs children
Tests generalization across age groups
"""
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pickle

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from src.features.dataset import load_features_and_create_dataset
from src.models.mlp import MLPClassifier

def check_age_groups():
    """Check if dataset contains both adults and children"""
    metadata = pd.read_csv('data/raw/indian_accents/metadata_full.csv')
    age_groups = metadata['age_group'].unique()
    
    print(f"\n{'='*80}")
    print("Dataset Age Group Check")
    print(f"{'='*80}\n")
    print(f"Age groups found: {age_groups}")
    print(f"\nDistribution:")
    print(metadata['age_group'].value_counts())
    
    if len(age_groups) < 2:
        print(f"\n❌ Cross-age analysis requires both adult and child data")
        print(f"   Currently only have: {age_groups}")
        print(f"\n💡 To add child data:")
        print(f"   python scripts/add_child_speakers.py")
        return False
    
    return True

def create_age_split_metadata(metadata_path='data/raw/indian_accents/metadata_full.csv',
                               output_dir='data/splits'):
    """
    Create separate train/test splits for cross-age experiments
    """
    metadata = pd.read_csv(metadata_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    adults = metadata[metadata['age_group'] == 'adult']
    children = metadata[metadata['age_group'] == 'child']
    
    print(f"\n{'='*80}")
    print("Creating Cross-Age Splits")
    print(f"{'='*80}\n")
    print(f"Adult samples: {len(adults)}")
    print(f"Child samples: {len(children)}")
    
    # Experiment 1: Train on adults, test on children
    adults_train = adults[adults['split'] == 'train']
    adults_val = adults[adults['split'] == 'val']
    children_test = children  # All children for testing
    
    adults_train.to_csv(output_path / 'adults_train.csv', index=False)
    adults_val.to_csv(output_path / 'adults_val.csv', index=False)
    children_test.to_csv(output_path / 'children_test.csv', index=False)
    
    print(f"✓ Experiment 1: Train on adults ({len(adults_train)}), test on children ({len(children_test)})")
    
    # Experiment 2: Train on children, test on adults (if enough child data)
    if len(children) > 100:  # Need sufficient training data
        # Split children into train/val
        from sklearn.model_selection import train_test_split
        children_train, children_val = train_test_split(
            children, test_size=0.2, random_state=42,
            stratify=children['native_language']
        )
        adults_test = adults[adults['split'] == 'test']
        
        children_train.to_csv(output_path / 'children_train.csv', index=False)
        children_val.to_csv(output_path / 'children_val.csv', index=False)
        adults_test.to_csv(output_path / 'adults_test.csv', index=False)
        
        print(f"✓ Experiment 2: Train on children ({len(children_train)}), test on adults ({len(adults_test)})")
        return True
    else:
        print(f"⚠️  Not enough child data for Experiment 2 (need >100, have {len(children)})")
        return False

def train_and_evaluate(train_csv, val_csv, test_csv, experiment_name, 
                       features_path='data/features/indian_accents/hubert_layer12_mean.pkl',
                       num_epochs=20):
    """
    Train model on one age group and test on another
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Load datasets
    train_dataset, label_encoder = load_features_and_create_dataset(features_path, train_csv)
    val_dataset, _ = load_features_and_create_dataset(features_path, val_csv, label_encoder=label_encoder)
    test_dataset, _ = load_features_and_create_dataset(features_path, test_csv, label_encoder=label_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model setup
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]
    num_classes = len(label_encoder.classes_)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim=input_dim, hidden_dims=[256, 128], 
                         num_classes=num_classes, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    # Load best model and test
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\nResults:")
    print(f"  Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Test Accuracy (cross-age): {test_acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                               target_names=label_encoder.classes_, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    return {
        'experiment': experiment_name,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'confusion_matrix': cm,
        'label_encoder': label_encoder,
        'predictions': test_preds,
        'true_labels': test_labels
    }

def plot_cross_age_comparison(results_list, output_path='experiments/cross_age_comparison.png'):
    """
    Create visualization comparing cross-age generalization
    """
    fig, axes = plt.subplots(1, len(results_list) + 1, figsize=(15, 5))
    
    # Plot 1: Accuracy comparison
    experiment_names = [r['experiment'] for r in results_list]
    test_accs = [r['test_acc'] * 100 for r in results_list]
    val_accs = [r['val_acc'] * 100 for r in results_list]
    
    x = range(len(experiment_names))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], val_accs, width, label='Val (same age)', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], test_accs, width, label='Test (cross-age)', alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Cross-Age Generalization')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name.replace(' ', '\n') for name in experiment_names], fontsize=9)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot confusion matrices
    for idx, result in enumerate(results_list):
        cm = result['confusion_matrix']
        classes = result['label_encoder'].classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes,
                   ax=axes[idx+1], cbar=False)
        axes[idx+1].set_title(result['experiment'])
        axes[idx+1].set_ylabel('True')
        axes[idx+1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_path}")

def main():
    """
    Run cross-age generalization experiments
    """
    print(f"\n{'#'*80}")
    print("# CROSS-AGE GENERALIZATION ANALYSIS")
    print(f"{'#'*80}\n")
    
    # Check if we have both age groups
    if not check_age_groups():
        return
    
    # Create age-specific splits
    has_both_experiments = create_age_split_metadata()
    
    # Check if features exist
    features_path = 'data/features/indian_accents/hubert_layer12_mean.pkl'
    if not Path(features_path).exists():
        print(f"\n❌ Features not found: {features_path}")
        print(f"   Run: python extract_indian_features.py")
        return
    
    results = []
    
    # Experiment 1: Train on adults, test on children
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: Train on Adults → Test on Children")
    print(f"{'='*80}")
    
    result1 = train_and_evaluate(
        train_csv='data/splits/adults_train.csv',
        val_csv='data/splits/adults_val.csv',
        test_csv='data/splits/children_test.csv',
        experiment_name='Adults→Children',
        features_path=features_path,
        num_epochs=20
    )
    results.append(result1)
    
    # Experiment 2: Train on children, test on adults (if possible)
    if has_both_experiments:
        print(f"\n{'='*80}")
        print("EXPERIMENT 2: Train on Children → Test on Adults")
        print(f"{'='*80}")
        
        result2 = train_and_evaluate(
            train_csv='data/splits/children_train.csv',
            val_csv='data/splits/children_val.csv',
            test_csv='data/splits/adults_test.csv',
            experiment_name='Children→Adults',
            features_path=features_path,
            num_epochs=20
        )
        results.append(result2)
    
    # Generate comparison plot
    plot_cross_age_comparison(results)
    
    # Summary
    print(f"\n{'#'*80}")
    print("# CROSS-AGE ANALYSIS SUMMARY")
    print(f"{'#'*80}\n")
    
    for result in results:
        print(f"{result['experiment']}:")
        print(f"  Validation Accuracy (same age): {result['val_acc']*100:.2f}%")
        print(f"  Test Accuracy (cross-age): {result['test_acc']*100:.2f}%")
        print(f"  Generalization Gap: {(result['val_acc']-result['test_acc'])*100:.2f}%")
        print()
    
    print(f"{'='*80}")
    print("Analysis complete! Check experiments/cross_age_comparison.png")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
