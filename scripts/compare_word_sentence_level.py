"""
Compare model performance on word-level vs sentence-level data
"""
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

def extract_word_features():
    """Extract HuBERT features for word-level data"""
    from src.features.hubert_extractor import HuBERTExtractor
    import torch
    import pickle
    from tqdm import tqdm
    
    print("\n" + "="*80)
    print("Extracting Word-Level Features")
    print("="*80 + "\n")
    
    metadata_path = Path('data/word_level/word_level_metadata.csv')
    output_path = Path('data/features/word_level')
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not metadata_path.exists():
        print("❌ Word-level metadata not found. Creating word-level dataset first...")
        from scripts.create_word_level_dataset import create_word_level_dataset
        create_word_level_dataset()
    
    metadata = pd.read_csv(metadata_path)
    extractor = HuBERTExtractor()
    
    features_dict = {}
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
        audio_path = Path('data') / row['filepath']
        
        if not audio_path.exists():
            continue
        
        try:
            # Extract HuBERT embeddings
            pooled = extractor.extract_from_file(str(audio_path), layer=12, pooling='mean')
            mean_vec = pooled['embeddings']['layer_12']
            
            # Also get std
            frame_level = extractor.extract_from_file(str(audio_path), layer=12, pooling=None)
            frames = frame_level['embeddings']['layer_12']
            std_vec = frames.std(axis=0)
            
            # Concatenate mean and std
            feature_vec = torch.cat([mean_vec, torch.from_numpy(std_vec)]).numpy()
            
            features_dict[row['filepath']] = {
                'features': feature_vec,
                'label': row['native_language'],
                'split': row['split']
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Save features
    output_file = output_path / 'hubert_layer12_mean.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"\n✓ Word-level features saved to {output_file}")
    print(f"Total features extracted: {len(features_dict)}")
    
    return output_file

def train_on_level(level='sentence', num_epochs=20):
    """Train model on specific data level"""
    from scripts.train_all_models import train_model
    
    if level == 'sentence':
        features_path = 'data/features/indian_accents/hubert_layer12_mean.pkl'
        output_dir = f'experiments/sentence_level_mlp'
    else:
        features_path = 'data/features/word_level/hubert_layer12_mean.pkl'
        output_dir = f'experiments/word_level_mlp'
    
    print(f"\n{'='*80}")
    print(f"Training on {level.upper()}-Level Data")
    print(f"{'='*80}\n")
    
    # Modify train_model to use custom paths
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score
    from src.features.dataset import load_features_and_create_dataset
    from src.models.mlp import MLPClassifier
    import torch.nn as nn
    import pickle
    from tqdm import tqdm
    
    # Load data
    if level == 'word':
        train_csv = 'data/word_level/word_level_metadata.csv'
        # Filter by split
        import pandas as pd
        df = pd.read_csv(train_csv)
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        # Save temporary split files
        train_csv_temp = 'data/word_level/train_temp.csv'
        val_csv_temp = 'data/word_level/val_temp.csv'
        test_csv_temp = 'data/word_level/test_temp.csv'
        train_df.to_csv(train_csv_temp, index=False)
        val_df.to_csv(val_csv_temp, index=False)
        test_df.to_csv(test_csv_temp, index=False)
        
        train_dataset, label_encoder = load_features_and_create_dataset(features_path, train_csv_temp)
        val_dataset, _ = load_features_and_create_dataset(features_path, val_csv_temp, label_encoder=label_encoder)
        test_dataset, _ = load_features_and_create_dataset(features_path, test_csv_temp, label_encoder=label_encoder)
    else:
        train_csv = 'data/splits/train.csv'
        val_csv = 'data/splits/val.csv'
        test_csv = 'data/splits/test.csv'
        
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
    history = {'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_preds, train_labels = [], []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
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
        
        print(f"Epoch {epoch+1}: Train Acc = {train_acc*100:.2f}%, Val Acc = {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
    
    # Test
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    
    results = {
        'level': level,
        'train_acc': history['train_acc'][-1],
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'history': history
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ {level.capitalize()}-level results:")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    
    return results

def compare_levels():
    """Main comparison function"""
    print(f"\n{'='*80}")
    print("Comparing Word-Level vs Sentence-Level Performance")
    print(f"{'='*80}\n")
    
    results = {}
    
    # Train on sentence-level (should already exist)
    sentence_results_path = Path('experiments/sentence_level_mlp/results.json')
    if sentence_results_path.exists():
        with open(sentence_results_path, 'r') as f:
            results['sentence'] = json.load(f)
        print("✓ Loaded existing sentence-level results")
    else:
        print("Training on sentence-level data...")
        results['sentence'] = train_on_level('sentence', num_epochs=20)
    
    # Extract word-level features if needed
    word_features_path = Path('data/features/word_level/hubert_layer12_mean.pkl')
    if not word_features_path.exists():
        print("\nExtracting word-level features...")
        extract_word_features()
    
    # Train on word-level
    print("\nTraining on word-level data...")
    results['word'] = train_on_level('word', num_epochs=20)
    
    # Visualization
    print(f"\n{'='*80}")
    print("Generating Comparison Plots")
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
    axes[0].set_ylim([0, 100])
    
    # Plot 2: Training curves
    epochs_s = range(1, len(results['sentence']['history']['val_acc']) + 1)
    epochs_w = range(1, len(results['word']['history']['val_acc']) + 1)
    
    axes[1].plot(epochs_s, [acc * 100 for acc in results['sentence']['history']['val_acc']], 
                label='Sentence-Level Val', marker='o', markersize=4, linewidth=2, color='#3498db')
    axes[1].plot(epochs_w, [acc * 100 for acc in results['word']['history']['val_acc']], 
                label='Word-Level Val', marker='s', markersize=4, linewidth=2, color='#e74c3c')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Training Curves: Word vs Sentence Level', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = 'experiments/word_vs_sentence_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}\n")
    print(f"{'Level':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 56)
    print(f"{'Sentence-Level':<20} {train_accs[0]:>9.2f}% {val_accs[0]:>10.2f}% {test_accs[0]:>10.2f}%")
    print(f"{'Word-Level':<20} {train_accs[1]:>9.2f}% {val_accs[1]:>10.2f}% {test_accs[1]:>10.2f}%")
    
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    if test_accs[0] > test_accs[1]:
        diff = test_accs[0] - test_accs[1]
        print(f"✓ Sentence-level performs better by {diff:.2f}%")
        print("  → Suggests prosodic and rhythmic cues are important for accent detection")
    else:
        diff = test_accs[1] - test_accs[0]
        print(f"✓ Word-level performs better by {diff:.2f}%")
        print("  → Suggests phonetic details at word level are more discriminative")
    
    # Save comparison results
    comparison_file = 'experiments/word_sentence_comparison_results.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Comparison results saved to {comparison_file}")
    
    return results

if __name__ == '__main__':
    compare_levels()
