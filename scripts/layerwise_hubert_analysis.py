"""
Layer-wise HuBERT analysis: extract pooled features for selected layers and evaluate accuracy.

Default layers: [0, 3, 6, 9, 12] for a quick but representative sweep.
Outputs per-layer results to experiments/hubert_layerwise/results.json
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
# Ensure project root is on path so `src` is importable when running this script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.hubert_extractor import HuBERTExtractor
from src.features.dataset import load_features_and_create_dataset
from src.models.mlp import MLPClassifier


def extract_layer_features(layer: int, pooling: str = 'mean') -> Path:
    """Extract HuBERT features for a specific layer with pooling.

    Returns path to saved pkl file.
    """
    from tqdm import tqdm
    
    metadata_csv = Path('data/raw/indian_accents/metadata_with_splits.csv')
    assert metadata_csv.exists(), f"Missing metadata: {metadata_csv}"

    extractor = HuBERTExtractor(model_name="facebook/hubert-base-ls960")

    df = pd.read_csv(metadata_csv)
    base_path = Path('data/raw/indian_accents')

    features_list = []
    errors = []
    
    print(f"\nExtracting layer {layer} features from {len(df)} audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Layer {layer}"):
        audio_path = base_path / row['filepath']
        if not audio_path.exists():
            errors.append((str(audio_path), "File not found"))
            continue
        try:
            result = extractor.extract_from_file(str(audio_path), extract_layer=layer, pooling=pooling)
            if result is None:
                errors.append((str(audio_path), "Extraction returned None"))
                continue
            emb = result['embeddings']
            key = f'layer_{layer}'
            if key not in emb:
                # fallback to first key
                key = list(emb.keys())[0]
            feat = np.asarray(emb[key], dtype=np.float32)
            features_list.append({
                'features': feat,
                'native_language': row['native_language'],
                'split': row['split'],
                'filepath': str(audio_path)
            })
        except Exception as e:
            errors.append((str(audio_path), str(e)))
            continue

    print(f"✓ Extracted {len(features_list)} samples (errors: {len(errors)})")
    if errors and len(errors) < 10:
        print("First errors:")
        for path, err in errors[:5]:
            print(f"  - {Path(path).name}: {err}")

    out_dir = Path('data/features/indian_accents')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'hubert_layer{layer}_{pooling}.pkl'

    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump(features_list, f)
    
    print(f"✓ Saved to {out_path}")
    return out_path


def train_eval(features_path: str, output_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_csv = 'data/splits/train.csv'
    val_csv = 'data/splits/val.csv'
    test_csv = 'data/splits/test.csv'

    train_dataset, label_encoder = load_features_and_create_dataset(features_path, train_csv)
    val_dataset, _ = load_features_and_create_dataset(features_path, val_csv, label_encoder=label_encoder)
    test_dataset, _ = load_features_and_create_dataset(features_path, test_csv, label_encoder=label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(label_encoder.classes_)

    model = MLPClassifier(input_dim=input_dim, hidden_dims=[256, 128], num_classes=num_classes, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def run_epoch(loader, train=False):
        if train:
            model.train()
        else:
            model.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.set_grad_enabled(train):
            for feats, labs, _ in loader:
                feats, labs = feats.to(device), labs.to(device)
                if train:
                    optimizer.zero_grad()
                out = model(feats)
                loss = criterion(out, labs)
                if train:
                    loss.backward(); optimizer.step()
                preds = out.argmax(1)
                correct += (preds == labs).sum().item()
                total += labs.size(0)
                loss_sum += loss.item()
        return loss_sum / max(1, len(loader)), (100.0 * correct / max(1, total))

    best_val = 0.0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader, train=False)
        if va_acc > best_val:
            best_val = va_acc
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': va_acc}, Path(output_dir) / 'best_model.pt')
        print(f"Epoch {ep:02d}: train_acc={tr_acc:.2f}% val_acc={va_acc:.2f}%")

    # Load best and evaluate test
    ckpt = torch.load(Path(output_dir) / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    te_loss, te_acc = run_epoch(test_loader, train=False)
    return best_val, te_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, default='0,3,6,9,12', help='Comma-separated list of layers to evaluate')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean','max'], help='Temporal pooling method')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
    results = {}

    for L in layers:
        print(f"\n=== Layer {L} ({args.pooling}) ===")
        feat_path = extract_layer_features(L, pooling=args.pooling)
        out_dir = f'experiments/indian_accents_hubert_layer{L}_{args.pooling}'
        val_acc, test_acc = train_eval(str(feat_path), out_dir, epochs=args.epochs)
        print(f"Layer {L}: best_val={val_acc:.2f}% test={test_acc:.2f}%")
        results[str(L)] = {'val_acc': val_acc, 'test_acc': test_acc, 'pooling': args.pooling}

    # Save summary
    summary_dir = Path('experiments/hubert_layerwise')
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved layer-wise results to {summary_dir / 'results.json'}")


if __name__ == '__main__':
    main()
