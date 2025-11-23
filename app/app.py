"""
Flask Web App - Native Language Identification
Uses trained MLP model on AccentDB Extended dataset
"""
import sys
from pathlib import Path

# Add parent directory to path to import src modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from flask import Flask, render_template, request, jsonify
import torch
import pickle
import numpy as np
import os

from src.models.mlp import MLPClassifier

# We'll import HuBERTExtractor lazily to avoid heavy imports during server start
HubertExtractor = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and label encoder
MODEL_PATH = root_dir / 'experiments' / 'indian_accents_hubert' / 'best_model.pt'
ENCODER_PATH = root_dir / 'experiments' / 'indian_accents_hubert' / 'label_encoder.pkl'

print("Loading model...")
checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')
with open(str(ENCODER_PATH), 'rb') as f:
    label_encoder = pickle.load(f)

model = MLPClassifier(input_dim=1536, hidden_dims=[256, 128], num_classes=6, dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize HuBERT feature extractor (model is loaded in the extractor's constructor)
hubert_extractor = None  # Lazy init to avoid blocking server start on network issues

# Safe printing of accuracy whether numeric or not
_acc = checkpoint.get('test_acc', checkpoint.get('val_acc', None))
if isinstance(_acc, (int, float)):
    print(f"✓ Model weights loaded: Test accuracy {_acc:.2f}%")
else:
    print(f"✓ Model weights loaded. Test accuracy: {_acc}")
print(f"✓ Classes: {list(label_encoder.classes_)}")
print("→ HuBERT extractor will initialize on first prediction request.")


def extract_features(audio_path):
    """Extract HuBERT features (mean+std over frames from layer 12 -> 1536 dims)."""
    try:
        global hubert_extractor
        if hubert_extractor is None:
            try:
                print("Initializing HuBERT extractor (lazy)...")
                # Lazy import to prevent heavy transformer/torch submodules from loading at app import time
                global HubertExtractor
                if HubertExtractor is None:
                    from importlib import import_module
                    HubertExtractor = getattr(import_module('src.features.hubert_extractor'), 'HuBERTExtractor')
                hubert_extractor = HubertExtractor()
            except Exception as init_err:
                print(f"HuBERT init failed: {init_err}")
                return None

        # Ask extractor for frame-level embeddings from layer 12
        result = hubert_extractor.extract_from_file(
            audio_path, extract_layer=12, pooling=None
        )

        if result is None:
            print(f"Feature extraction returned None for {audio_path}")
            return None

        # Pull out frames for the requested layer
        frames = None
        if isinstance(result, dict):
            embeddings_dict = result.get('embeddings')
            if isinstance(embeddings_dict, dict):
                if 'layer_12' in embeddings_dict:
                    frames = embeddings_dict['layer_12']  # shape: (T, 768)
                elif 12 in embeddings_dict:
                    frames = embeddings_dict[12]
                else:
                    # Fallback to first key
                    first_key = list(embeddings_dict.keys())[0]
                    print(f"Warning: using {first_key} instead of layer_12")
                    frames = embeddings_dict[first_key]
            else:
                # Unexpected shape; try to use as frames
                frames = embeddings_dict
        else:
            frames = result

        if frames is None:
            print(f"No frame-level embeddings found for {audio_path}")
            return None

        frames = np.asarray(frames)
        if frames.ndim == 1:
            # Already pooled to (768,), compute std as zeros to keep 1536 dims
            mean_vec = frames.astype(np.float32)
            std_vec = np.zeros_like(mean_vec, dtype=np.float32)
        else:
            # Compute mean and std over time dimension
            mean_vec = np.nanmean(frames, axis=0).astype(np.float32)
            std_vec = np.nanstd(frames, axis=0).astype(np.float32)

        # Replace NaNs/Infs if any
        mean_vec = np.nan_to_num(mean_vec, nan=0.0, posinf=0.0, neginf=0.0)
        std_vec = np.nan_to_num(std_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate to get 1536-dim feature
        features = np.concatenate([mean_vec, std_vec]).astype(np.float32)
        print(f"HuBERT features ready: {features.shape} (mean+std)")

        return features

    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_cuisine_recommendations(state):
    """Get cuisine recommendations based on predicted state"""
    cuisine_map = {
        'andhra_pradesh': [
            {'name': 'Hyderabadi Biryani', 'region': 'Andhra Pradesh', 'description': 'Aromatic rice dish with marinated meat'},
            {'name': 'Pesarattu', 'region': 'Andhra Pradesh', 'description': 'Green gram dosa served with ginger chutney'},
            {'name': 'Gongura Mutton', 'region': 'Andhra Pradesh', 'description': 'Tangy mutton curry with sorrel leaves'}
        ],
        'gujrat': [
            {'name': 'Dhokla', 'region': 'Gujarat', 'description': 'Steamed fermented rice and chickpea snack'},
            {'name': 'Gujarati Thali', 'region': 'Gujarat', 'description': 'Complete meal with variety of sweet and savory dishes'},
            {'name': 'Khandvi', 'region': 'Gujarat', 'description': 'Rolled gram flour snacks with tempered spices'}
        ],
        'jharkhand': [
            {'name': 'Litti Chokha', 'region': 'Jharkhand', 'description': 'Roasted wheat balls with mashed vegetables'},
            {'name': 'Dhuska', 'region': 'Jharkhand', 'description': 'Deep-fried rice and lentil pancakes'},
            {'name': 'Rugra', 'region': 'Jharkhand', 'description': 'Mushroom curry with local spices'}
        ],
        'karnataka': [
            {'name': 'Bisi Bele Bath', 'region': 'Karnataka', 'description': 'Spicy rice and lentil dish with vegetables'},
            {'name': 'Mysore Masala Dosa', 'region': 'Karnataka', 'description': 'Crispy dosa with red chutney and potato filling'},
            {'name': 'Ragi Mudde', 'region': 'Karnataka', 'description': 'Finger millet balls with sambar'}
        ],
        'kerala': [
            {'name': 'Appam with Stew', 'region': 'Kerala', 'description': 'Soft rice pancakes with coconut milk stew'},
            {'name': 'Kerala Fish Curry', 'region': 'Kerala', 'description': 'Fish cooked in coconut milk with spices'},
            {'name': 'Puttu and Kadala', 'region': 'Kerala', 'description': 'Steamed rice cylinders with chickpea curry'}
        ],
        'tamil': [
            {'name': 'Chettinad Chicken', 'region': 'Tamil Nadu', 'description': 'Spicy chicken curry with aromatic spices'},
            {'name': 'Idli Sambar', 'region': 'Tamil Nadu', 'description': 'Steamed rice cakes with lentil soup'},
            {'name': 'Pongal', 'region': 'Tamil Nadu', 'description': 'Savory rice and lentil dish with pepper'}
        ]
    }
    
    return cuisine_map.get(state.lower(), [])


def predict_accent(audio_path):
    """Predict accent/language from audio file"""
    try:
        features = extract_features(audio_path)
        if features is None:
            return {'success': False, 'error': 'Feature extraction failed'}

        # Ensure float32 1-D
        features = np.asarray(features, dtype=np.float32).reshape(-1)
        print(f"Feature vector final shape: {features.shape}")

        expected_dim = 1536
        if features.size != expected_dim:
            print(f"Warning: resizing feature vector from {features.size} to {expected_dim}")
            if features.size < expected_dim:
                features = np.pad(features, (0, expected_dim - features.size))
            else:
                features = features[:expected_dim]

        features_tensor = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        idx = torch.argmax(probs).item()
        predicted = str(label_encoder.classes_[idx])
        confidence = probs[idx].item() * 100
        top_k = min(3, len(label_encoder.classes_))
        top_probs, top_indices = torch.topk(probs, top_k)
        top_predictions = [
            {'accent': str(label_encoder.classes_[i.item()]), 'confidence': p.item() * 100}
            for p, i in zip(top_probs, top_indices)
        ]
        return {
            'success': True,
            'predicted_accent': predicted,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'recommendations': get_cuisine_recommendations(predicted)
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'success': False, 'error': f'Prediction failure: {e}'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file uploaded'})
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    audio_file.save(filepath)
    
    # Make prediction
    result = predict_accent(filepath)
    
    # Clean up
    try:
        os.remove(filepath)
    except:
        pass
    
    return jsonify(result)


@app.route('/info')
def info():
    """Display model information"""
    return jsonify({
        'model': 'MLP Classifier - Indian Accents',
        'architecture': '[1536 → 256 → 128 → 6]',
        'parameters': '188,294',
        'test_accuracy': f"{checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))}",
        'dataset': 'IndicAccentDB (2,798 samples)',
        'features': 'HuBERT Layer 12 mean+std (1536-dimensional)',
        'classes': [str(c) for c in label_encoder.classes_],
        'num_classes': len(label_encoder.classes_)
    })


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌐 NATIVE LANGUAGE IDENTIFICATION - INDIAN ACCENTS")
    print("=" * 80)
    try:
        test_acc = checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))
        print(f"✓ Model: MLP Classifier ({test_acc:.2f}% test accuracy)")
    except Exception:
        print(f"✓ Model: MLP Classifier (test accuracy: {checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))})")
    print(f"✓ Classes: {', '.join([str(c) for c in label_encoder.classes_])}")
    print(f"✓ Server starting on: http://127.0.0.1:5000")
    print("=" * 80)
    print("\n📝 Usage:")
    print("  1. Open http://127.0.0.1:5000 in your browser")
    print("  2. Upload an audio file (WAV, MP3, etc.)")
    print("  3. Get Indian state accent prediction with confidence scores")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
