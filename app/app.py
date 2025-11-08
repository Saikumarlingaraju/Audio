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
import librosa
import os
import soundfile as sf

from src.models.mlp import MLPClassifier
from src.features.mfcc_extractor import MFCCExtractor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and label encoder
MODEL_PATH = root_dir / 'experiments' / 'indian_accents_mlp' / 'best_model.pt'
ENCODER_PATH = root_dir / 'experiments' / 'indian_accents_mlp' / 'label_encoder.pkl'

print("Loading model...")
checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')
with open(str(ENCODER_PATH), 'rb') as f:
    label_encoder = pickle.load(f)

model = MLPClassifier(input_dim=600, hidden_dims=[256, 128], num_classes=6, dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize MFCC extractor
mfcc_extractor = MFCCExtractor(
    sr=16000,
    n_mfcc=40,
    include_delta=True,
    include_delta_delta=True
)

print(f"✓ Model loaded: Test accuracy {checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A')):.2f}%")
print(f"✓ Classes: {list(label_encoder.classes_)}")


def extract_features(audio_path):
    """Extract MFCC features from audio file"""
    features_dict = mfcc_extractor.extract_from_file(audio_path, return_frames=False)
    # Extract just the array (not the dict)
    features = features_dict['features'] if isinstance(features_dict, dict) else features_dict
    return features


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
        # Extract features
        features = extract_features(audio_path)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
        
        # Get prediction and confidence
        predicted_label = label_encoder.classes_[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100
        
        # Get top 3 predictions
        top_k = min(3, len(label_encoder.classes_))
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        top_predictions = [
            {
                'accent': str(label_encoder.classes_[idx.item()]),
                'confidence': prob.item() * 100
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        # Get cuisine recommendations
        recommendations = get_cuisine_recommendations(str(predicted_label))
        
        return {
            'success': True,
            'predicted_accent': str(predicted_label),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'recommendations': recommendations
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


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
        'architecture': '[600 → 256 → 128 → 6]',
        'parameters': '188,294',
        'test_accuracy': '100.00%',
        'dataset': 'IndicAccentDB (2,798 samples)',
        'features': 'MFCC (600-dimensional)',
        'classes': [str(c) for c in label_encoder.classes_],
        'num_classes': len(label_encoder.classes_)
    })


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌐 NATIVE LANGUAGE IDENTIFICATION - INDIAN ACCENTS")
    print("=" * 80)
    print(f"✓ Model: MLP Classifier (100.00% test accuracy)")
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
