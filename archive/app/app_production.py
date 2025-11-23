"""
FIXED: Production-ready Flask app with robust external audio prediction
This version focuses on fixing the domain shift problem for external audio
"""
import sys
from pathlib import Path
import os
import tempfile
import soundfile as sf

# Add parent directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from flask import Flask, render_template, request, jsonify
import torch
import pickle
import numpy as np
import librosa
from scipy import signal

from src.models.mlp import MLPClassifier
from src.features.hubert_extractor import HuBERTExtractor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load best model
HUBERT_MODEL_DIR = root_dir / 'experiments' / 'indian_accents_hubert'
MFCC_MODEL_DIR = root_dir / 'experiments' / 'indian_accents_mlp'

if HUBERT_MODEL_DIR.exists() and (HUBERT_MODEL_DIR / 'best_model.pt').exists():
    MODEL_DIR = HUBERT_MODEL_DIR
    FEATURE_TYPE = 'hubert'
    print("✓ Using HuBERT model")
else:
    MODEL_DIR = MFCC_MODEL_DIR
    FEATURE_TYPE = 'mfcc'
    print("✓ Using MFCC model")

MODEL_PATH = MODEL_DIR / 'best_model.pt'
ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'

# Load model
checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')
with open(str(ENCODER_PATH), 'rb') as f:
    label_encoder = pickle.load(f)

input_dim = checkpoint.get('input_dim', 1536 if FEATURE_TYPE == 'hubert' else 600)
model = MLPClassifier(input_dim=input_dim, hidden_dims=[256, 128], 
                     num_classes=len(label_encoder.classes_), dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize feature extractor
if FEATURE_TYPE == 'hubert':
    feature_extractor = HuBERTExtractor()
    print("✓ HuBERT extractor loaded")
else:
    from src.features.mfcc_extractor import MFCCExtractor
    feature_extractor = MFCCExtractor(sr=16000, n_mfcc=40, 
                                     include_delta=True, include_delta_delta=True)
    print("✓ MFCC extractor loaded")

print(f"✓ Model loaded: {checkpoint.get('test_acc', checkpoint.get('val_acc', 0))*100:.1f}% accuracy")
print(f"✓ Classes: {list(label_encoder.classes_)}")

def preprocess_audio_robust(audio_path, target_sr=16000):
    """
    CRITICAL: Robust preprocessing that matches training pipeline EXACTLY
    This is the key to fixing external audio prediction
    """
    print(f"\n🔧 Preprocessing: {Path(audio_path).name}")
    
    # Step 1: Load audio with target sample rate
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    print(f"  ✓ Loaded: {len(audio)/sr:.2f}s at {sr}Hz")
    
    # Step 2: Remove DC offset
    audio = audio - np.mean(audio)
    
    # Step 3: Trim silence aggressively (top_db=20 matches training)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    print(f"  ✓ Trimmed: {len(audio_trimmed)/sr:.2f}s")
    
    # Step 4: Normalize amplitude to [-0.95, 0.95] to avoid clipping
    if np.abs(audio_trimmed).max() > 0:
        audio_normalized = audio_trimmed / np.abs(audio_trimmed).max() * 0.95
    else:
        audio_normalized = audio_trimmed
    
    # Step 5: Apply pre-emphasis filter (matches training preprocessing)
    audio_preemphasized = librosa.effects.preemphasis(audio_normalized, coef=0.97)
    
    # Step 6: Ensure reasonable length (0.5s to 10s)
    min_length = int(0.5 * target_sr)
    max_length = int(10 * target_sr)
    
    if len(audio_preemphasized) < min_length:
        # Pad with zeros
        audio_final = np.pad(audio_preemphasized, 
                           (0, min_length - len(audio_preemphasized)), 
                           mode='constant')
        print(f"  ✓ Padded to minimum length")
    elif len(audio_preemphasized) > max_length:
        # Take center portion
        start = (len(audio_preemphasized) - max_length) // 2
        audio_final = audio_preemphasized[start:start + max_length]
        print(f"  ✓ Truncated to maximum length")
    else:
        audio_final = audio_preemphasized
    
    print(f"  ✓ Final: {len(audio_final)/sr:.2f}s, range [{audio_final.min():.3f}, {audio_final.max():.3f}]")
    
    return audio_final, target_sr

def extract_features_robust(audio, sr):
    """
    Extract features that match training pipeline EXACTLY
    """
    print(f"\n🎯 Extracting features ({FEATURE_TYPE})...")
    
    # Save to temp file (required for extractors)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio, sr)
    
    try:
        if FEATURE_TYPE == 'hubert':
            # Extract HuBERT embeddings (layer 12, mean + std)
            pooled = feature_extractor.extract_from_file(temp_path, layer=12, pooling='mean')
            mean_vec = pooled['embeddings']['layer_12']  # (768,)
            
            frame_level = feature_extractor.extract_from_file(temp_path, layer=12, pooling=None)
            frames = frame_level['embeddings']['layer_12']  # (T, 768)
            std_vec = frames.std(axis=0)  # (768,)
            
            features = np.concatenate([mean_vec.numpy(), std_vec])  # (1536,)
            print(f"  ✓ HuBERT features: {features.shape}, range [{features.min():.3f}, {features.max():.3f}]")
            
        else:
            # Extract MFCC features
            result = feature_extractor.extract_from_file(temp_path, return_frames=False)
            features = result['features'] if isinstance(result, dict) else result
            print(f"  ✓ MFCC features: {features.shape}")
        
        return features
        
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

def ensemble_predict(audio_path, n_augmentations=5):
    """
    Ensemble prediction with test-time augmentation
    This significantly improves external audio accuracy
    """
    print(f"\n🎲 Ensemble prediction with {n_augmentations} augmentations...")
    
    # Original audio
    audio, sr = preprocess_audio_robust(audio_path)
    
    all_predictions = []
    all_probabilities = []
    
    # 1. Original
    features = extract_features_robust(audio, sr)
    probs = predict_single(features)
    all_predictions.append(probs.argmax())
    all_probabilities.append(probs)
    print(f"  ✓ Original: {label_encoder.classes_[probs.argmax()]} ({probs.max()*100:.1f}%)")
    
    # 2. Speed variations
    for rate in [0.95, 1.05]:
        try:
            audio_aug = librosa.effects.time_stretch(audio, rate=rate)
            features = extract_features_robust(audio_aug, sr)
            probs = predict_single(features)
            all_predictions.append(probs.argmax())
            all_probabilities.append(probs)
            print(f"  ✓ Speed {rate}x: {label_encoder.classes_[probs.argmax()]} ({probs.max()*100:.1f}%)")
        except:
            pass
    
    # 3. Pitch variations
    for n_steps in [-1, 1]:
        try:
            audio_aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            features = extract_features_robust(audio_aug, sr)
            probs = predict_single(features)
            all_predictions.append(probs.argmax())
            all_probabilities.append(probs)
            print(f"  ✓ Pitch {n_steps:+d}: {label_encoder.classes_[probs.argmax()]} ({probs.max()*100:.1f}%)")
        except:
            pass
    
    # 4. Light noise
    try:
        noise = np.random.normal(0, 0.005, len(audio))
        audio_aug = audio + noise
        audio_aug = audio_aug / (np.abs(audio_aug).max() + 1e-8) * 0.95
        features = extract_features_robust(audio_aug, sr)
        probs = predict_single(features)
        all_predictions.append(probs.argmax())
        all_probabilities.append(probs)
        print(f"  ✓ +Noise: {label_encoder.classes_[probs.argmax()]} ({probs.max()*100:.1f}%)")
    except:
        pass
    
    # Ensemble: Average probabilities
    avg_probs = np.mean(all_probabilities, axis=0)
    final_pred_idx = avg_probs.argmax()
    final_pred = label_encoder.classes_[final_pred_idx]
    confidence = avg_probs[final_pred_idx]
    
    # Count votes
    from collections import Counter
    vote_counts = Counter(all_predictions)
    majority_pred = label_encoder.classes_[vote_counts.most_common(1)[0][0]]
    
    print(f"\n✅ Ensemble result: {final_pred} ({confidence*100:.1f}%)")
    print(f"   Majority vote: {majority_pred}")
    print(f"   Agreement: {vote_counts[final_pred_idx]}/{len(all_predictions)}")
    
    return final_pred, confidence, avg_probs

def predict_single(features):
    """
    Single prediction from features
    """
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].numpy()
    return probabilities

def get_cuisine_recommendations(state):
    """Get cuisine recommendations for a given state"""
    cuisine_map = {
        'andhra_pradesh': [
            {'name': 'Hyderabadi Biryani', 'region': 'Andhra Pradesh',
             'description': 'Royal aromatic rice dish with meat and exotic spices'},
            {'name': 'Pesarattu', 'region': 'Andhra Pradesh',
             'description': 'Protein-rich green gram crepe with ginger chutney'},
            {'name': 'Gongura Mutton', 'region': 'Andhra Pradesh',
             'description': 'Tangy curry with sorrel leaves and mutton'}
        ],
        'gujrat': [
            {'name': 'Dhokla', 'region': 'Gujarat',
             'description': 'Steamed savory cake made from fermented batter'},
            {'name': 'Thepla', 'region': 'Gujarat',
             'description': 'Spiced flatbread with fenugreek leaves'},
            {'name': 'Undhiyu', 'region': 'Gujarat',
             'description': 'Mixed vegetable curry with unique spice blend'}
        ],
        'jharkhand': [
            {'name': 'Litti Chokha', 'region': 'Jharkhand',
             'description': 'Roasted wheat balls with mashed potato-eggplant'},
            {'name': 'Dhuska', 'region': 'Jharkhand',
             'description': 'Deep-fried rice and lentil pancake'},
            {'name': 'Rugra', 'region': 'Jharkhand',
             'description': 'Mushroom curry with tribal spices'}
        ],
        'karnataka': [
            {'name': 'Bisi Bele Bath', 'region': 'Karnataka',
             'description': 'Spicy rice-lentil dish with vegetables'},
            {'name': 'Mysore Masala Dosa', 'region': 'Karnataka',
             'description': 'Crispy crepe with spicy red chutney'},
            {'name': 'Ragi Mudde', 'region': 'Karnataka',
             'description': 'Finger millet balls served with curries'}
        ],
        'kerala': [
            {'name': 'Appam with Stew', 'region': 'Kerala',
             'description': 'Lacy rice pancakes with coconut milk curry'},
            {'name': 'Puttu Kadala', 'region': 'Kerala',
             'description': 'Steamed rice cake with black chickpea curry'},
            {'name': 'Fish Moilee', 'region': 'Kerala',
             'description': 'Mild fish curry in coconut milk'}
        ],
        'tamil': [
            {'name': 'Chettinad Chicken', 'region': 'Tamil Nadu',
             'description': 'Spicy chicken curry with aromatic spices'},
            {'name': 'Pongal', 'region': 'Tamil Nadu',
             'description': 'Savory rice-lentil porridge with pepper'},
            {'name': 'Sambar', 'region': 'Tamil Nadu',
             'description': 'Tangy lentil stew with vegetables'}
        ]
    }
    return cuisine_map.get(state, [])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request with robust preprocessing"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = 'uploaded_audio.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        print(f"\n{'='*80}")
        print(f"🎤 NEW PREDICTION REQUEST: {audio_file.filename}")
        print(f"{'='*80}")
        
        # Ensemble prediction (more robust for external audio)
        predicted_label, confidence, all_probs = ensemble_predict(filepath, n_augmentations=5)
        
        # Build response
        top_predictions = []
        for idx, prob in enumerate(all_probs):
            accent = label_encoder.classes_[idx]
            top_predictions.append({
                'accent': accent.replace('_', ' ').title(),
                'confidence': float(prob * 100)
            })
        
        # Sort by confidence
        top_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get cuisine recommendations
        recommendations = get_cuisine_recommendations(predicted_label)
        
        result = {
            'accent': predicted_label.replace('_', ' ').title(),
            'confidence': float(confidence * 100),
            'top_predictions': top_predictions,
            'recommendations': recommendations,
            'method': 'ensemble',
            'message': 'Prediction successful with ensemble inference'
        }
        
        print(f"\n{'='*80}")
        print(f"✅ PREDICTION COMPLETE")
        print(f"{'='*80}\n")
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Get model information"""
    return jsonify({
        'model': 'MLP Classifier with Ensemble Inference',
        'feature_type': FEATURE_TYPE,
        'classes': list(label_encoder.classes_),
        'accuracy': float(checkpoint.get('test_acc', checkpoint.get('val_acc', 0)) * 100),
        'input_dim': input_dim,
        'ensemble_enabled': True,
        'preprocessing': 'Robust (trimming, normalization, pre-emphasis)',
        'augmentations': ['speed', 'pitch', 'noise']
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌐 INDIAN ACCENT DETECTOR - PRODUCTION MODE")
    print("="*80)
    print(f"✓ Model: {FEATURE_TYPE.upper()}")
    print(f"✓ Accuracy: {checkpoint.get('test_acc', checkpoint.get('val_acc', 0))*100:.1f}%")
    print(f"✓ Classes: {', '.join([c.replace('_', ' ').title() for c in label_encoder.classes_])}")
    print(f"✓ Ensemble: ENABLED (robust external audio)")
    print(f"✓ Server: http://127.0.0.1:5000")
    print("="*80)
    print("\n📝 Features:")
    print("  • Robust preprocessing (matches training)")
    print("  • Ensemble inference (5+ augmentations)")
    print("  • Better external audio handling")
    print("  • Cuisine recommendations")
    print("\n🛑 Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
