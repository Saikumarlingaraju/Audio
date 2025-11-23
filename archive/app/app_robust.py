"""
Update Flask app to use augmented model with better preprocessing
"""
import sys
from pathlib import Path

# Add parent directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from flask import Flask, render_template, request, jsonify
import torch
import pickle
import numpy as np
import librosa
import os

from src.models.mlp import MLPClassifier
from src.features.mfcc_extractor import MFCCExtractor
from src.features.hubert_extractor import HuBERTExtractor
from src.utils.robust_prediction import RobustPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['USE_ENSEMBLE'] = True  # Enable ensemble inference for external audio

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Prefer HuBERT primary model; fallback to MFCC model
HUBERT_MODEL_DIR = root_dir / 'experiments' / 'indian_accents_hubert'
MFCC_MODEL_DIR = root_dir / 'experiments' / 'indian_accents_mlp'

if HUBERT_MODEL_DIR.exists() and (HUBERT_MODEL_DIR / 'best_model.pt').exists():
    MODEL_DIR = HUBERT_MODEL_DIR
    PRIMARY_FEATURE_TYPE = 'hubert'
    print("✅ Using HuBERT primary model")
else:
    MODEL_DIR = MFCC_MODEL_DIR
    PRIMARY_FEATURE_TYPE = 'mfcc'
    print("✅ HuBERT model not found - falling back to MFCC model")

MODEL_PATH = MODEL_DIR / 'best_model.pt'
ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'

print("Loading model...")
checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')
with open(str(ENCODER_PATH), 'rb') as f:
    label_encoder = pickle.load(f)

if PRIMARY_FEATURE_TYPE == 'hubert':
    inferred_dim = checkpoint.get('input_dim', 1536)  # mean+std of 768
    model = MLPClassifier(input_dim=inferred_dim, hidden_dims=[256, 128], num_classes=6, dropout=0.3)
else:
    model = MLPClassifier(input_dim=600, hidden_dims=[256, 128], num_classes=6, dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded: Test accuracy {checkpoint['val_acc']:.2f}%")

if PRIMARY_FEATURE_TYPE == 'mfcc':
    mfcc_extractor = MFCCExtractor(
        sr=16000,
        n_mfcc=40,
        include_delta=True,
        include_delta_delta=True
    )
else:
    hubert_extractor = HuBERTExtractor()

# Initialize robust predictor for ensemble inference
robust_predictor = None
if app.config['USE_ENSEMBLE']:
    try:
        robust_predictor = RobustPredictor(str(MODEL_PATH), str(ENCODER_PATH))
        print("✓ Robust predictor initialized (ensemble inference enabled)")
    except Exception as e:
        print(f"⚠️  Could not initialize robust predictor: {e}")
        print("   Falling back to simple prediction")

print(f"✓ Model loaded: Test accuracy {checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A')):.2f}%")
print(f"✓ Classes: {label_encoder.classes_}")

def get_cuisine_recommendations(state):
    """Get cuisine recommendations for a given state"""
    cuisine_map = {
        'andhra_pradesh': [
            {
                'name': 'Hyderabadi Biryani',
                'region': 'Andhra Pradesh',
                'description': 'A royal and aromatic rice dish made with basmati rice, tender meat, and exotic spices, slow-cooked to perfection.'
            },
            {
                'name': 'Pesarattu',
                'region': 'Andhra Pradesh',
                'description': 'A protein-rich crepe made from green gram (moong dal), often served with ginger chutney for breakfast.'
            },
            {
                'name': 'Gongura Mutton',
                'region': 'Andhra Pradesh',
                'description': 'A tangy and spicy curry made with tender mutton and sorrel leaves (gongura), a signature dish of the region.'
            }
        ],
        'gujrat': [
            {
                'name': 'Dhokla',
                'region': 'Gujarat',
                'description': 'A steamed savory cake made from fermented rice and chickpea batter, light and fluffy with a tangy-sweet flavor.'
            },
            {
                'name': 'Gujarati Thali',
                'region': 'Gujarat',
                'description': 'A traditional platter featuring a variety of sweet and savory dishes including dal, kadhi, vegetable curries, roti, rice, and sweets.'
            },
            {
                'name': 'Khandvi',
                'region': 'Gujarat',
                'description': 'Delicate rolls made from gram flour and yogurt, seasoned with mustard seeds and curry leaves - a perfect snack.'
            }
        ],
        'jharkhand': [
            {
                'name': 'Litti Chokha',
                'region': 'Jharkhand',
                'description': 'Roasted wheat balls stuffed with sattu (roasted gram flour) served with mashed vegetables - a rustic and flavorful dish.'
            },
            {
                'name': 'Dhuska',
                'region': 'Jharkhand',
                'description': 'Deep-fried bread made from rice and lentil batter, crispy outside and soft inside, often served with chutney or curry.'
            },
            {
                'name': 'Rugra',
                'region': 'Jharkhand',
                'description': 'A traditional curry made with mushrooms found in the forests of Jharkhand, cooked with local spices.'
            }
        ],
        'karnataka': [
            {
                'name': 'Bisi Bele Bath',
                'region': 'Karnataka',
                'description': 'A comforting one-pot meal of rice, lentils, vegetables, and aromatic spices - the ultimate comfort food from Karnataka.'
            },
            {
                'name': 'Mysore Masala Dosa',
                'region': 'Karnataka',
                'description': 'A crispy rice crepe spread with spicy red chutney and filled with potato masala - a breakfast favorite.'
            },
            {
                'name': 'Ragi Mudde',
                'region': 'Karnataka',
                'description': 'Nutritious finger millet balls served with sambar or spicy curries, a staple food in rural Karnataka.'
            }
        ],
        'kerala': [
            {
                'name': 'Appam with Stew',
                'region': 'Kerala',
                'description': 'Soft, lacy rice pancakes with crispy edges served with a mildly spiced coconut milk-based vegetable or meat stew.'
            },
            {
                'name': 'Kerala Fish Curry',
                'region': 'Kerala',
                'description': 'Fresh fish cooked in a tangy tamarind and coconut-based gravy with curry leaves - bursting with coastal flavors.'
            },
            {
                'name': 'Puttu and Kadala',
                'region': 'Kerala',
                'description': 'Steamed cylinders of ground rice and coconut served with spicy black chickpea curry - a traditional breakfast.'
            }
        ],
        'tamil': [
            {
                'name': 'Chettinad Chicken',
                'region': 'Tamil Nadu',
                'description': 'A fiery and aromatic chicken curry from the Chettinad region, made with freshly ground spices and black pepper.'
            },
            {
                'name': 'Idli Sambar',
                'region': 'Tamil Nadu',
                'description': 'Soft, fluffy steamed rice cakes served with a flavorful lentil-based vegetable stew - a South Indian classic.'
            },
            {
                'name': 'Pongal',
                'region': 'Tamil Nadu',
                'description': 'A savory rice and lentil dish tempered with black pepper, cumin, and ghee - comfort food at its best.'
            }
        ]
    }
    
    return cuisine_map.get(state, [])

def preprocess_audio(audio, sr, target_duration=5.0):
    """
    Preprocess audio to match training data characteristics
    - Normalize audio
    - Trim silence
    - Pad/truncate to target duration
    """
    # Trim silence from beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize audio to [-1, 1]
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max()
    
    # Target length in samples
    target_length = int(target_duration * sr)
    
    # Pad or truncate
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        # Take center portion
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    
    return audio

def enhance_audio_quality(audio, sr):
    """
    Apply audio enhancements to match training data characteristics.
    This helps external recordings work better.
    """
    # 1. Trim silence from start/end (helps with phone recordings)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    
    # 2. Normalize amplitude to standard range
    audio_normalized = librosa.util.normalize(audio_trimmed)
    
    # 3. Apply light noise reduction using spectral gating
    # Compute magnitude spectrogram
    S = np.abs(librosa.stft(audio_normalized))
    
    # Estimate noise floor from quiet sections (bottom 10%)
    noise_floor = np.percentile(S, 10, axis=1, keepdims=True)
    
    # Apply soft gating (reduce but don't eliminate low-energy components)
    S_filtered = np.maximum(S - 0.5 * noise_floor, 0)
    
    # Reconstruct audio
    audio_clean = librosa.istft(S_filtered * np.exp(1j * np.angle(librosa.stft(audio_normalized))))
    
    # 4. Apply gentle high-pass filter to remove low-frequency rumble
    audio_filtered = librosa.effects.preemphasis(audio_clean, coef=0.97)
    
    return audio_filtered

def extract_features(audio_path):
    """Extract features for inference based on PRIMARY_FEATURE_TYPE."""
    try:
        if PRIMARY_FEATURE_TYPE == 'hubert':
            # Mean pooled layer 12 + std over frames
            pooled = hubert_extractor.extract_from_file(str(audio_path), extract_layer=12, pooling='mean')
            if pooled is None:
                raise RuntimeError("HuBERT pooled extraction failed")
            mean_vec = pooled['embeddings']['layer_12']  # (768,)
            frame_level = hubert_extractor.extract_from_file(str(audio_path), extract_layer=12, pooling=None)
            if frame_level and 'layer_12' in frame_level['embeddings']:
                frames = frame_level['embeddings']['layer_12']  # (T, 768)
                std_vec = frames.std(axis=0)
            else:
                std_vec = np.zeros_like(mean_vec)
            return np.concatenate([mean_vec, std_vec])  # (1536,)
        else:
            result = mfcc_extractor.extract_from_file(str(audio_path), return_frames=False)
            if isinstance(result, dict) and 'features' in result:
                return result['features']
            audio, sr = librosa.load(str(audio_path), sr=16000)
            mfcc_features = mfcc_extractor.extract_mfcc(audio)
            features = mfcc_extractor.compute_statistics(mfcc_features)
            return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        raise

def predict_accent(audio_path, use_ensemble=None):
    """
    Predict accent from audio file
    
    Args:
        audio_path: Path to audio file
        use_ensemble: Override app config to force ensemble on/off (None = use config)
    """
    try:
        # Determine whether to use ensemble
        should_use_ensemble = app.config['USE_ENSEMBLE'] if use_ensemble is None else use_ensemble
        
        # Use robust predictor with ensemble if available
        if should_use_ensemble and robust_predictor is not None:
            result = robust_predictor.predict_ensemble(audio_path, n_augmentations='auto')
            
            # Convert to app format
            predicted_label = result['prediction']
            confidence = result['confidence'] * 100
            
            # Build top predictions from class probabilities
            top_predictions = []
            for accent, prob in sorted(result['class_probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True):
                top_predictions.append({
                    'accent': accent.replace('_', ' ').title(),
                    'confidence': float(prob * 100)
                })
            
            # Get cuisine recommendations
            recommendations = get_cuisine_recommendations(str(predicted_label))
            
            return {
                'accent': predicted_label.replace('_', ' ').title(),
                'confidence': float(confidence),
                'top_predictions': top_predictions,
                'recommendations': recommendations,
                'method': 'ensemble',
                'n_augmentations': result['n_augmentations']
            }
        
        # Fallback to simple prediction
        # Extract features
        features = extract_features(audio_path)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item() * 100
        
        # Get label
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        
        # Get all predictions sorted by confidence
        all_probs = probabilities.numpy()
        sorted_indices = np.argsort(all_probs)[::-1]
        
        top_predictions = []
        for idx in sorted_indices:
            accent = label_encoder.inverse_transform([idx])[0]
            conf = all_probs[idx] * 100
            top_predictions.append({
                'accent': accent.replace('_', ' ').title(),
                'confidence': float(conf)
            })
        
        # Get cuisine recommendations
        recommendations = get_cuisine_recommendations(str(predicted_label))
        
        return {
            'accent': predicted_label.replace('_', ' ').title(),
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'recommendations': recommendations,
            'method': 'simple'
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
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
        
        # Predict
        result = predict_accent(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Get model information"""
    return jsonify({
        'model': 'MLP Classifier',
        'classes': list(label_encoder.classes_),
        'test_accuracy': float(checkpoint.get('test_acc', checkpoint.get('val_acc', 0))),
        'augmented': False,
        'primary_feature_type': PRIMARY_FEATURE_TYPE,
        'feature_pipeline': 'hubert_layer12_mean+std' if PRIMARY_FEATURE_TYPE=='hubert' else 'mfcc_stats',
        'input_dim': model.layers[0].in_features,
        'model_path': str(MODEL_DIR)
    })

# Optional debug endpoint to test a local file path without uploading
@app.route('/debug_predict', methods=['GET'])
def debug_predict():
    """Predict from a server-side file path for debugging.
    Usage: /debug_predict?path=relative/or/absolute/path.wav
    """
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({'error': 'Missing query param: path'}), 400

        # Resolve relative paths against project root
        p = Path(file_path)
        if not p.is_absolute():
            p = root_dir / file_path

        if not p.exists():
            return jsonify({'error': f'File not found: {p}'}), 404

        result = predict_accent(str(p))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌐 NATIVE LANGUAGE IDENTIFICATION - INDIAN ACCENTS")
    print("="*80)
    print(f"✓ Model: MLP Classifier ({checkpoint.get('test_acc', checkpoint.get('val_acc', 0)):.2f}% test accuracy)")
    print(f"✓ Classes: {', '.join([c.replace('_', ' ').title() for c in label_encoder.classes_])}")
    print(f"✓ Clean Model: No augmentation (consistent training/inference)")
    if PRIMARY_FEATURE_TYPE == 'hubert':
        print(f"✓ Feature Pipeline: HuBERT layer12 pooled (mean+std)")
    else:
        if PRIMARY_FEATURE_TYPE == 'hubert':
            print(f"✓ Feature Pipeline: HuBERT layer12 pooled (mean+std)")
        else:
            print(f"✓ Feature Pipeline: extract_from_file (MFCC stats)")
    print(f"✓ Server starting on: http://127.0.0.1:5000")
    print("="*80)
    print("\n📝 Usage:")
    print("  1. Open http://127.0.0.1:5000 in your browser")
    print("  2. Upload an audio file (WAV, MP3, etc.)")
    print("  3. Get Indian state accent prediction with confidence scores")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
