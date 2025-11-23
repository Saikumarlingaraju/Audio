"""
Enhanced Flask Web App - Native Language Identification with Robust Predictions
Uses ensemble prediction and test-time augmentation for better generalization
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
import traceback
from typing import Dict, Any, Optional

from src.models.robust_mlp import RobustMLPClassifier
from src.features.robust_extraction import RobustFeatureExtractor, create_robust_feature_extractor
from src.utils.robust_prediction import RobustPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration
USE_ENSEMBLE = True  # Enable ensemble prediction for better accuracy
USE_ROBUST_MODEL = True  # Use robust model if available

# Model paths
MODEL_PATH = root_dir / 'experiments' / 'indian_accents_robust' / 'best_model.pt'
ENCODER_PATH = root_dir / 'experiments' / 'indian_accents_robust' / 'label_encoder.pkl'
FALLBACK_MODEL_PATH = root_dir / 'experiments' / 'indian_accents_hubert' / 'best_model.pt'
FALLBACK_ENCODER_PATH = root_dir / 'experiments' / 'indian_accents_hubert' / 'label_encoder.pkl'

# Global variables
model = None
label_encoder = None
robust_predictor = None
feature_extractor = None
model_info = {}

def load_model_and_encoder():
    """Load model and label encoder with fallback options."""
    global model, label_encoder, robust_predictor, feature_extractor, model_info
    
    # Try to load robust model first
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    encoder_path = ENCODER_PATH if ENCODER_PATH.exists() else FALLBACK_ENCODER_PATH
    
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("No trained model found. Please train a model first.")
    
    print(f"Loading model from: {model_path}")
    print(f"Loading encoder from: {encoder_path}")
    
    # Load label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine model type and input dimension
    input_dim = checkpoint.get('input_dim', 1536)
    num_classes = len(label_encoder.classes_)
    
    # Create model (try robust model first, fallback to regular MLP)
    try:
        if USE_ROBUST_MODEL and 'model_type' in checkpoint:
            from src.models.robust_mlp import create_robust_model
            model = create_robust_model(
                input_dim=input_dim,
                num_classes=num_classes,
                model_type=checkpoint.get('model_type', 'robust_mlp'),
                hidden_dims=checkpoint.get('hidden_dims', [256, 128]),
                dropout=checkpoint.get('dropout', 0.5)
            )
        else:
            # Fallback to regular MLP
            from src.models.mlp import MLPClassifier
            model = MLPClassifier(
                input_dim=input_dim, 
                hidden_dims=[256, 128], 
                num_classes=num_classes, 
                dropout=0.3
            )
    except Exception as e:
        print(f"Could not load robust model: {e}, falling back to regular MLP")
        from src.models.mlp import MLPClassifier
        model = MLPClassifier(
            input_dim=input_dim, 
            hidden_dims=[256, 128], 
            num_classes=num_classes, 
            dropout=0.3
        )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize robust predictor for ensemble prediction
    if USE_ENSEMBLE:
        try:
            robust_predictor = RobustPredictor(str(model_path), str(encoder_path))
            print("✓ Robust predictor initialized (ensemble prediction enabled)")
        except Exception as e:
            print(f"⚠️ Could not initialize robust predictor: {e}")
            print("   Falling back to simple prediction")
            robust_predictor = None
    
    # Initialize robust feature extractor
    feature_extractor = create_robust_feature_extractor(
        feature_type='hubert',
        normalize=True
    )
    
    # Store model info
    model_info = {
        'test_accuracy': checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A')),
        'model_type': 'Robust MLP' if USE_ROBUST_MODEL else 'Standard MLP',
        'classes': list(label_encoder.classes_),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'ensemble_enabled': robust_predictor is not None
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  Type: {model_info['model_type']}")
    print(f"  Test accuracy: {model_info['test_accuracy']}")
    print(f"  Classes: {model_info['classes']}")
    print(f"  Ensemble prediction: {'Enabled' if model_info['ensemble_enabled'] else 'Disabled'}")

def get_cuisine_recommendations(state: str) -> list:
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

def predict_accent_robust(audio_path: str) -> Dict[str, Any]:
    """
    Predict accent using robust ensemble method.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Prediction result dictionary
    """
    try:
        if robust_predictor is not None:
            # Use ensemble prediction
            print("Using ensemble prediction...")
            result = robust_predictor.predict_ensemble(
                audio_path, 
                n_augmentations='auto'
            )
            
            predicted_accent = result['prediction']
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
            recommendations = get_cuisine_recommendations(predicted_accent)
            
            return {
                'success': True,
                'predicted_accent': predicted_accent.replace('_', ' ').title(),
                'confidence': float(confidence),
                'top_predictions': top_predictions,
                'recommendations': recommendations,
                'method': 'ensemble',
                'n_augmentations': result.get('n_augmentations', 1),
                'vote_distribution': result.get('vote_distribution', {})
            }
        else:
            # Fallback to simple prediction
            return predict_accent_simple(audio_path)
    
    except Exception as e:
        print(f"Ensemble prediction failed: {e}")
        traceback.print_exc()
        return predict_accent_simple(audio_path)

def predict_accent_simple(audio_path: str) -> Dict[str, Any]:
    """
    Simple prediction fallback method.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Prediction result dictionary
    """
    try:
        print("Using simple prediction...")
        
        # Extract features
        features = feature_extractor.extract_features(audio_path)
        if features is None:
            return {'success': False, 'error': 'Feature extraction failed'}
        
        # Ensure correct dimensions
        expected_dim = model_info['input_dim']
        if features.size != expected_dim:
            print(f"Warning: resizing feature vector from {features.size} to {expected_dim}")
            if features.size < expected_dim:
                features = np.pad(features, (0, expected_dim - features.size))
            else:
                features = features[:expected_dim]
        
        # Predict
        features_tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        # Get prediction
        idx = torch.argmax(probs).item()
        predicted_accent = str(label_encoder.classes_[idx])
        confidence = probs[idx].item() * 100
        
        # Get top predictions
        top_k = min(3, len(label_encoder.classes_))
        top_probs, top_indices = torch.topk(probs, top_k)
        top_predictions = [
            {
                'accent': str(label_encoder.classes_[i.item()]).replace('_', ' ').title(),
                'confidence': p.item() * 100
            }
            for p, i in zip(top_probs, top_indices)
        ]
        
        # Get cuisine recommendations
        recommendations = get_cuisine_recommendations(predicted_accent)
        
        return {
            'success': True,
            'predicted_accent': predicted_accent.replace('_', ' ').title(),
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'recommendations': recommendations,
            'method': 'simple'
        }
    
    except Exception as e:
        print(f"Simple prediction failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': f'Prediction failed: {str(e)}'}

# Initialize model on startup
try:
    load_model_and_encoder()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train a model first using the training script.")

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file uploaded'})
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
        audio_file.save(filepath)
        
        # Make prediction using robust method
        result = predict_accent_robust(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/info')
def info():
    """Display model information"""
    return jsonify({
        'model': model_info.get('model_type', 'Unknown'),
        'test_accuracy': model_info.get('test_accuracy', 'N/A'),
        'classes': model_info.get('classes', []),
        'num_classes': model_info.get('num_classes', 0),
        'input_dim': model_info.get('input_dim', 0),
        'ensemble_enabled': model_info.get('ensemble_enabled', False),
        'features': 'HuBERT Layer 12 mean+std (1536-dimensional)',
        'dataset': 'Indian Accents Dataset',
        'robustness_features': [
            'Ensemble prediction with test-time augmentation',
            'Robust audio preprocessing',
            'Domain-adapted feature extraction',
            'Multiple regularization techniques'
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None,
        'ensemble_available': robust_predictor is not None
    })

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌐 ROBUST NATIVE LANGUAGE IDENTIFICATION - INDIAN ACCENTS")
    print("=" * 80)
    
    if model is not None:
        print(f"✓ Model: {model_info.get('model_type', 'Unknown')}")
        
        try:
            acc = model_info.get('test_accuracy', 'N/A')
            if isinstance(acc, (int, float)):
                print(f"✓ Test Accuracy: {acc:.2f}%")
            else:
                print(f"✓ Test Accuracy: {acc}")
        except:
            print(f"✓ Test Accuracy: {model_info.get('test_accuracy', 'N/A')}")
            
        print(f"✓ Classes: {', '.join([c.replace('_', ' ').title() for c in model_info.get('classes', [])])}")
        print(f"✓ Ensemble Prediction: {'Enabled' if model_info.get('ensemble_enabled') else 'Disabled'}")
    else:
        print("⚠️ No model loaded - please train a model first")
    
    print(f"✓ Server starting on: http://127.0.0.1:5000")
    print("=" * 80)
    print("\n📝 Usage:")
    print("  1. Open http://127.0.0.1:5000 in your browser")
    print("  2. Upload an audio file (WAV, MP3, etc.)")
    print("  3. Get robust accent prediction with confidence scores")
    print("\n🚀 Enhanced Features:")
    print("  • Ensemble prediction for better accuracy on unseen data")
    print("  • Test-time augmentation for robustness")
    print("  • Robust audio preprocessing")
    print("  • Domain adaptation techniques")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)