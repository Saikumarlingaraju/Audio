"""
Robust prediction for external audio using ensemble inference and test-time augmentation
Fixes domain shift issues between training and external recordings
"""
import sys
from pathlib import Path
import numpy as np
import torch
import librosa
import pickle
from collections import Counter

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from src.features.hubert_extractor import HuBERTExtractor
from src.models.mlp import MLPClassifier

class RobustPredictor:
    """
    Robust accent predictor with ensemble inference and test-time augmentation
    """
    
    def __init__(self, model_path, label_encoder_path):
        """
        Initialize robust predictor
        
        Args:
            model_path: Path to trained model checkpoint
            label_encoder_path: Path to label encoder pickle
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_dim = checkpoint.get('input_dim', 1536)
        self.num_classes = checkpoint.get('num_classes', len(self.label_encoder.classes_))
        
        self.model = MLPClassifier(
            input_dim=self.input_dim,
            hidden_dims=[256, 128],
            num_classes=self.num_classes,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize HuBERT extractor
        self.extractor = HuBERTExtractor()
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
    
    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Robust audio preprocessing matching training pipeline
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (must match training)
        
        Returns:
            Preprocessed audio array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Trim leading/trailing silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize amplitude
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Ensure minimum length (at least 0.5 seconds)
        min_length = int(0.5 * target_sr)
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        # Limit maximum length (max 10 seconds)
        max_length = int(10 * target_sr)
        if len(audio) > max_length:
            audio = audio[:max_length]
        
        return audio
    
    def augment_audio(self, audio, sr=16000):
        """
        Generate augmented versions of audio for test-time augmentation
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            List of augmented audio arrays
        """
        augmented = [audio]  # Original
        
        # 1. Speed perturbation (0.9x and 1.1x)
        for rate in [0.9, 1.1]:
            aug = librosa.effects.time_stretch(audio, rate=rate)
            augmented.append(aug)
        
        # 2. Pitch shift (+/- 2 semitones)
        for n_steps in [-2, 2]:
            aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            augmented.append(aug)
        
        # 3. Add light background noise
        for noise_level in [0.005, 0.01]:
            noise = np.random.randn(len(audio)) * noise_level
            aug = audio + noise
            aug = aug / (np.max(np.abs(aug)) + 1e-8)  # Renormalize
            augmented.append(aug)
        
        return augmented
    
    def extract_features(self, audio, sr=16000):
        """
        Extract HuBERT features matching training pipeline
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            Feature vector (1536-dim: mean + std)
        """
        import tempfile
        import soundfile as sf
        
        # Save to temporary file for HuBERT extractor
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sr)
        
        try:
            # Extract mean pooled embeddings
            pooled = self.extractor.extract_from_file(temp_path, layer=12, pooling='mean')
            mean_vec = pooled['embeddings']['layer_12']
            
            # Extract std from frame-level embeddings
            frame_level = self.extractor.extract_from_file(temp_path, layer=12, pooling=None)
            frames = frame_level['embeddings']['layer_12']
            std_vec = torch.from_numpy(frames.std(axis=0))
            
            # Concatenate mean and std
            features = torch.cat([mean_vec, std_vec]).numpy()
            
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
        
        return features
    
    def predict_single(self, features):
        """
        Single prediction from features
        
        Args:
            features: Feature vector
        
        Returns:
            (predicted_label, probabilities)
        """
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            pred_label = self.label_encoder.classes_[pred_idx]
        
        return pred_label, probs
    
    def predict_ensemble(self, audio_path, n_augmentations='auto', use_majority_vote=True):
        """
        Robust prediction with ensemble inference and test-time augmentation
        
        Args:
            audio_path: Path to audio file
            n_augmentations: Number of augmentations ('auto' or int)
            use_majority_vote: Use majority voting instead of probability averaging
        
        Returns:
            Dictionary with prediction results and confidence
        """
        print(f"\nProcessing: {audio_path}")
        
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        print(f"✓ Preprocessed audio: {len(audio)/16000:.2f}s")
        
        # Generate augmented versions
        if n_augmentations == 'auto':
            augmented_audios = self.augment_audio(audio)
        elif isinstance(n_augmentations, int):
            augmented_audios = self.augment_audio(audio)[:n_augmentations+1]
        else:
            augmented_audios = [audio]
        
        print(f"✓ Generated {len(augmented_audios)} augmented versions")
        
        # Extract features and predict for each augmentation
        all_predictions = []
        all_probs = []
        
        for i, aug_audio in enumerate(augmented_audios):
            try:
                features = self.extract_features(aug_audio)
                pred_label, probs = self.predict_single(features)
                all_predictions.append(pred_label)
                all_probs.append(probs)
            except Exception as e:
                print(f"  ⚠️ Augmentation {i} failed: {e}")
                continue
        
        if not all_predictions:
            raise RuntimeError("All augmentations failed to produce predictions")
        
        # Ensemble decision
        if use_majority_vote:
            # Majority voting
            vote_counts = Counter(all_predictions)
            final_prediction = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[final_prediction] / len(all_predictions)
        else:
            # Average probabilities
            avg_probs = np.mean(all_probs, axis=0)
            final_prediction = self.label_encoder.classes_[np.argmax(avg_probs)]
            confidence = np.max(avg_probs)
        
        # Calculate per-class probabilities
        avg_probs = np.mean(all_probs, axis=0)
        class_probs = {class_name: float(prob) 
                      for class_name, prob in zip(self.label_encoder.classes_, avg_probs)}
        
        # Vote distribution
        vote_counts = Counter(all_predictions)
        vote_distribution = {class_name: vote_counts.get(class_name, 0) / len(all_predictions)
                            for class_name in self.label_encoder.classes_}
        
        print(f"\n✓ Ensemble prediction: {final_prediction}")
        print(f"  Confidence: {confidence*100:.2f}%")
        print(f"  Votes: {dict(vote_counts)}")
        
        return {
            'prediction': final_prediction,
            'confidence': float(confidence),
            'class_probabilities': class_probs,
            'vote_distribution': vote_distribution,
            'n_augmentations': len(all_predictions),
            'all_predictions': all_predictions
        }
    
    def predict_simple(self, audio_path):
        """
        Simple prediction without ensemble (for comparison)
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with prediction results
        """
        audio = self.preprocess_audio(audio_path)
        features = self.extract_features(audio)
        pred_label, probs = self.predict_single(features)
        
        class_probs = {class_name: float(prob) 
                      for class_name, prob in zip(self.label_encoder.classes_, probs)}
        
        return {
            'prediction': pred_label,
            'confidence': float(np.max(probs)),
            'class_probabilities': class_probs
        }


def test_robust_prediction():
    """Test robust prediction on sample audio"""
    print(f"\n{'='*80}")
    print("Testing Robust Prediction")
    print(f"{'='*80}\n")
    
    # Find best model
    experiments_dir = Path('experiments')
    model_dirs = [d for d in experiments_dir.iterdir() 
                 if d.is_dir() and (d / 'best_model.pt').exists()]
    
    if not model_dirs:
        print("❌ No trained models found. Train a model first.")
        return
    
    # Use MLP model (or find best performing one)
    model_dir = experiments_dir / 'indian_accents_mlp'
    if not model_dir.exists():
        model_dir = model_dirs[0]
    
    model_path = model_dir / 'best_model.pt'
    label_encoder_path = model_dir / 'label_encoder.pkl'
    
    # Initialize predictor
    predictor = RobustPredictor(str(model_path), str(label_encoder_path))
    
    # Test on a sample from test set
    test_csv = Path('data/splits/test.csv')
    if test_csv.exists():
        import pandas as pd
        test_df = pd.read_csv(test_csv)
        sample = test_df.iloc[0]
        audio_path = Path('data/raw/indian_accents') / sample['filepath']
        
        if audio_path.exists():
            print("\nTest 1: Simple prediction (no ensemble)")
            simple_result = predictor.predict_simple(str(audio_path))
            print(f"Prediction: {simple_result['prediction']}")
            print(f"Confidence: {simple_result['confidence']*100:.2f}%")
            print(f"True label: {sample['native_language']}")
            
            print("\n" + "="*80)
            print("Test 2: Ensemble prediction with test-time augmentation")
            ensemble_result = predictor.predict_ensemble(str(audio_path))
            print(f"True label: {sample['native_language']}")
            
            # Compare
            print("\n" + "="*80)
            print("Comparison:")
            print(f"Simple:   {simple_result['prediction']} ({simple_result['confidence']*100:.1f}%)")
            print(f"Ensemble: {ensemble_result['prediction']} ({ensemble_result['confidence']*100:.1f}%)")
            print(f"Correct:  {sample['native_language']}")
    
    print(f"\n{'='*80}")
    print("✓ Robust prediction module ready for use")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    test_robust_prediction()
