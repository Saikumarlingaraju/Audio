"""
Robust feature extraction for accent classification.
Handles domain shift between training and inference data through
consistent preprocessing and feature normalization.
"""

import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from src.features.hubert_extractor import HuBERTExtractor


class RobustAudioPreprocessor:
    """
    Robust audio preprocessing pipeline that ensures consistent
    audio characteristics between training and inference.
    """
    
    def __init__(self, 
                 target_sr: int = 16000,
                 target_duration: Optional[float] = None,
                 normalize_amplitude: bool = True,
                 trim_silence: bool = True,
                 silence_threshold: float = 20.0):
        """
        Initialize robust audio preprocessor.
        
        Args:
            target_sr: Target sample rate
            target_duration: Target duration in seconds (None for variable)
            normalize_amplitude: Whether to normalize amplitude
            trim_silence: Whether to trim leading/trailing silence
            silence_threshold: Silence threshold in dB
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.normalize_amplitude = normalize_amplitude
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        
        # Audio quality checks
        self.min_duration = 0.5  # Minimum 0.5 seconds
        self.max_duration = 15.0  # Maximum 15 seconds
    
    def load_and_preprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Preprocessed audio array and sample rate
        """
        try:
            # Load audio with librosa (handles various formats)
            audio, sr = librosa.load(
                audio_path, 
                sr=self.target_sr, 
                mono=True,
                duration=self.max_duration
            )
            
            # Basic quality check
            if len(audio) == 0:
                raise ValueError("Audio file is empty")
            
            # Trim silence
            if self.trim_silence:
                audio, _ = librosa.effects.trim(
                    audio, 
                    top_db=self.silence_threshold,
                    frame_length=2048,
                    hop_length=512
                )
            
            # Ensure minimum duration
            min_samples = int(self.min_duration * self.target_sr)
            if len(audio) < min_samples:
                # Pad with zeros or repeat
                if len(audio) < min_samples // 2:
                    # Repeat audio if too short
                    repeat_factor = int(np.ceil(min_samples / len(audio)))
                    audio = np.tile(audio, repeat_factor)[:min_samples]
                else:
                    # Pad with zeros
                    padding = min_samples - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant')
            
            # Apply target duration if specified
            if self.target_duration is not None:
                target_samples = int(self.target_duration * self.target_sr)
                if len(audio) > target_samples:
                    # Randomly crop to maintain variability
                    start_idx = np.random.randint(0, len(audio) - target_samples + 1)
                    audio = audio[start_idx:start_idx + target_samples]
                elif len(audio) < target_samples:
                    # Pad to target length
                    padding = target_samples - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant')
            
            # Normalize amplitude
            if self.normalize_amplitude:
                audio = self._normalize_audio(audio)
            
            return audio, self.target_sr
            
        except Exception as e:
            print(f"Error preprocessing {audio_path}: {str(e)}")
            # Return silence as fallback
            fallback_duration = self.target_duration or 1.0
            fallback_samples = int(fallback_duration * self.target_sr)
            return np.zeros(fallback_samples), self.target_sr
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude consistently.
        
        Args:
            audio: Input audio array
        
        Returns:
            Normalized audio
        """
        # RMS normalization (more robust than peak normalization)
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 1e-8:  # Avoid division by zero
            # Target RMS level
            target_rms = 0.1
            audio = audio * (target_rms / rms)
        
        # Ensure no clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        
        return audio


class RobustFeatureExtractor:
    """
    Robust feature extraction that handles various input conditions
    and applies domain adaptation techniques.
    """
    
    def __init__(self, 
                 feature_type: str = 'hubert',
                 model_name: str = "facebook/hubert-base-ls960",
                 extract_layer: int = 12,
                 normalize_features: bool = True,
                 feature_stats_path: Optional[str] = None):
        """
        Initialize robust feature extractor.
        
        Args:
            feature_type: Type of features ('hubert', 'mfcc')
            model_name: HuggingFace model name (for HuBERT)
            extract_layer: Layer to extract (for HuBERT)
            normalize_features: Whether to normalize features
            feature_stats_path: Path to precomputed feature statistics
        """
        self.feature_type = feature_type
        self.extract_layer = extract_layer
        self.normalize_features = normalize_features
        
        # Initialize audio preprocessor
        self.audio_processor = RobustAudioPreprocessor()
        
        # Initialize feature extractors
        if feature_type == 'hubert':
            self.hubert_extractor = HuBERTExtractor(model_name=model_name)
        
        # Load feature statistics for normalization
        self.feature_stats = None
        if feature_stats_path and Path(feature_stats_path).exists():
            self.feature_stats = torch.load(feature_stats_path)
            print(f"✓ Loaded feature statistics from {feature_stats_path}")
    
    def extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract robust features from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Extracted features or None if failed
        """
        try:
            # Preprocess audio
            audio, sr = self.audio_processor.load_and_preprocess(audio_path)
            
            # Extract features based on type
            if self.feature_type == 'hubert':
                features = self._extract_hubert_features(audio, sr)
            elif self.feature_type == 'mfcc':
                features = self._extract_mfcc_features(audio, sr)
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
            
            # Normalize features
            if self.normalize_features and features is not None:
                features = self._normalize_features(features)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction failed for {audio_path}: {str(e)}")
            return None
    
    def _extract_hubert_features(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract HuBERT features with robust processing."""
        try:
            # Save audio temporarily for HuBERT processing
            temp_path = Path("temp_audio_for_extraction.wav")
            torchaudio.save(str(temp_path), torch.from_numpy(audio).unsqueeze(0), sr)
            
            # Extract embeddings
            result = self.hubert_extractor.extract_from_file(
                str(temp_path),
                extract_layer=self.extract_layer,
                pooling=None  # Get frame-level features
            )
            
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)
            
            if result is None:
                return None
            
            # Extract embeddings
            embeddings_dict = result.get('embeddings', {})
            
            # Get layer embeddings
            layer_key = f'layer_{self.extract_layer}'
            if layer_key in embeddings_dict:
                frames = embeddings_dict[layer_key]
            elif self.extract_layer in embeddings_dict:
                frames = embeddings_dict[self.extract_layer]
            else:
                # Use first available layer
                first_key = list(embeddings_dict.keys())[0]
                frames = embeddings_dict[first_key]
            
            frames = np.asarray(frames)
            
            # Compute robust statistics
            if frames.ndim == 1:
                # Already pooled
                mean_vec = frames.astype(np.float32)
                std_vec = np.zeros_like(mean_vec)
            else:
                # Compute mean and std with outlier handling
                mean_vec = np.nanmean(frames, axis=0).astype(np.float32)
                std_vec = np.nanstd(frames, axis=0).astype(np.float32)
                
                # Handle NaN/Inf values
                mean_vec = np.nan_to_num(mean_vec, nan=0.0, posinf=0.0, neginf=0.0)
                std_vec = np.nan_to_num(std_vec, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Concatenate mean and std for 1536-dim features
            features = np.concatenate([mean_vec, std_vec]).astype(np.float32)
            
            return features
            
        except Exception as e:
            print(f"HuBERT extraction error: {str(e)}")
            return None
    
    def _extract_mfcc_features(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract MFCC features with robust processing."""
        try:
            # Compute MFCC with error handling
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512,
                window='hann'
            )
            
            # Compute delta and delta-delta
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine all features
            all_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])  # 39 x T
            
            # Compute statistics
            mean_mfcc = np.mean(all_mfccs, axis=1)
            std_mfcc = np.std(all_mfccs, axis=1)
            
            # Additional statistics for robustness
            min_mfcc = np.min(all_mfccs, axis=1)
            max_mfcc = np.max(all_mfccs, axis=1)
            
            # Concatenate all statistics
            features = np.concatenate([
                mean_mfcc, std_mfcc, min_mfcc, max_mfcc
            ]).astype(np.float32)
            
            return features
            
        except Exception as e:
            print(f"MFCC extraction error: {str(e)}")
            return None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using training statistics or robust normalization.
        
        Args:
            features: Input features
        
        Returns:
            Normalized features
        """
        if self.feature_stats is not None:
            # Use training statistics
            mean = self.feature_stats.get('mean', 0.0)
            std = self.feature_stats.get('std', 1.0)
            
            features = (features - mean) / (std + 1e-8)
        else:
            # Robust standardization (clip outliers)
            features = np.clip(features, 
                             np.percentile(features, 1),
                             np.percentile(features, 99))
            
            # Z-score normalization
            mean = np.mean(features)
            std = np.std(features)
            features = (features - mean) / (std + 1e-8)
        
        return features


class FeatureStatisticsComputer:
    """
    Computes and stores feature statistics from training data
    for consistent normalization during inference.
    """
    
    def __init__(self, feature_extractor: RobustFeatureExtractor):
        """
        Initialize statistics computer.
        
        Args:
            feature_extractor: Feature extractor to use
        """
        self.feature_extractor = feature_extractor
        self.all_features = []
    
    def add_features_from_file(self, audio_path: str):
        """Add features from a single file."""
        features = self.feature_extractor.extract_features(audio_path)
        if features is not None:
            self.all_features.append(features)
    
    def add_features_from_dataset(self, audio_paths: list):
        """Add features from multiple files."""
        for audio_path in audio_paths:
            self.add_features_from_file(audio_path)
    
    def compute_and_save_statistics(self, output_path: str):
        """
        Compute statistics and save to file.
        
        Args:
            output_path: Path to save statistics
        """
        if not self.all_features:
            raise ValueError("No features collected")
        
        # Stack all features
        all_features_array = np.stack(self.all_features, axis=0)
        
        # Compute robust statistics
        mean = np.mean(all_features_array, axis=0)
        std = np.std(all_features_array, axis=0)
        
        # Additional statistics
        median = np.median(all_features_array, axis=0)
        q25 = np.percentile(all_features_array, 25, axis=0)
        q75 = np.percentile(all_features_array, 75, axis=0)
        
        stats = {
            'mean': mean,
            'std': std,
            'median': median,
            'q25': q25,
            'q75': q75,
            'n_samples': len(self.all_features),
            'feature_dim': all_features_array.shape[1]
        }
        
        # Save statistics
        torch.save(stats, output_path)
        print(f"✓ Feature statistics saved to {output_path}")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Feature dimension: {stats['feature_dim']}")
        
        return stats


def create_robust_feature_extractor(feature_type: str = 'hubert',
                                  normalize: bool = True,
                                  stats_path: Optional[str] = None) -> RobustFeatureExtractor:
    """
    Factory function to create robust feature extractor.
    
    Args:
        feature_type: Type of features to extract
        normalize: Whether to normalize features
        stats_path: Path to feature statistics
    
    Returns:
        Configured feature extractor
    """
    return RobustFeatureExtractor(
        feature_type=feature_type,
        normalize_features=normalize,
        feature_stats_path=stats_path
    )


if __name__ == "__main__":
    print("Testing robust feature extraction...")
    
    # Create feature extractor
    extractor = create_robust_feature_extractor(feature_type='hubert')
    
    # Test with dummy audio
    print("\n✓ Robust feature extractor created successfully!")
    print(f"  Feature type: {extractor.feature_type}")
    print(f"  Normalization: {extractor.normalize_features}")
    
    print("\n✓ Robust feature extraction module ready!")