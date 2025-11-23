"""
Robust data augmentation for accent classification training.
Improves model generalization by creating diverse training samples.
"""

import numpy as np
import librosa
import torch
import torchaudio
import random
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioAugmentator:
    """
    Audio data augmentation for accent classification training.
    Implements various transformations to improve model generalization.
    """
    
    def __init__(self, sr: int = 16000, aug_prob: float = 0.8):
        """
        Initialize augmentator.
        
        Args:
            sr: Target sample rate
            aug_prob: Probability of applying augmentation (0.0 to 1.0)
        """
        self.sr = sr
        self.aug_prob = aug_prob
        
        # Augmentation parameters
        self.speed_factors = [0.9, 0.95, 1.05, 1.1]  # Speed perturbation
        self.pitch_shifts = [-2, -1, 1, 2]  # Semitone shifts
        self.noise_levels = [0.005, 0.01, 0.015]  # Background noise levels
        self.gain_factors = [0.8, 0.9, 1.1, 1.2]  # Volume changes
    
    def speed_perturbation(self, audio: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Apply speed perturbation (time stretching without pitch change).
        
        Args:
            audio: Input audio array
            factor: Speed factor (None for random)
        
        Returns:
            Augmented audio
        """
        if factor is None:
            factor = random.choice(self.speed_factors)
        
        try:
            # Use librosa for high-quality time stretching
            augmented = librosa.effects.time_stretch(audio, rate=factor)
            return augmented
        except Exception:
            # Fallback: simple resampling
            target_length = int(len(audio) / factor)
            return np.interp(
                np.linspace(0, len(audio) - 1, target_length),
                np.arange(len(audio)),
                audio
            )
    
    def pitch_shift(self, audio: np.ndarray, n_steps: Optional[int] = None) -> np.ndarray:
        """
        Apply pitch shifting.
        
        Args:
            audio: Input audio array
            n_steps: Number of semitones (None for random)
        
        Returns:
            Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = random.choice(self.pitch_shifts)
        
        try:
            augmented = librosa.effects.pitch_shift(
                audio, sr=self.sr, n_steps=n_steps
            )
            return augmented
        except Exception:
            # Fallback: no change
            return audio
    
    def add_noise(self, audio: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        """
        Add background noise.
        
        Args:
            audio: Input audio array
            noise_level: Noise level (None for random)
        
        Returns:
            Noisy audio
        """
        if noise_level is None:
            noise_level = random.choice(self.noise_levels)
        
        # Generate white noise
        noise = np.random.randn(len(audio)) * noise_level
        
        # Add noise and renormalize
        augmented = audio + noise
        
        # Prevent clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val
        
        return augmented
    
    def gain_change(self, audio: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Apply gain (volume) changes.
        
        Args:
            audio: Input audio array
            factor: Gain factor (None for random)
        
        Returns:
            Gain-adjusted audio
        """
        if factor is None:
            factor = random.choice(self.gain_factors)
        
        augmented = audio * factor
        
        # Prevent clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val
        
        return augmented
    
    def frequency_masking(self, spectrogram: np.ndarray, 
                         max_mask_width: int = 10) -> np.ndarray:
        """
        Apply frequency masking to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (freq x time)
            max_mask_width: Maximum mask width in frequency bins
        
        Returns:
            Masked spectrogram
        """
        spec_aug = spectrogram.copy()
        n_freq_bins = spec_aug.shape[0]
        
        # Random mask width and position
        mask_width = random.randint(1, min(max_mask_width, n_freq_bins // 4))
        mask_start = random.randint(0, n_freq_bins - mask_width)
        
        # Apply mask (set to mean value)
        mask_value = np.mean(spec_aug)
        spec_aug[mask_start:mask_start + mask_width, :] = mask_value
        
        return spec_aug
    
    def time_masking(self, spectrogram: np.ndarray, 
                    max_mask_width: int = 20) -> np.ndarray:
        """
        Apply time masking to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (freq x time)
            max_mask_width: Maximum mask width in time frames
        
        Returns:
            Masked spectrogram
        """
        spec_aug = spectrogram.copy()
        n_time_frames = spec_aug.shape[1]
        
        # Random mask width and position
        mask_width = random.randint(1, min(max_mask_width, n_time_frames // 4))
        mask_start = random.randint(0, n_time_frames - mask_width)
        
        # Apply mask (set to mean value)
        mask_value = np.mean(spec_aug)
        spec_aug[:, mask_start:mask_start + mask_width] = mask_value
        
        return spec_aug
    
    def augment_audio(self, audio: np.ndarray, 
                     augmentation_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply random augmentation to audio.
        
        Args:
            audio: Input audio array
            augmentation_types: List of augmentation types to choose from
                              (None for all available)
        
        Returns:
            Augmented audio
        """
        # Skip augmentation based on probability
        if random.random() > self.aug_prob:
            return audio
        
        available_augs = {
            'speed': self.speed_perturbation,
            'pitch': self.pitch_shift,
            'noise': self.add_noise,
            'gain': self.gain_change
        }
        
        if augmentation_types is None:
            augmentation_types = list(available_augs.keys())
        
        # Apply random augmentation
        aug_type = random.choice(augmentation_types)
        augmented = available_augs[aug_type](audio)
        
        return augmented
    
    def multiple_augmentations(self, audio: np.ndarray, 
                             n_augmentations: int = 3) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of the same audio.
        
        Args:
            audio: Input audio array
            n_augmentations: Number of augmentations to generate
        
        Returns:
            List of augmented audio arrays (including original)
        """
        augmented_list = [audio]  # Include original
        
        for _ in range(n_augmentations):
            aug_audio = self.augment_audio(audio)
            augmented_list.append(aug_audio)
        
        return augmented_list


class HuBERTAugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies augmentation to HuBERT features during training.
    """
    
    def __init__(self, base_dataset, augmentator: AudioAugmentator,
                 augment_prob: float = 0.5, max_augmentations: int = 2):
        """
        Initialize augmented dataset.
        
        Args:
            base_dataset: Original dataset
            augmentator: AudioAugmentator instance
            augment_prob: Probability of augmentation per sample
            max_augmentations: Maximum augmentations per sample
        """
        self.base_dataset = base_dataset
        self.augmentator = augmentator
        self.augment_prob = augment_prob
        self.max_augmentations = max_augmentations
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        features, label, speaker_id = self.base_dataset[idx]
        
        # Apply feature-level augmentation (if features allow it)
        if random.random() < self.augment_prob:
            features = self._augment_features(features)
        
        return features, label, speaker_id
    
    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to extracted features.
        
        Args:
            features: Input feature tensor
        
        Returns:
            Augmented feature tensor
        """
        # Convert to numpy for processing
        features_np = features.numpy()
        
        # For HuBERT features (mean+std of embeddings), we can apply:
        # 1. Gaussian noise
        # 2. Feature dropout (randomly zero some dimensions)
        # 3. Feature scaling
        
        augmented = features_np.copy()
        
        # Gaussian noise (small amount)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented += noise
        
        # Feature dropout (randomly zero 5-10% of features)
        if random.random() < 0.3:
            dropout_mask = np.random.random(augmented.shape) < 0.05
            augmented[dropout_mask] = 0
        
        # Feature scaling (slight variations)
        if random.random() < 0.3:
            scale_factor = np.random.uniform(0.95, 1.05)
            augmented *= scale_factor
        
        return torch.from_numpy(augmented.astype(np.float32))


def create_augmented_dataset(original_dataset, sr: int = 16000, 
                           augment_prob: float = 0.5) -> HuBERTAugmentedDataset:
    """
    Create an augmented version of the dataset for training.
    
    Args:
        original_dataset: Original dataset
        sr: Sample rate for audio processing
        augment_prob: Probability of augmentation
    
    Returns:
        Augmented dataset wrapper
    """
    augmentator = AudioAugmentator(sr=sr, aug_prob=augment_prob)
    augmented_dataset = HuBERTAugmentedDataset(
        original_dataset, 
        augmentator, 
        augment_prob=augment_prob
    )
    
    print(f"✓ Created augmented dataset wrapper")
    print(f"  Augmentation probability: {augment_prob}")
    print(f"  Base dataset size: {len(original_dataset)}")
    
    return augmented_dataset


if __name__ == "__main__":
    # Test augmentation
    print("Testing audio augmentation...")
    
    # Create dummy audio
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Initialize augmentator
    augmentator = AudioAugmentator(sr=sr)
    
    # Test different augmentations
    print(f"Original audio shape: {audio.shape}")
    
    # Speed perturbation
    speed_aug = augmentator.speed_perturbation(audio, factor=1.1)
    print(f"Speed augmented shape: {speed_aug.shape}")
    
    # Pitch shift
    pitch_aug = augmentator.pitch_shift(audio, n_steps=2)
    print(f"Pitch shifted shape: {pitch_aug.shape}")
    
    # Add noise
    noise_aug = augmentator.add_noise(audio, noise_level=0.01)
    print(f"Noise augmented shape: {noise_aug.shape}")
    
    # Generate multiple augmentations
    multi_augs = augmentator.multiple_augmentations(audio, n_augmentations=3)
    print(f"Multiple augmentations: {len(multi_augs)} versions")
    
    print("✓ Audio augmentation test completed successfully!")