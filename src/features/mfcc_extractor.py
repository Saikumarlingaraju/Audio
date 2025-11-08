"""
MFCC feature extraction with delta and delta-delta features.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


class MFCCExtractor:
    """Extract MFCC features from audio files."""
    
    def __init__(self, sr=16000, n_mfcc=40, n_fft=512, hop_length=160,
                 win_length=400, n_mels=80, fmin=0, fmax=8000,
                 include_delta=True, include_delta_delta=True):
        """
        Initialize MFCC extractor.
        
        Args:
            sr: Sampling rate
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window length
            n_mels: Number of Mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            include_delta: Include delta features
            include_delta_delta: Include delta-delta features
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta
        
    def extract_mfcc(self, audio):
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal (numpy array)
        
        Returns:
            dict: Dictionary containing MFCCs and statistics
        """
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        features = {'mfcc': mfcc}
        
        # Delta features (first derivative)
        if self.include_delta:
            delta = librosa.feature.delta(mfcc)
            features['delta'] = delta
        
        # Delta-delta features (second derivative)
        if self.include_delta_delta:
            delta_delta = librosa.feature.delta(mfcc, order=2)
            features['delta_delta'] = delta_delta
        
        return features
    
    def compute_statistics(self, features):
        """
        Compute statistical features from frame-level features.
        
        Args:
            features: Dictionary of features (mfcc, delta, delta_delta)
        
        Returns:
            numpy array: Concatenated statistics
        """
        stats = []
        
        for name, feat in features.items():
            # Mean
            mean = np.mean(feat, axis=1)
            # Standard deviation
            std = np.std(feat, axis=1)
            # Min
            min_val = np.min(feat, axis=1)
            # Max
            max_val = np.max(feat, axis=1)
            # Median
            median = np.median(feat, axis=1)
            
            # Concatenate all statistics
            stats.extend([mean, std, min_val, max_val, median])
        
        # Flatten and concatenate
        stats_array = np.concatenate(stats)
        
        return stats_array
    
    def extract_from_file(self, filepath, return_frames=False):
        """
        Extract MFCC features from audio file.
        
        Args:
            filepath: Path to audio file
            return_frames: If True, return frame-level features; else return statistics
        
        Returns:
            dict: Features and metadata
        """
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=self.sr)
            
            # Extract features
            features = self.extract_mfcc(audio)
            
            if return_frames:
                # Return frame-level features
                # Stack all features
                feature_list = [features['mfcc']]
                if 'delta' in features:
                    feature_list.append(features['delta'])
                if 'delta_delta' in features:
                    feature_list.append(features['delta_delta'])
                
                feature_matrix = np.vstack(feature_list)  # Shape: (n_features, n_frames)
                
                return {
                    'features': feature_matrix,
                    'n_frames': feature_matrix.shape[1],
                    'feature_dim': feature_matrix.shape[0],
                    'filepath': str(filepath)
                }
            else:
                # Return statistical features
                stats = self.compute_statistics(features)
                
                return {
                    'features': stats,
                    'feature_dim': len(stats),
                    'filepath': str(filepath)
                }
                
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            return None
    
    def get_feature_dim(self):
        """Get the dimension of the extracted features."""
        base_dim = self.n_mfcc
        
        # Count feature types
        n_feature_types = 1  # MFCC
        if self.include_delta:
            n_feature_types += 1
        if self.include_delta_delta:
            n_feature_types += 1
        
        # Statistics per feature: mean, std, min, max, median
        n_stats = 5
        
        total_dim = base_dim * n_feature_types * n_stats
        
        return total_dim


def extract_features_from_dataset(metadata_path, audio_dir, output_dir,
                                  return_frames=False, n_mfcc=40):
    """
    Extract MFCC features from entire dataset.
    
    Args:
        metadata_path: Path to metadata CSV
        audio_dir: Directory containing audio files
        output_dir: Directory to save features
        return_frames: Whether to return frame-level features
        n_mfcc: Number of MFCCs to extract
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"✓ Loaded metadata: {len(df)} samples")
    
    # Initialize extractor
    extractor = MFCCExtractor(
        sr=16000,
        n_mfcc=n_mfcc,
        include_delta=True,
        include_delta_delta=True
    )
    
    print("\n" + "=" * 80)
    print("MFCC Extraction Configuration")
    print("=" * 80)
    print(f"Sampling rate: {extractor.sr} Hz")
    print(f"Number of MFCCs: {extractor.n_mfcc}")
    print(f"FFT size: {extractor.n_fft}")
    print(f"Hop length: {extractor.hop_length}")
    print(f"Window length: {extractor.win_length}")
    print(f"Include delta: {extractor.include_delta}")
    print(f"Include delta-delta: {extractor.include_delta_delta}")
    print(f"Return frames: {return_frames}")
    if not return_frames:
        print(f"Feature dimension: {extractor.get_feature_dim()}")
    print("=" * 80 + "\n")
    
    # Extract features
    features_list = []
    failed_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        # Get audio file path
        if 'filepath' in row:
            audio_path = Path(audio_dir) / row['filepath']
        else:
            continue
        
        if not audio_path.exists():
            failed_files.append(str(audio_path))
            continue
        
        # Extract features
        result = extractor.extract_from_file(audio_path, return_frames=return_frames)
        
        if result is not None:
            # Add metadata
            result['speaker_id'] = row.get('speaker_id', 'unknown')
            result['native_language'] = row.get('native_language', 'unknown')
            result['age_group'] = row.get('age_group', 'unknown')
            result['utterance_type'] = row.get('utterance_type', 'unknown')
            
            features_list.append(result)
        else:
            failed_files.append(str(audio_path))
    
    print("\n" + "=" * 80)
    print("Extraction Complete")
    print("=" * 80)
    print(f"✓ Successfully extracted: {len(features_list)} files")
    print(f"✗ Failed: {len(failed_files)} files")
    
    if len(features_list) == 0:
        print("✗ No features extracted!")
        return
    
    # Save features
    output_file = output_dir / ("mfcc_frames.pkl" if return_frames else "mfcc_stats.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(features_list, f)
    
    print(f"✓ Features saved to: {output_file}")
    
    # Save feature info
    info = {
        'n_samples': len(features_list),
        'feature_type': 'frame-level' if return_frames else 'statistical',
        'n_mfcc': n_mfcc,
        'feature_dim': features_list[0]['feature_dim'],
        'include_delta': True,
        'include_delta_delta': True
    }
    
    info_file = output_dir / ("mfcc_frames_info.pkl" if return_frames else "mfcc_stats_info.pkl")
    with open(info_file, 'wb') as f:
        pickle.dump(info, f)
    
    print(f"✓ Feature info saved to: {info_file}")
    
    # Print sample statistics
    if not return_frames:
        feature_matrix = np.array([item['features'] for item in features_list])
        print(f"\nFeature matrix shape: {feature_matrix.shape}")
        print(f"Feature range: [{feature_matrix.min():.2f}, {feature_matrix.max():.2f}]")
        print(f"Feature mean: {feature_matrix.mean():.2f}")
        print(f"Feature std: {feature_matrix.std():.2f}")
    
    if failed_files:
        failed_file_path = output_dir / 'failed_files.txt'
        with open(failed_file_path, 'w') as f:
            f.write('\n'.join(failed_files))
        print(f"✗ Failed files list saved to: {failed_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MFCC features from audio dataset"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted features"
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=40,
        help="Number of MFCCs to extract"
    )
    parser.add_argument(
        "--return_frames",
        action="store_true",
        help="Return frame-level features instead of statistics"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MFCC Feature Extraction")
    print("=" * 80)
    
    extract_features_from_dataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        return_frames=args.return_frames,
        n_mfcc=args.n_mfcc
    )
    
    print("\n✅ Feature extraction completed!")


if __name__ == "__main__":
    main()
