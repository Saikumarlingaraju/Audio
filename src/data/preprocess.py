"""
Audio preprocessing pipeline: resampling, normalization, silence trimming.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Preprocess audio files for NLI model training."""
    
    def __init__(self, target_sr=16000, trim_silence=True, 
                 top_db=20, normalize=True):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sr: Target sampling rate (default 16000 Hz)
            trim_silence: Whether to trim silence from audio
            top_db: Threshold for silence detection in dB
            normalize: Whether to normalize audio amplitude
        """
        self.target_sr = target_sr
        self.trim_silence = trim_silence
        self.top_db = top_db
        self.normalize = normalize
        
    def load_audio(self, filepath):
        """Load audio file."""
        try:
            audio, sr = librosa.load(filepath, sr=None)
            return audio, sr
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None, None
    
    def resample_audio(self, audio, orig_sr):
        """Resample audio to target sampling rate."""
        if orig_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio
    
    def trim_silence_from_audio(self, audio):
        """Trim silence from beginning and end of audio."""
        if self.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=self.top_db)
        return audio
    
    def normalize_audio(self, audio):
        """Normalize audio amplitude to [-1, 1]."""
        if self.normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        return audio
    
    def preprocess(self, audio, sr):
        """Apply full preprocessing pipeline."""
        # Resample
        audio = self.resample_audio(audio, sr)
        
        # Trim silence
        audio = self.trim_silence_from_audio(audio)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        return audio
    
    def process_file(self, input_path, output_path):
        """
        Process a single audio file.
        
        Returns:
            dict: Metadata about processed file
        """
        # Load audio
        audio, sr = self.load_audio(input_path)
        
        if audio is None:
            return None
        
        # Store original duration
        orig_duration = len(audio) / sr
        
        # Preprocess
        audio_processed = self.preprocess(audio, sr)
        
        # Calculate new duration
        new_duration = len(audio_processed) / self.target_sr
        
        # Save processed audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_processed, self.target_sr)
        
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'orig_duration': orig_duration,
            'processed_duration': new_duration,
            'sampling_rate': self.target_sr,
            'samples': len(audio_processed)
        }


def process_dataset(input_dir, output_dir, metadata_path=None, 
                   target_sr=16000, trim_silence=True, top_db=20):
    """
    Process entire dataset.
    
    Args:
        input_dir: Directory with raw audio files
        output_dir: Directory to save processed files
        metadata_path: Path to metadata CSV (optional)
        target_sr: Target sampling rate
        trim_silence: Whether to trim silence
        top_db: Silence detection threshold
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = AudioPreprocessor(
        target_sr=target_sr,
        trim_silence=trim_silence,
        top_db=top_db,
        normalize=True
    )
    
    # Load metadata if available
    if metadata_path and Path(metadata_path).exists():
        metadata_df = pd.read_csv(metadata_path)
        audio_files = [input_dir / fp for fp in metadata_df['filepath']]
        print(f"Using metadata from: {metadata_path}")
        print(f"Found {len(audio_files)} files in metadata")
    else:
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(input_dir.rglob(ext))
        print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return
    
    print("\n" + "=" * 80)
    print("Audio Preprocessing Configuration")
    print("=" * 80)
    print(f"Target sampling rate: {target_sr} Hz")
    print(f"Trim silence: {trim_silence}")
    print(f"Silence threshold: {top_db} dB")
    print(f"Normalize amplitude: True")
    print("=" * 80 + "\n")
    
    # Process files
    processed_metadata = []
    failed_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Maintain directory structure
            rel_path = audio_file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.wav')
            
            # Process file
            metadata = preprocessor.process_file(audio_file, output_path)
            
            if metadata:
                processed_metadata.append(metadata)
            else:
                failed_files.append(str(audio_file))
                
        except Exception as e:
            print(f"\nError processing {audio_file}: {str(e)}")
            failed_files.append(str(audio_file))
            continue
    
    # Save processing metadata
    if processed_metadata:
        processed_df = pd.DataFrame(processed_metadata)
        metadata_output = output_dir / "preprocessing_metadata.csv"
        processed_df.to_csv(metadata_output, index=False)
        
        # Update original metadata if available
        if metadata_path and Path(metadata_path).exists():
            original_df = pd.read_csv(metadata_path)
            # Update filepaths to point to processed files
            original_df['original_filepath'] = original_df['filepath']
            original_df['filepath'] = original_df['filepath'].apply(
                lambda x: str(Path(x).with_suffix('.wav'))
            )
            original_df['processed'] = True
            original_df['processed_sampling_rate'] = target_sr
            
            updated_metadata_path = output_dir / "metadata.csv"
            original_df.to_csv(updated_metadata_path, index=False)
            print(f"\n✓ Updated metadata saved to: {updated_metadata_path}")
        
        print("\n" + "=" * 80)
        print("Preprocessing Complete!")
        print("=" * 80)
        print(f"✓ Successfully processed: {len(processed_metadata)} files")
        print(f"✗ Failed: {len(failed_files)} files")
        print(f"✓ Output directory: {output_dir}")
        print(f"✓ Preprocessing metadata: {metadata_output}")
        
        # Statistics
        print("\nDuration Statistics:")
        print(f"  Original total duration: {processed_df['orig_duration'].sum():.2f} seconds")
        print(f"  Processed total duration: {processed_df['processed_duration'].sum():.2f} seconds")
        print(f"  Average duration: {processed_df['processed_duration'].mean():.2f} ± {processed_df['processed_duration'].std():.2f} seconds")
        print(f"  Min duration: {processed_df['processed_duration'].min():.2f} seconds")
        print(f"  Max duration: {processed_df['processed_duration'].max():.2f} seconds")
        
        if failed_files:
            print(f"\n⚠ Failed files saved to: {output_dir / 'failed_files.txt'}")
            with open(output_dir / 'failed_files.txt', 'w') as f:
                f.write('\n'.join(failed_files))
    
    else:
        print("\n✗ No files were successfully processed!")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio files for NLI training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed audio files"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sampling rate in Hz"
    )
    parser.add_argument(
        "--no_trim",
        action="store_true",
        help="Disable silence trimming"
    )
    parser.add_argument(
        "--top_db",
        type=int,
        default=20,
        help="Threshold for silence detection in dB"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Audio Preprocessing Pipeline")
    print("=" * 80)
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        target_sr=args.target_sr,
        trim_silence=not args.no_trim,
        top_db=args.top_db
    )
    
    print("\n✅ Preprocessing completed!")


if __name__ == "__main__":
    main()
