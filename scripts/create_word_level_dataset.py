"""
Create word-level dataset from sentence-level audio for accent analysis
Segments audio using energy-based Voice Activity Detection (VAD)
"""
import sys
from pathlib import Path
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import numpy as np

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

def detect_words(audio, sr, min_silence_len=0.1, silence_thresh=-40):
    """
    Detect word boundaries using energy-based VAD
    
    Args:
        audio: Audio signal
        sr: Sample rate
        min_silence_len: Minimum silence duration in seconds
        silence_thresh: Energy threshold in dB
    
    Returns:
        List of (start, end) sample indices for each word
    """
    # Compute short-time energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    
    # Find speech/silence segments
    is_speech = energy_db > silence_thresh
    
    # Convert frame indices to sample indices
    word_segments = []
    in_word = False
    word_start = 0
    
    for i, speech in enumerate(is_speech):
        sample_idx = i * hop_length
        
        if speech and not in_word:
            # Start of word
            word_start = sample_idx
            in_word = True
        elif not speech and in_word:
            # End of word
            word_end = sample_idx
            duration = (word_end - word_start) / sr
            
            # Only keep segments longer than minimum duration
            if duration > 0.2:  # At least 200ms
                word_segments.append((word_start, word_end))
            
            in_word = False
    
    # Handle case where audio ends during speech
    if in_word:
        word_segments.append((word_start, len(audio)))
    
    return word_segments

def create_word_level_dataset(
    input_metadata='data/raw/indian_accents/metadata_with_splits.csv',
    audio_base_dir='data/raw/indian_accents',
    output_dir='data/word_level',
    min_words=3,
    max_words=10
):
    """
    Create word-level dataset by segmenting sentence-level audio
    
    Args:
        input_metadata: Path to metadata CSV with sentence-level data
        audio_base_dir: Base directory containing audio files
        output_dir: Output directory for word-level audio
        min_words: Minimum number of words to extract per utterance
        max_words: Maximum number of words to extract per utterance
    
    Returns:
        DataFrame with word-level metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output directories for each state
    metadata = pd.read_csv(input_metadata)
    states = metadata['native_language'].unique()
    
    for state in states:
        (output_path / state).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Creating Word-Level Dataset")
    print(f"{'='*80}\n")
    print(f"Input metadata: {input_metadata}")
    print(f"Output directory: {output_dir}")
    print(f"Processing {len(metadata)} audio files...\n")
    
    word_level_data = []
    word_count = 0
    skipped = 0
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing audio"):
        audio_path = Path(audio_base_dir) / row['filepath']
        
        if not audio_path.exists():
            skipped += 1
            continue
        
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            
            # Detect word segments
            word_segments = detect_words(audio, sr)
            
            # Limit number of words extracted
            word_segments = word_segments[:max_words]
            
            if len(word_segments) < min_words:
                skipped += 1
                continue
            
            # Extract and save each word
            for word_idx, (start, end) in enumerate(word_segments):
                word_audio = audio[start:end]
                
                # Skip very short segments
                if len(word_audio) < sr * 0.2:  # Less than 200ms
                    continue
                
                # Save word audio
                output_filename = f"{audio_path.stem}_word{word_idx:02d}.wav"
                output_filepath = output_path / row['native_language'] / output_filename
                
                sf.write(str(output_filepath), word_audio, sr)
                
                # Add to metadata
                word_level_data.append({
                    'filepath': str(output_filepath.relative_to(output_path.parent)),
                    'native_language': row['native_language'],
                    'split': row['split'],
                    'source_file': row['filepath'],
                    'word_index': word_idx,
                    'duration': len(word_audio) / sr,
                    'unit_type': 'word'
                })
                
                word_count += 1
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            skipped += 1
            continue
    
    # Save word-level metadata
    word_df = pd.DataFrame(word_level_data)
    word_df.to_csv(output_path / 'word_level_metadata.csv', index=False)
    
    print(f"\n{'='*80}")
    print("Word-Level Dataset Created")
    print(f"{'='*80}\n")
    print(f"Total word segments: {word_count}")
    print(f"Skipped utterances: {skipped}")
    print(f"Metadata saved to: {output_path / 'word_level_metadata.csv'}")
    
    # Print distribution
    print("\nWord distribution by state:")
    print(word_df.groupby('native_language').size())
    
    print("\nWord distribution by split:")
    print(word_df.groupby('split').size())
    
    print("\nAverage duration statistics:")
    print(f"Mean: {word_df['duration'].mean():.3f}s")
    print(f"Median: {word_df['duration'].median():.3f}s")
    print(f"Min: {word_df['duration'].min():.3f}s")
    print(f"Max: {word_df['duration'].max():.3f}s")
    
    return word_df

if __name__ == '__main__':
    create_word_level_dataset()
