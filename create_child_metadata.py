"""
Create metadata for child audio samples
Run this locally before uploading to Kaggle
"""
import pandas as pd
from pathlib import Path
import librosa

def create_child_metadata():
    """Create metadata CSV for child audio samples"""
    
    child_audio_dir = Path('data/raw/child_audios')
    
    if not child_audio_dir.exists():
        print(f"Error: {child_audio_dir} does not exist!")
        return
    
    metadata = []
    
    # Scan all audio files in child_audios folder
    for audio_file in child_audio_dir.rglob('*.wav'):
        # Try to infer state from filename or folder structure
        # Adjust this based on your naming convention
        relative_path = audio_file.relative_to(Path('data/raw'))
        
        # Get audio duration
        try:
            duration = librosa.get_duration(path=str(audio_file))
        except:
            duration = 0.0
        
        # Extract state from path or filename (customize based on your naming)
        # Example: child_audios/andhra_pradesh/file.wav or child_audios/andhra_child_001.wav
        parts = audio_file.stem.lower()
        
        state = 'unknown'
        if 'andhra' in parts or 'andhra_pradesh' in str(audio_file.parent):
            state = 'andhra_pradesh'
        elif 'gujrat' in parts or 'gujarat' in parts:
            state = 'gujrat'
        elif 'jharkhand' in parts:
            state = 'jharkhand'
        elif 'karnataka' in parts:
            state = 'karnataka'
        elif 'kerala' in parts:
            state = 'kerala'
        elif 'tamil' in parts:
            state = 'tamil'
        
        metadata.append({
            'filepath': str(relative_path).replace('\\', '/'),
            'native_language': state,
            'age_group': 'child',
            'duration': duration,
            'filename': audio_file.name
        })
    
    # Create DataFrame
    df = pd.DataFrame(metadata)
    
    # Save metadata
    output_file = 'data/raw/child_audios/child_metadata.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Created metadata for {len(df)} child audio files")
    print(f"  Saved to: {output_file}")
    print(f"\nDistribution by state:")
    print(df['native_language'].value_counts())
    print(f"\nTotal duration: {df['duration'].sum():.2f} seconds")
    
    # Show samples
    print(f"\nSample entries:")
    print(df.head())
    
    return df

if __name__ == '__main__':
    create_child_metadata()
