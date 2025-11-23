"""
Compare external audio with training data to diagnose prediction issues
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def analyze_audio(audio_path):
    """Analyze audio file properties"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        # Get audio stats
        print(f"\n{'='*60}")
        print(f"File: {Path(audio_path).name}")
        print(f"{'='*60}")
        print(f"Sample Rate: {sr} Hz")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Samples: {len(y)}")
        print(f"Amplitude Range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"RMS Energy: {np.sqrt(np.mean(y**2)):.4f}")
        print(f"Zero Crossing Rate: {np.mean(librosa.zero_crossings(y)):.4f}")
        
        # Spectral stats
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        print(f"Avg Spectral Centroid: {np.mean(spectral_centroids):.2f} Hz")
        
        return {
            'sr': sr,
            'duration': duration,
            'rms': np.sqrt(np.mean(y**2)),
            'amplitude_range': (y.min(), y.max())
        }
    except Exception as e:
        print(f"Error analyzing {audio_path}: {e}")
        return None

if __name__ == "__main__":
    print("\n🔍 AUDIO COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Check training data samples
    print("\n📊 TRAINING DATA SAMPLES:")
    train_dir = Path("data/raw/indian_accents")
    
    for state_dir in train_dir.iterdir():
        if state_dir.is_dir():
            audio_files = list(state_dir.glob("*.wav"))[:2]  # First 2 files
            if audio_files:
                print(f"\n{state_dir.name.upper()}:")
                for audio_file in audio_files:
                    analyze_audio(str(audio_file))
                    break  # Just one per state
    
    # Check uploads folder for external audio
    print("\n\n📤 UPLOADED/EXTERNAL AUDIO:")
    uploads_dir = Path("app/uploads")
    if uploads_dir.exists():
        external_files = list(uploads_dir.glob("*.*"))
        if external_files:
            for ext_file in external_files[-3:]:  # Last 3 uploads
                analyze_audio(str(ext_file))
        else:
            print("No files in uploads folder")
    else:
        print("Uploads folder doesn't exist")
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("1. Check if sample rates match (should be 16000 Hz)")
    print("2. Ensure duration is 3-10 seconds")
    print("3. Verify audio quality (no clipping, good RMS level)")
    print("4. Record in quiet environment with clear speech")
    print("="*60)
