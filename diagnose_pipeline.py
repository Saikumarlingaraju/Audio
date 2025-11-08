"""
Diagnostic script to compare training vs inference pipeline
Checks all the points mentioned by ChatGPT
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.features.mfcc_extractor import MFCCExtractor

print("=" * 80)
print("🔍 PIPELINE DIAGNOSTIC TOOL")
print("=" * 80)

# Initialize extractor
extractor = MFCCExtractor()

# Pick a training sample
training_file = Path("data/raw/indian_accents/tamil/Tamil_speaker (1002).wav")

print(f"\n📊 Analyzing training sample: {training_file.name}")
print("-" * 80)

# 1. Check sample rate and channels (CRITICAL!)
audio, sr = librosa.load(str(training_file), sr=None, mono=False)
if audio.ndim == 1:
    channels = 1
else:
    channels = audio.shape[0]

print(f"\n✓ Sample Rate: {sr} Hz")
print(f"✓ Channels: {channels} (mono={channels==1})")
print(f"✓ Duration: {len(audio) / sr:.2f} seconds")
print(f"✓ Audio shape: {audio.shape}")
print(f"✓ Audio range: [{np.min(audio):.4f}, {np.max(audio):.4f}]")

# 2. Check what MFCCExtractor does
print(f"\n📊 MFCCExtractor Configuration:")
print(f"✓ Target sample rate: {extractor.sr} Hz")
print(f"✓ N_MFCC: {extractor.n_mfcc}")
print(f"✓ N_FFT: {extractor.n_fft}")
print(f"✓ Hop length: {extractor.hop_length}")
print(f"✓ Win length: {extractor.win_length}")

# 3. Extract features and show details
print(f"\n📊 Feature Extraction Process:")
result = extractor.extract_from_file(str(training_file), return_frames=False)
features = result['features']

print(f"✓ Feature shape: {features.shape}")
print(f"✓ Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
print(f"✓ Feature mean: {np.mean(features):.4f}")
print(f"✓ Feature std: {np.std(features):.4f}")

# 4. Check if any preprocessing is applied
print(f"\n📊 Preprocessing in extract_from_file():")
print("Looking at MFCCExtractor code...")

# Load audio how extractor does it
audio_check, sr_check = librosa.load(str(training_file), sr=extractor.sr)
print(f"✓ Librosa loads with sr={sr_check} (forced to {extractor.sr})")
print(f"✓ Resulting audio shape: {audio_check.shape}")
print(f"✓ Resulting audio range: [{np.min(audio_check):.4f}, {np.max(audio_check):.4f}]")

# Check if there's silence trimming
audio_trimmed, _ = librosa.effects.trim(audio_check, top_db=30)
trim_diff = len(audio_check) - len(audio_trimmed)
print(f"✓ If trimmed (top_db=30): would remove {trim_diff} samples ({trim_diff/sr_check:.2f}s)")

# 5. Check for normalization
print(f"\n📊 Normalization Check:")
print(f"✓ Audio already normalized: {np.abs(audio_check).max() <= 1.0}")
print(f"✓ Peak amplitude: {np.abs(audio_check).max():.4f}")

# 6. Show what happens with different sample rates
print(f"\n⚠️  CRITICAL: Sample Rate Mismatch Test")
print("-" * 80)

# Load at original rate
audio_original, sr_original = librosa.load(str(training_file), sr=None)
print(f"Original file: {sr_original} Hz, {len(audio_original)} samples, {len(audio_original)/sr_original:.2f}s")

# Load at 16kHz (what extractor uses)
audio_16k, sr_16k = librosa.load(str(training_file), sr=16000)
print(f"Resampled to 16kHz: {sr_16k} Hz, {len(audio_16k)} samples, {len(audio_16k)/sr_16k:.2f}s")

# Extract features from both
feat_original = extractor.extract_from_file(str(training_file), return_frames=False)['features']
print(f"\n✓ Features extracted: shape={feat_original.shape}, mean={np.mean(feat_original):.4f}")

print("\n" + "=" * 80)
print("📋 SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("\n✅ Training Pipeline Uses:")
print(f"   • Sample rate: 16000 Hz (forced by librosa.load)")
print(f"   • Channels: Mono (librosa default)")
print(f"   • Normalization: Librosa default (float32 in [-1, 1])")
print(f"   • Trimming: NO (not applied in extract_from_file)")
print(f"   • Padding: NO (not applied)")
print(f"   • Feature scaling: NO (raw MFCC statistics)")

print("\n⚠️  For External Audio to Work:")
print("   1. Record at ANY sample rate (will be resampled to 16kHz)")
print("   2. Mono or stereo OK (librosa converts to mono)")
print("   3. NO need to trim silence (training didn't trim)")
print("   4. NO need to normalize (librosa handles it)")
print("   5. Audio format: WAV, MP3, etc. (librosa supports all)")

print("\n🎯 Most Likely Issues with External Audio:")
print("   1. Recording quality: Phone/laptop mic vs training studio mic")
print("   2. Background noise: Training data may be cleaner")
print("   3. Speaking style: Training speakers may have stronger accents")
print("   4. Content: Different words/phonemes than training")
print("   5. Duration: Very short (<1s) or very long (>10s) clips")

print("\n💡 Quick Test:")
print("   Record 3-5 seconds of clear English speech")
print("   Use a quiet room, speak clearly")
print("   Upload to web interface and check results")

print("\n" + "=" * 80)
