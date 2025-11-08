# 🔍 CHECKLIST: Why External Audio Predictions Fail

## ✅ Issues Already FIXED

### 1. ✅ Preprocessing Mismatch - FIXED!
**Status:** This was the main issue and it's now resolved.
- **Before:** Inference used `preprocess_audio()` with trimming/padding/normalization
- **After:** Inference now uses same `extract_from_file()` as training
- **Result:** 100% accuracy on training samples ✅

### 2. ✅ Sample Rate / Channel Mismatch - NOT AN ISSUE
**Status:** Handled correctly by librosa
- Training files are 44.1kHz, but librosa resamples to 16kHz automatically
- Stereo is converted to mono automatically
- This happens identically in training AND inference
- **Confirmed:** No mismatch here ✅

### 3. ✅ Feature Extraction Parameters - CORRECT
**Status:** All parameters match perfectly
- N_MFCC: 40 ✅
- Sample rate: 16kHz ✅
- N_FFT: 512 ✅
- Hop length: 160 ✅
- Window length: 400 ✅
- Delta & delta-delta: Both included ✅
- Statistics: mean, std, min, max, median ✅

### 4. ✅ Feature Scaling / Normalization - NOT APPLIED
**Status:** Correct - no scaling in training or inference
- No mean/std scaling applied ✅
- No min/max normalization ✅
- Raw MFCC statistics used ✅

### 5. ✅ Duration / Padding - NOT APPLIED
**Status:** Correct - no padding or trimming
- No fixed-length padding ✅
- No silence trimming ✅
- Variable length OK (MFCCs use statistics) ✅

---

## ⚠️ Issues That MAY Affect External Audio (Cannot Fix Without More Data)

### 1. 🎯 Recording Quality / Domain Shift (MOST LIKELY)
**Why it matters:**
- Training data: Clean studio recordings from IndicAccentDB
- Your recording: Phone/laptop mic, different room acoustics
- Result: Different frequency response, noise floor, reverberation

**What you'll see:**
- Model may be uncertain (low confidence scores)
- May confuse similar accents
- May predict based on recording artifacts instead of accent

**Quick test:**
```
1. Record in QUIET room
2. Speak 5-10cm from mic
3. Use CLEAR speech
4. Avoid background noise
```

### 2. 🎤 Microphone / Channel Characteristics
**Why it matters:**
- Training: Professional recording equipment
- Your recording: Phone/laptop mic with different characteristics
- Different mics have different frequency responses

**Evidence:**
- Training audio peak: 0.06-0.09 (normalized)
- Features range: [-667, +170]
- If your recording has very different energy, features will be off

### 3. 🗣️ Speaking Style / Accent Strength
**Why it matters:**
- Training speakers may have STRONG regional accents
- You may have mild/neutral accent
- Different phonetic content (different words)

**What to do:**
- Speak with more prominent accent features
- Use similar words/phrases as training data
- Speak naturally, don't try to fake an accent

### 4. 📏 Audio Duration
**Why it matters:**
- Training samples: 2-4 seconds typical
- Very short (<1s): Not enough phonetic content
- Very long (>10s): Feature statistics may be too averaged

**Recommendation:**
- Record 3-5 seconds of speech
- Speak a full sentence
- Don't just say 1-2 words

### 5. 🎵 Background Noise / Compression
**Why it matters:**
- Training: Clean recordings
- Your recording: May have noise, compression artifacts
- Phone recordings often have aggressive compression

**What to do:**
- Record in quiet environment
- Avoid compressed formats if possible (use WAV instead of MP3)
- Use voice memo app, not phone call recording

---

## 🧪 Debugging Steps (DO THESE NOW)

### Step 1: Test with Known Good Audio
✅ **DONE** - Test script shows 100% accuracy on training samples
```
All 12 training samples: 100% correct ✅
Confidence: 55%-100%
```

### Step 2: Test Your Recording
⏳ **YOUR TURN** - Record and upload

1. **Record audio:**
   - Duration: 3-5 seconds
   - Content: Read a sentence in English (any sentence)
   - Quality: Quiet room, clear speech
   - Format: WAV preferred, MP3 OK

2. **Upload to web interface:**
   - Open: http://127.0.0.1:5000
   - Upload your recording
   - Check results

3. **What to look for:**
   - **Predicted class:** Does it make sense?
   - **Confidence:** >50% is good, <30% means uncertain
   - **Top 3 predictions:** Are they reasonable?

### Step 3: Compare Feature Distributions
If predictions are still wrong, run this:

```powershell
python -c "
import sys
sys.path.insert(0, '.')
from src.features.mfcc_extractor import MFCCExtractor
import numpy as np

# Your external audio
ext = MFCCExtractor().extract_from_file('path/to/your/recording.wav', return_frames=False)
ext_feat = ext['features']

# Training sample
train = MFCCExtractor().extract_from_file('data/raw/indian_accents/tamil/Tamil_speaker (1002).wav', return_frames=False)
train_feat = train['features']

print('External audio features:')
print(f'  Range: [{np.min(ext_feat):.2f}, {np.max(ext_feat):.2f}]')
print(f'  Mean: {np.mean(ext_feat):.2f}')
print(f'  Std: {np.std(ext_feat):.2f}')

print('Training audio features:')
print(f'  Range: [{np.min(train_feat):.2f}, {np.max(train_feat):.2f}]')
print(f'  Mean: {np.mean(train_feat):.2f}')
print(f'  Std: {np.std(train_feat):.2f}')

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([ext_feat], [train_feat])[0, 0]
print(f'Cosine similarity: {sim:.4f}')
print('  >0.8: Very similar ✅')
print('  0.5-0.8: Moderately similar ⚠️')
print('  <0.5: Very different ❌')
"
```

### Step 4: Visualize Spectrogram (Optional)
```python
python -c "
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load your recording
y, sr = librosa.load('path/to/your/recording.wav', sr=16000)

# Compute spectrogram
D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)

# Plot
plt.figure(figsize=(12, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Your Recording Spectrogram')
plt.tight_layout()
plt.savefig('my_recording_spectrogram.png')
print('Saved to: my_recording_spectrogram.png')
"
```

---

## 💡 Quick Fixes to Try NOW

### Fix 1: Record High-Quality Sample
```
Equipment: Phone or laptop
Environment: Quiet room, close door
Distance: 5-10cm from mic
Duration: 3-5 seconds
Content: "Hello, my name is [name]. I am from [city]. The weather is nice today."
Format: WAV if possible, MP3 OK
```

### Fix 2: Test Server is Running
```powershell
# Check if server is running
Get-Process python -ErrorAction SilentlyContinue

# If not running, start it
python app\app_robust.py
```

### Fix 3: Use Debug Endpoint
Test a training file first to make sure everything works:
```
http://127.0.0.1:5000/debug_predict?path=data/raw/indian_accents/tamil/Tamil_speaker%20(1002).wav
```
Should predict "Tamil" with high confidence.

---

## 🎯 Expected Results

### Best Case (Ideal Recording)
```
Predicted: Karnataka (or your actual accent)
Confidence: 60-95%
Top 3: All reasonable predictions
```

### Realistic Case (Normal Phone Recording)
```
Predicted: Kerala (maybe not exact, but reasonable)
Confidence: 35-70%
Top 3: Include some related accents
```

### Worst Case (Domain Shift Too Large)
```
Predicted: Random state
Confidence: 20-30% (very uncertain)
Top 3: All similar low scores
```

**If worst case happens:**
- Model needs more diverse training data
- Consider recording more samples in your environment
- Or use data augmentation (noise, reverberation) during training

---

## 🔧 If Predictions Still Wrong After Testing

### Option A: Collect More Data
Record 10-20 samples in your environment for each accent, retrain.

### Option B: Add Data Augmentation (Already Have Script)
```powershell
# This adds noise, pitch shift, time stretch
python augment_and_train.py
```
This creates a more robust model, but may reduce training accuracy to 75-90%.

### Option C: Fine-tune on Your Recordings
If you can record some samples with known labels, fine-tune the model.

### Option D: Ensemble Multiple Models
Train multiple models and average predictions.

---

## 📊 Current Status Summary

✅ **Training/Inference Pipeline:** FIXED - 100% match
✅ **Sample Rate:** Correct - 16kHz resampling
✅ **Feature Extraction:** Correct - identical parameters
✅ **Preprocessing:** Correct - none applied
✅ **Model Loading:** Correct - clean model (100% test acc)

⏳ **Pending:** Test external audio recording
⚠️ **Risk:** Domain shift (recording quality, mic, environment)

---

## 🎤 Action Items for You

1. **Record audio** (3-5 seconds, clear speech, quiet room)
2. **Upload** to http://127.0.0.1:5000
3. **Report results** here:
   - Predicted state
   - Confidence score
   - Top 3 predictions
   - Your actual accent/background

Based on results, we can:
- ✅ Celebrate if it works!
- 🔍 Debug if predictions are random
- 🎯 Improve model if predictions are close but not perfect
