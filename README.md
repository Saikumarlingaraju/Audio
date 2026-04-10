# Indian Accent Classification using HuBERT and MFCC

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive deep learning project for classifying Indian accents from 6 different states using HuBERT transformer models and traditional MFCC features. Achieves **100% test accuracy** with optimized feature extraction.

## 🎯 Project Overview

This project implements and compares multiple approaches for Indian accent classification:
- **MFCC Baseline**: Traditional audio features with MLP classifier
- **HuBERT Layer-wise Analysis**: Systematic evaluation of transformer layers
- **Word vs Sentence Comparison**: Impact of temporal context on accuracy
- **Cross-Age Generalization**: Robustness evaluation across different age groups

### Key Achievements
- ✅ **100% Test Accuracy** with HuBERT Layer 3 + Sentence-Level Audio
- ✅ **99.76% Test Accuracy** with MFCC Baseline
- ✅ Discovered early-middle transformer layers outperform final layers for accent tasks
- ✅ Proved sentence-level audio (4.75% better) beats word-level segments
- ✅ **Cross-Age Study**: MFCC generalizes 33.57% better than HuBERT to children's speech

## 📊 Results Summary

| Approach | Val Accuracy | Test Accuracy | Feature Dim |
|----------|--------------|---------------|-------------|
| **HuBERT Layer 3 (Sentence)** ⭐ | **100.00%** | **100.00%** | 768 |
| MFCC Baseline (Sentence) | 100.00% | 99.76% | 600 |
| HuBERT Layer 0 (Sentence) | 99.76% | 99.52% | 768 |
| HuBERT Layer 12 (Sentence) | 99.52% | 99.05% | 768 |
| HuBERT Layer 3 (Word-Level) | 96.38% | 94.77% | 768 |

### Cross-Age Generalization (Adult → Child)

| Feature Type | Adult Test | Child Test | Generalization Gap |
|--------------|------------|------------|-------------------|
| **MFCC** ⭐ | 99.28% | 33.33% | 65.95% |
| HuBERT Layer 3 | 99.52% | 0.00% | 99.52% |

**Key Finding**: MFCC features demonstrate **33.57% better age robustness** than HuBERT!

## 🗂️ Dataset

**Indian Accents Dataset**: 6 Indian states
- **States**: Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, Tamil Nadu
- **Total Samples**: 2,798 audio utterances
- **Splits**: Train (1,958), Validation (420), Test (420)
- **Audio Format**: 16kHz, mono, WAV
- **Duration**: 2-5 seconds per utterance

### Data Distribution
```
Andhra Pradesh: ~467 utterances
Gujarat: ~467 utterances
Jharkhand: ~467 utterances
Karnataka: ~467 utterances
Kerala: ~467 utterances
Tamil Nadu: ~467 utterances
```



## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for HuBERT)
- 8GB+ RAM
- ~10GB disk space for data and models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manvita22/iiitpro.git
   cd iiitpro
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

#### 1. MFCC Baseline (Fastest)

Extract MFCC features and train:
```bash
# Extract MFCC features
python extract_mfcc_features.py

# Train baseline model
python train_simple.py --feature_type mfcc --epochs 20
```

**Expected**: 99.76% test accuracy in ~5 minutes

#### 2. HuBERT Layer-wise Analysis

Run on all layers (0, 3, 6, 9, 12):
```bash
python scripts/layerwise_hubert_analysis.py --layers 0,3,6,9,12 --pooling mean --epochs 10
```

**Expected**: Layer 3 achieves 100% test accuracy in ~15 minutes

#### 3. Word vs Sentence Comparison

```bash
python run_word_sentence_comparison_kaggle.py
```

**Expected**: Sentence-level 99.52% vs Word-level 94.77% in ~40 minutes

## 🧪 Experiments

### Task 1: MFCC Baseline ✅

**Objective**: Establish baseline using traditional audio features

```bash
python extract_mfcc_features.py
python train_simple.py --feature_type mfcc
```

**Results**:
- Validation: 100%
- Test: 99.76%
- Features: 40 MFCC × 5 statistics × 3 types = 600 dimensions

### Task 2: HuBERT Layer-wise Analysis ✅

**Objective**: Find optimal transformer layer for accent classification

```bash
python scripts/layerwise_hubert_analysis.py --layers 0,3,6,9,12 --pooling mean --epochs 10
```

**Results**:
| Layer | Val Acc | Test Acc |
|-------|---------|----------|
| 0 | 99.76% | 99.52% |
| **3** | **100%** | **100%** ⭐ |
| 6 | 100% | 99.29% |
| 9 | 100% | 99.05% |
| 12 | 99.52% | 99.05% |

**Key Finding**: Early-middle layers (Layer 3) outperform final layers!

### Task 3: Word vs Sentence Level ✅

**Objective**: Evaluate impact of temporal context

```bash
python run_word_sentence_comparison_kaggle.py
```

**Results**:
- **Sentence-Level**: 99.52% test accuracy (2,798 samples)
- **Word-Level**: 94.77% test accuracy (5,358 samples)
- **Difference**: +4.75% for sentence-level

**Key Finding**: Prosodic patterns in full sentences are crucial!

### Task 4: Cross-Age Generalization ✅

**Objective**: Test if adult-trained models work on children's speech

```bash
python cross_age_generalization_standalone.py
```

**Dataset**:
- Adult: 2,798 samples (training + testing)
- Children: 6 samples (testing only, never seen during training)

**Results**:

| Feature | Adult Acc | Child Acc | Gap |
|---------|-----------|-----------|-----|
| MFCC | 99.28% | 33.33% | 65.95% |
| HuBERT L3 | 99.52% | 0.00% | 99.52% |

**Key Findings**:
- ✅ **MFCC** maintains partial accuracy on children (33%)
- ❌ **HuBERT** completely fails on children (0%)
- 🏆 **Winner**: MFCC is 33.57% more age-robust!

**Why?**
- MFCC: Handcrafted features capture age-invariant acoustic properties
- HuBERT: Learned features overfit to adult voice characteristics (pitch, formants)

**See**: `CROSS_AGE_ANALYSIS.md` for detailed analysis

## 🎓 Key Findings

### 1. Layer Selection Matters
- **Layer 3** (early-middle) achieves perfect 100% accuracy
- **Layer 12** (final) achieves only 99.05% accuracy
- Accent features emerge early in transformer networks
- Later layers focus on semantic content, not accent

### 2. Context is Critical
- Sentence-level audio: 99.52% accuracy
- Word-level segments: 94.77% accuracy
- **4.75% improvement** with full temporal context
- Prosody, intonation, rhythm are essential for accent detection

### 3. Both MFCC and HuBERT Excel on Adult Speech
- MFCC: 99.76% (fast, lightweight, no GPU needed)
- HuBERT Layer 3: 100% (optimal, GPU recommended)
- Choose based on deployment constraints

### 4. Age Robustness Differs Dramatically
- **MFCC**: 33% accuracy on children (age-invariant features)
- **HuBERT**: 0% accuracy on children (overfits to adult voices)
- **Critical**: Always test on diverse age groups before deployment!
- **Recommendation**: Use MFCC for age-diverse applications

## 💻 Model Architecture

### Feature Extractors

**1. MFCC Extractor**
- 40 Mel-frequency cepstral coefficients
- Statistics: Mean, Std, Min, Max, Median
- Types: MFCC, Delta, Delta-Delta
- Output: 600-dimensional features

**2. HuBERT Extractor**
- Model: `facebook/hubert-base-ls960`
- 12 transformer layers
- Mean pooling over temporal dimension
- Output: 768-dimensional features per layer

### Classifier

**Multi-Layer Perceptron (MLP)**
```python
Input Layer: 600 (MFCC) or 768 (HuBERT)
Hidden Layer 1: 256 neurons + ReLU + Dropout(0.3)
Hidden Layer 2: 128 neurons + ReLU + Dropout(0.3)
Output Layer: 6 classes + Softmax
```

**Training**:
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch Size: 32
- Early Stopping: Based on validation accuracy

## 📈 Performance Metrics

### Best Configuration (HuBERT Layer 3 + Sentence)

| Metric | Score |
|--------|-------|
| Training Accuracy | 99.64% |
| Validation Accuracy | 100.00% |
| Test Accuracy | 100.00% |
| Precision (avg) | 100% |
| Recall (avg) | 100% |
| F1-Score (avg) | 100% |

**Confusion Matrix**: Perfect diagonal (zero misclassifications)

### Computational Efficiency

| Task | Time (Tesla T4 GPU) | Memory |
|------|---------------------|--------|
| MFCC Extraction | ~2 minutes | 2GB RAM |
| HuBERT Extraction (per layer) | ~2 minutes | 4GB GPU |
| Model Training | ~2-3 minutes | 2GB GPU |
| Inference (per sample) | ~14ms | <1GB GPU |

## 🌐 Web Application

A Flask-based web demo is available for interactive accent classification.

### Run the Demo

```bash
cd app
python app.py
```

Open browser to `http://localhost:5000`

**Features**:
- Upload audio file (WAV, MP3)
- Real-time accent prediction
- Confidence scores for all 6 states
- Audio waveform visualization

## 📦 Pretrained Models

Download pretrained models from experiments:

| Model | Test Accuracy | Size | Download |
|-------|---------------|------|----------|
| HuBERT Layer 3 (Sentence) | 100% | 230KB | `experiments/indian_accents_hubert_layer3_mean/best_model.pt` |
| MFCC Baseline | 99.76% | 230KB | `experiments/indian_accents_mfcc/best_model.pt` |
| HuBERT Layer 3 (Word) | 94.77% | 230KB | `experiments/word_level_layer3/best_model.pt` |
| **MFCC Cross-Age** ⭐ | **33% (Child)** | 230KB | `experiments/cross_age_mfcc/model.pt` |
| HuBERT Cross-Age | 0% (Child) | 230KB | `experiments/cross_age_hubert/model.pt` |

## 🔬 Reproducibility

### Hardware Used
- **Platform**: Kaggle
- **GPU**: Tesla T4
- **CUDA**: 12.4
- **RAM**: 29GB

### Software Versions
- Python: 3.11
- PyTorch: 2.6.0+cu124
- Transformers: 4.30.0
- Librosa: 0.11.0
- NumPy: 1.26.4
- Scikit-learn: 1.5.2



## 📚 Documentation

Detailed documentation available in:
- **layer_wise_results.txt**: Complete layer-wise analysis
- **word_vs_sent_analysis.txt**: Word vs sentence comparison analysis
- **CROSS_AGE_ANALYSIS.md**: Cross-age generalization study (NEW! ⭐)

## ☁️ Colab Training Starter (Class Data)

Use the starter notebook to run full training without local heavy compute:

- Notebook: `notebooks/colab_class_training_starter.ipynb`
- It performs: Drive mount -> repo sync -> metadata prep -> split creation -> validation -> MFCC/HuBERT extraction -> training

### Colab Quick Run

1. Open `notebooks/colab_class_training_starter.ipynb` in Colab or VS Code Colab extension.
2. Set `REPO_URL` and `REPO_BRANCH` if needed.
3. Set `DRIVE_AUDIO_ROOT` to your Google Drive class-audio folder.
4. Run all cells in order.

## 🔄 Exact Audio Upload and Sync Workflow

This is the recommended production workflow for class collection data.

### Step 1: Keep metadata in GitHub

- Commit and push:
   - `data/collections/class_students/organized_by_state_clean/metadata_all.csv`
- Do not commit raw audio blobs to GitHub.

### Step 2: Upload audio to Google Drive

Create this folder layout in Drive:

```text
MyDrive/
   iiitpro_audio/
      organized_by_state_clean/
         andhra_pradesh/
         telangana/
         telugu/
         ...
```

Important:
- Copy only state folders under `organized_by_state_clean` (not inside another nested folder).
- Keep filenames unchanged (`prompt_1.webm`, `prompt_2.webm`, `prompt_3.webm`).

### Step 3: Normalize metadata for training paths

Run:

```bash
python scripts/prepare_class_metadata_for_training.py \
   --metadata data/collections/class_students/organized_by_state_clean/metadata_all.csv \
   --audio_root /content/drive/MyDrive/iiitpro_audio/organized_by_state_clean \
   --output data/collections/class_students/organized_by_state_clean/metadata_training_ready.csv \
   --min_bytes 1000 --strict
```

What it does:
- Filters tiny/invalid files.
- Normalizes path mismatches between export metadata and real folder layout.
- Normalizes language labels (for example `telengana` -> `telangana`).

### Step 4: Create speaker-disjoint splits

```bash
python src/data/create_splits.py \
   --metadata data/collections/class_students/organized_by_state_clean/metadata_training_ready.csv \
   --output_dir data/splits/class_students \
   --train_ratio 0.7 --dev_ratio 0.15 --test_ratio 0.15
```

### Step 5: Validate before training

```bash
python scripts/validate_class_dataset.py \
   --metadata data/collections/class_students/organized_by_state_clean/metadata_training_ready.csv \
   --audio_root /content/drive/MyDrive/iiitpro_audio/organized_by_state_clean \
   --split_dir data/splits/class_students
```

### Step 6: Extract features and train

MFCC baseline:

```bash
python src/features/mfcc_extractor.py \
   --metadata data/collections/class_students/organized_by_state_clean/metadata_training_ready.csv \
   --audio_dir /content/drive/MyDrive/iiitpro_audio/organized_by_state_clean \
   --output_dir data/features/class_students/mfcc

python train_simple.py \
   --features_path data/features/class_students/mfcc/mfcc_stats.pkl \
   --train_csv data/splits/class_students/train.csv \
   --val_csv data/splits/class_students/dev.csv \
   --test_csv data/splits/class_students/test.csv \
   --output_dir experiments/class_students_colab_mfcc
```

HuBERT + robust model:

```bash
python src/features/hubert_extractor.py \
   --metadata data/collections/class_students/organized_by_state_clean/metadata_training_ready.csv \
   --audio_dir /content/drive/MyDrive/iiitpro_audio/organized_by_state_clean \
   --output_dir data/features/class_students/hubert \
   --extract_layer 3 --pooling mean

python train_robust.py \
   --features_path data/features/class_students/hubert/hubert_layer3_mean.pkl \
   --train_csv data/splits/class_students/train.csv \
   --val_csv data/splits/class_students/dev.csv \
   --test_csv data/splits/class_students/test.csv \
   --output_dir experiments/class_students_colab_hubert_robust \
   --model_type robust_mlp --num_epochs 30
```

## ⚙️ GitHub Actions: Auto-Retrain on New Commits

Workflow file:

- `.github/workflows/retrain-class-data.yml`

Triggers:
- Push to `main` when training/data code changes.
- Weekly schedule.
- Manual run with input controls.

What it runs automatically:
1. (Optional) export latest class recordings from Aiven.
2. Prepare normalized metadata.
3. Create speaker-disjoint splits.
4. Validate integrity.
5. Extract MFCC features.
6. Retrain baseline model (`train_simple.py`).
7. Upload artifacts (model + splits + features + metadata).

Required repository secret:

- `AIVEN_DB_URI` (needed when export step is enabled).

Notes:
- GitHub-hosted runners are CPU-only.
- This workflow is intentionally CPU-safe (MFCC baseline).
- Use Colab (or a self-hosted GPU runner) for HuBERT-heavy retraining.









## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🔮 Future Work

- [ ] Extend to more Indian states (currently 6)
- [ ] Test on spontaneous speech vs read speech
- [ ] Add attention visualization for interpretability
- [ ] Deploy as REST API with Docker
- [ ] Create mobile app for real-time detection
- [ ] Study cross-lingual transfer to other languages
- [ ] Add speaker diarization capabilities
- [ ] Implement adversarial robustness testing

---


