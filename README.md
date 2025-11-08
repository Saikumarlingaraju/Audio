# Native Language Identification of Indian English Speakers Using HuBERT

## Overview

This project develops a Native Language Identification (NLI) system that identifies the native language (L1) of Indian speakers speaking English by analyzing accent patterns in their speech. The system primarily uses self-supervised speech representations (HuBERT embeddings) with optional traditional acoustic features (MFCCs) for comparison to classify speakers' native languages from diverse Indian linguistic backgrounds.

## Project Objectives

1. **Develop NLI Model**: Build and train models to identify Indian speakers' native languages from 6 Indian states (Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, Tamil Nadu) from English speech
2. **HuBERT-Primary Architecture**: Use HuBERT layer 12 pooled embeddings (mean + std) as the primary feature representation
3. **Feature Comparison**: Compare effectiveness of HuBERT embeddings vs. traditional MFCCs for accent modeling
4. **Model Performance**: Achieve high-accuracy accent classification using MLP architecture
5. **Real-World Application**: Demonstrate accent-aware cuisine recommendation system
6. **Production Deployment**: Web application with Flask for real-time accent detection

## Dataset

**Indian Accent Database (Custom Collection)**
- English speech recordings from Indian speakers across 6 states
- **States covered**: Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, Tamil Nadu
- **Total samples**: 2,800 audio recordings
- **Speaker type**: Adult speakers reading English text
- **Recording conditions**: Varied sampling rates (44.1kHz, 48kHz) - preprocessed to 16kHz
- **Duration**: ~2-3 seconds per utterance
- **Data splits**: Train (70%), Validation (15%), Test (15%) - speaker-disjoint

## Project Structure

```
firstiiit/
├── data/
│   ├── raw/
│   │   └── indian_accents/        # Raw audio files by state
│   │       ├── andhra_pradesh/
│   │       ├── gujrat/
│   │       ├── jharkhand/
│   │       ├── karnataka/
│   │       ├── kerala/
│   │       ├── tamil/
│   │       ├── metadata_full.csv
│   │       └── metadata_with_splits.csv
│   ├── processed/
│   │   └── commonvoice/           # Preprocessed audio (16kHz)
│   ├── features/
│   │   └── indian_accents/        # Extracted features
│   │       ├── hubert_layer12_mean.pkl  # HuBERT embeddings (primary)
│   │       ├── mfcc_stats.pkl           # MFCC features (comparison)
│   │       ├── train_mean.npy
│   │       └── train_std.npy
│   └── splits/                     # Train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── data/
│   │   ├── download_dataset.py
│   │   ├── download_accentdb_extended.py
│   │   ├── download_commonvoice.py
│   │   ├── download_sample.py
│   │   ├── preprocess.py
│   │   └── create_splits.py
│   ├── features/
│   │   ├── hubert_extractor.py    # HuBERT feature extraction
│   │   ├── mfcc_extractor.py      # MFCC feature extraction
│   │   └── dataset.py             # PyTorch Dataset
│   ├── models/
│   │   ├── mlp.py                 # MLP classifier (primary)
│   │   ├── cnn.py                 # CNN classifier
│   │   ├── bilstm.py              # BiLSTM classifier
│   │   └── transformer.py         # Transformer classifier
│   └── utils/
│       ├── config.py
│       └── metrics.py
├── models/
│   ├── checkpoints/
│   │   └── demo_model.pt          # Trained model checkpoint
│   └── label_encoder.pkl          # Label encoder for classes
├── experiments/
│   ├── indian_accents_mlp/        # MLP experiment results
│   │   └── best_model.pt
│   └── indian_accents_mlp_augmented/  # Augmented training results
│       └── best_model.pt
├── app/
│   ├── app.py                     # Basic Flask app
│   ├── app_robust.py              # Production Flask app (HuBERT)
│   ├── start_server.bat           # Windows startup script
│   └── templates/
│       └── index.html             # Web interface
├── train_simple.py                # Simple training script (HuBERT primary)
├── train_fast.py                  # Fast training script
├── train_robust.py                # Robust training with validation
├── train_clean.py                 # Clean training pipeline
├── augment_and_train.py           # Training with data augmentation
├── extract_indian_features.py     # HuBERT feature extraction
├── create_indian_metadata.py      # Create metadata CSV
├── test_training_accuracy.py      # Evaluate model accuracy
├── test_audio_comparison.py       # Compare audio processing
├── diagnose_pipeline.py           # Pipeline diagnostics
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── PROJECT_SUMMARY.md             # Implementation summary
├── LICENSE                        # MIT License
├── .gitignore                     # Git exclusions
└── README.md                      # This file
```

## Setup Instructions

### 1. Create Environment

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate nli-indian-english
```

**Using pip:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Dataset

The dataset is already included in the repository under `data/raw/indian_accents/` with 2,800 audio files from 6 Indian states.

**Create metadata with splits:**
```bash
python create_indian_metadata.py
```

**Extract features:**
```bash
python extract_indian_features.py
```

This will create:
- `data/splits/train.csv`, `val.csv`, `test.csv` - Data splits
- `data/features/indian_accents/hubert_layer12_mean.pkl` - HuBERT embeddings

## Usage

### Feature Extraction

**Extract HuBERT embeddings (Primary):**
```bash
python extract_indian_features.py
```
This extracts HuBERT layer 12 pooled embeddings (mean + std) for all audio files.

**Extract MFCC features (Optional - for comparison):**
```bash
python src/features/mfcc_extractor.py --audio_dir data/processed --output_dir data/features/mfcc
```

### Model Training

**Train HuBERT-based model (Primary):**
```bash
python train_simple.py
```
This trains an MLP classifier on HuBERT layer 12 pooled embeddings (mean + std).

**Fast training (recommended for quick experiments):**
```bash
python train_fast.py
```

**Robust training with validation:**
```bash
python train_robust.py
```

**Clean training pipeline:**
```bash
python train_clean.py
```

### Evaluation

**Test training accuracy:**
```bash
python test_training_accuracy.py
```

**Audio comparison tests:**
```bash
python test_audio_comparison.py
```

### Run Accent-Aware Cuisine Recommendation App

**Using HuBERT model (recommended):**
```bash
python app\app_robust.py
```

**Using basic app:**
```bash
python app\app.py
```

Then navigate to `http://localhost:5000` in your browser to upload audio and get accent predictions with regional cuisine recommendations.

## Key Features

1. **HuBERT-Primary Pipeline**: Uses HuBERT layer 12 pooled embeddings (mean + std over time) for rich acoustic-phonetic representation
2. **MLP Architecture**: Efficient multi-layer perceptron classifier with [256, 128] hidden dimensions
3. **6-State Classification**: Classifies accents from Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, and Tamil Nadu
4. **Real-time Web App**: Flask-based application for uploading audio and getting instant predictions
5. **Cuisine Recommendations**: Provides regional cuisine suggestions based on detected accent

## Project Results

- ✅ Functional NLI model for 6 Indian state accents in English
- ✅ HuBERT layer 12 pooled embeddings as primary feature representation (1536-dim: 768 mean + 768 std)
- ✅ MLP classifier achieving high accuracy on test set
- ✅ Optional MFCC comparison showing HuBERT superiority for accent modeling
- ✅ Working web application for real-time accent detection and cuisine recommendations
- ✅ Production-ready Flask deployment with audio preprocessing pipeline

## Technologies Used

- **Deep Learning**: PyTorch, PyTorch Lightning
- **Speech Processing**: Librosa, Torchaudio, Soundfile
- **Pre-trained Models**: HuggingFace Transformers, Fairseq
- **ML Tools**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly, UMAP
- **Experiment Tracking**: Weights & Biases, TensorBoard
- **Web Framework**: Flask

## Conceptual Background

An **accent** is the distinct pronunciation pattern influenced by a speaker's native language (L1). When Indian speakers speak English, their pronunciation carries acoustic traces of their regional language, creating recognizable patterns specific to each state.

These accent cues manifest in:
- **Vowel quality**: Different vowel realizations based on L1 phonology
- **Consonant articulation**: Retroflex sounds, aspiration patterns
- **Prosodic rhythm**: Stress, intonation, and timing patterns

**HuBERT** (Hidden-Unit BERT) is a self-supervised speech model that learns robust speech representations from unlabeled audio. This project uses HuBERT's layer 12 representations, which capture rich acoustic-phonetic information ideal for accent classification. We pool these embeddings using mean and standard deviation statistics across time, creating a fixed 1536-dimensional representation per utterance.

**Why HuBERT over MFCCs?**
- MFCCs capture only spectral envelope (basic acoustic properties)
- HuBERT learns contextual phonetic patterns through self-supervised pre-training
- HuBERT embeddings encode prosody, rhythm, and subtle pronunciation differences
- Superior performance for accent and speaker characteristics

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{indian_accent_nli_2025,
  title={Native Language Identification of Indian English Speakers Using HuBERT},
  author={Manvita and Sathwik and Sahasra},
  year={2025},
  howpublished={\url{https://github.com/Manvita22/iiitpro}}
}
```

## License

MIT License - See LICENSE file for details

## Contributors

- Manvita


## Acknowledgments

- Indian Accent speech data contributors
- HuggingFace for Transformers library and model hosting
- Meta AI for HuBERT pre-trained models (facebook/hubert-base-ls960)
- PyTorch and scikit-learn communities
