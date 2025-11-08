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

**Indian Accent Database (IndicAccentDb)**
- Source: [HuggingFace Dataset](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)
- Contains English speech recordings from Indian speakers with various L1 backgrounds
- Includes both adult and child speaker subsets
- Multiple native languages represented: Hindi, Tamil, Telugu, Malayalam, Kannada, and more

## Project Structure

```
firstiiit/
├── data/
│   ├── raw/              # Raw audio files from IndicAccentDb
│   ├── processed/        # Preprocessed audio files (16kHz, normalized)
│   └── splits/           # Train/dev/test splits metadata
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_hubert_analysis.ipynb
│   └── 05_results_visualization.ipynb
├── src/
│   ├── data/
│   │   ├── download_dataset.py
│   │   ├── preprocess.py
│   │   └── create_splits.py
│   ├── features/
│   │   ├── mfcc_extractor.py
│   │   ├── hubert_extractor.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── bilstm.py
│   │   └── transformer.py
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       └── config.py
├── models/
│   └── checkpoints/      # Saved model checkpoints
├── experiments/
│   ├── configs/          # Experiment configuration files
│   └── logs/             # Training logs
├── results/
│   ├── figures/          # Visualizations and plots
│   └── metrics/          # Performance metrics and reports
├── app/
│   ├── app.py            # Flask application for cuisine recommendation
│   ├── templates/        # HTML templates
│   └── static/           # CSS, JS, and static assets
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment specification
└── README.md            # This file
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

### 2. Download Dataset

```bash
python src/data/download_dataset.py --output_dir data/raw
```

### 3. Preprocess Audio

```bash
python src/data/preprocess.py --input_dir data/raw --output_dir data/processed
```

### 4. Create Data Splits

```bash
python src/data/create_splits.py --data_dir data/processed --output_dir data/splits
```

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
@misc{indicaccentdb,
  title={Indian Accent Database},
  author={Darshana S},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/DarshanaS/IndicAccentDb}}
}
```

## License

MIT License - See LICENSE file for details

## Contributors

- Manvita
- Sathwik
- Sahasra

## Acknowledgments

- IndicAccentDb dataset creators
- HuggingFace for model hosting and datasets
- Meta AI for HuBERT pre-trained models
