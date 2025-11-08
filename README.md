# Native Language Identification of Indian English Speakers Using HuBERT

## Overview

This project develops a Native Language Identification (NLI) system that identifies the native language (L1) of Indian speakers speaking English by analyzing accent patterns in their speech. The system leverages both traditional acoustic features (MFCCs) and self-supervised speech representations (HuBERT embeddings) to classify speakers' native languages from diverse Indian linguistic backgrounds.

## Project Objectives

1. **Develop NLI Model**: Build and train models to identify Indian speakers' native languages (Hindi, Tamil, Telugu, Malayalam, Kannada, etc.) from English speech
2. **Feature Comparison**: Compare effectiveness of MFCCs vs. HuBERT embeddings for accent modeling
3. **HuBERT Layer Analysis**: Conduct layer-by-layer analysis to determine which HuBERT layer best captures accent information
4. **Cross-Age Generalization**: Evaluate model robustness when training on adults and testing on children
5. **Linguistic Level Analysis**: Compare word-level vs. sentence-level accent detection performance
6. **Real-World Application**: Demonstrate accent-aware cuisine recommendation system

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

**Extract MFCC features:**
```bash
python src/features/mfcc_extractor.py --audio_dir data/processed --output_dir data/features/mfcc
```

**Extract HuBERT embeddings:**
```bash
python src/features/hubert_extractor.py --audio_dir data/processed --output_dir data/features/hubert --extract_all_layers
```

### Model Training

**Train baseline MFCC model:**
```bash
python src/models/train.py --config experiments/configs/mfcc_baseline.yaml
```

**Train HuBERT-based model:**
```bash
python src/models/train.py --config experiments/configs/hubert_layer_6.yaml
```

**HuBERT layer-wise analysis:**
```bash
python src/models/layer_analysis.py --config experiments/configs/layer_sweep.yaml
```

### Evaluation

**Evaluate model:**
```bash
python src/models/evaluate.py --checkpoint models/checkpoints/best_model.ckpt --test_split data/splits/test.csv
```

**Cross-age evaluation:**
```bash
python src/models/evaluate.py --checkpoint models/checkpoints/adult_trained.ckpt --test_split data/splits/children_test.csv
```

### Run Accent-Aware Cuisine Recommendation App

```bash
cd app
python app.py
```

Then navigate to `http://localhost:5000` in your browser.

## Key Experiments

1. **MFCC Baseline**: Traditional acoustic features with MLP/CNN classifiers
2. **HuBERT Layer Analysis**: Evaluate each of the 12 HuBERT layers to find optimal representation
3. **Architecture Comparison**: Compare MLP, CNN, BiLSTM, and Transformer classifiers
4. **Cross-Age Generalization**: Train on adults → Test on children
5. **Word vs. Sentence**: Compare accent cues at different linguistic granularities

## Expected Outcomes

- Functional NLI model for Indian English accents
- Comparative analysis: MFCC vs. HuBERT performance
- Identification of optimal HuBERT layer for accent modeling
- Insights on cross-age generalization capabilities
- Understanding of word-level vs. sentence-level accent detection
- Working demonstration of accent-aware application

## Technologies Used

- **Deep Learning**: PyTorch, PyTorch Lightning
- **Speech Processing**: Librosa, Torchaudio, Soundfile
- **Pre-trained Models**: HuggingFace Transformers, Fairseq
- **ML Tools**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly, UMAP
- **Experiment Tracking**: Weights & Biases, TensorBoard
- **Web Framework**: Flask

## Conceptual Background

An **accent** is the distinct pronunciation pattern influenced by a speaker's native language (L1). When Indian speakers speak English, their pronunciation carries acoustic traces of their L1, creating recognizable patterns like Hindi-English, Tamil-English, or Malayalam-English accents.

These accent cues manifest in:
- **Vowel quality**: Different vowel realizations based on L1 phonology
- **Consonant articulation**: Retroflex sounds, aspiration patterns
- **Prosodic rhythm**: Stress, intonation, and timing patterns

**HuBERT** (Hidden-Unit BERT) is a self-supervised speech model that learns robust speech representations from unlabeled audio. This project explores how HuBERT's learned representations encode accent characteristics and their effectiveness compared to traditional acoustic features.

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
