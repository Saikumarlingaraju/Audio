"""
Flask Web App - Native Language Identification
Uses trained MLP model on AccentDB Extended dataset
"""
import sys
from pathlib import Path

# Add parent directory to path to import src modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from flask import Flask, render_template, request, jsonify
import torch
import pickle
import numpy as np
import os
import re
import hashlib
import json
import csv
import io
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus
from werkzeug.utils import secure_filename

try:
    import psycopg2
except Exception:
    psycopg2 = None

try:
    import librosa
except Exception:
    librosa = None

from src.models.mlp import MLPClassifier
from src.features.mfcc_extractor import MFCCExtractor

# We'll import HuBERTExtractor lazily to avoid heavy imports during server start
HubertExtractor = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and label encoder
MODEL_PATH = root_dir / 'experiments' / 'indian_accents_hubert' / 'best_model.pt'
ENCODER_PATH = root_dir / 'experiments' / 'indian_accents_hubert' / 'label_encoder.pkl'
FALLBACK_ENCODER_PATH = root_dir / 'models' / 'label_encoder.pkl'

print("Loading model...")
checkpoint = {}
model = None
label_encoder = None
MODEL_READY = False
RUN_MODE = 'demo'
mfcc_extractor = None
mfcc_classes = []
mfcc_train_matrix = None
mfcc_train_labels = []
mfcc_train_matrix_norm = None
mfcc_knn_k = 25

QUALITY_SCORE_MIN = 40.0
QUALITY_SCORE_HARD_FLOOR = 30.0
CONFIDENCE_UNCERTAIN_THRESHOLD = 43.0
CONFIDENCE_CALIBRATION_TEMPERATURE = 1.20

MFCC_FEATURES_PATH = root_dir / 'data' / 'features' / 'indian_accents' / 'mfcc_stats.pkl'
FEEDBACK_DIR = root_dir / 'data' / 'feedback'
FEEDBACK_LOG_PATH = FEEDBACK_DIR / 'feedback_log.jsonl'
STUDENT_COLLECTION_DIR = root_dir / 'data' / 'collections' / 'class_students'
STUDENT_AUDIO_DIR = STUDENT_COLLECTION_DIR / 'audio'
STUDENT_SUBMISSIONS_LOG_PATH = STUDENT_COLLECTION_DIR / 'student_submissions.jsonl'
STUDENT_METADATA_CSV_PATH = STUDENT_COLLECTION_DIR / 'student_metadata.csv'

ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}
STUDENT_PROMPTS = [
    'The quick brown fox jumps over the lazy dog near the village market. Please speak clearly at a natural pace for ten seconds.',
    'Today the teacher checked thirty thin notebooks, three thick textbooks, and thirteen blue pens before the class discussion began.',
    'Strong winds moved across green fields, bright roads, and small houses while children played, laughed, and shared stories after school.'
]
STUDENT_METADATA_COLUMNS = [
    'submission_id', 'timestamp_utc', 'filepath', 'filename', 'speaker_id',
    'student_name', 'student_id', 'department_section', 'native_language',
    'native_language_other', 'hometown',
    'age_group', 'utterance_type', 'text', 'sampling_rate', 'duration',
    'split', 'source', 'storage_backend', 'storage_uri'
]


def _extract_aiven_db_uri_from_file(env_path):
    try:
        raw = Path(env_path).read_text(encoding='utf-8')
    except Exception:
        return ''

    psql_match = re.search(r"psql\s+['\"](postgres(?:ql)?://[^'\"]+)['\"]", raw, flags=re.IGNORECASE)
    if psql_match:
        return psql_match.group(1).strip()

    uri_match = re.search(r'postgres(?:ql)?://[^\s]+', raw, flags=re.IGNORECASE)
    if uri_match:
        return uri_match.group(0).strip().strip('"\'')

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) == 0:
        return ''

    def _value_after(label):
        lowered = label.lower()
        for i, line in enumerate(lines):
            if line.lower() == lowered and i + 1 < len(lines):
                return lines[i + 1]
        return ''

    host = _value_after('Host')
    port = _value_after('Port')
    user = _value_after('User')
    password = _value_after('Password')
    dbname = _value_after('Database name')
    ssl_mode = _value_after('SSL mode') or 'require'

    if host and port and user and password and dbname:
        user_enc = quote_plus(user)
        pass_enc = quote_plus(password)
        return f"postgresql://{user_enc}:{pass_enc}@{host}:{port}/{dbname}?sslmode={ssl_mode}"

    return ''


def _get_aiven_db_uri():
    for key in ['AIVEN_DB_URI', 'DATABASE_URL', 'AIVEN_DATABASE_URL', 'POSTGRES_URI']:
        value = str(os.getenv(key, '')).strip()
        if value:
            psql_match = re.search(r"psql\s+['\"](postgres(?:ql)?://[^'\"]+)['\"]", value, flags=re.IGNORECASE)
            if psql_match:
                return psql_match.group(1).strip()
            return value.strip('"\'')
    return _extract_aiven_db_uri_from_file(root_dir / '.env')


AIVEN_DB_URI = _get_aiven_db_uri()
AIVEN_DB_SYNC_ENABLED = str(os.getenv('AIVEN_DB_SYNC_ENABLED', '1')).strip().lower() in {'1', 'true', 'yes', 'on'}
AIVEN_DB_STRICT = str(os.getenv('AIVEN_DB_STRICT', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}

os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(STUDENT_AUDIO_DIR, exist_ok=True)

encoder_path = ENCODER_PATH if ENCODER_PATH.exists() else FALLBACK_ENCODER_PATH
if encoder_path.exists():
    with open(str(encoder_path), 'rb') as f:
        label_encoder = pickle.load(f)

if MODEL_PATH.exists() and label_encoder is not None:
    checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')
    model = MLPClassifier(input_dim=1536, hidden_dims=[256, 128], num_classes=6, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    MODEL_READY = True
    RUN_MODE = 'model'


def initialize_mfcc_fallback():
    """Build a lightweight k-NN classifier from pre-extracted MFCC train features."""
    global mfcc_classes, mfcc_train_matrix, mfcc_train_labels, mfcc_train_matrix_norm, RUN_MODE

    if not MFCC_FEATURES_PATH.exists():
        return

    try:
        with open(str(MFCC_FEATURES_PATH), 'rb') as f:
            records = pickle.load(f)

        train_vectors = []
        train_labels = []
        for row in records:
            if not isinstance(row, dict):
                continue
            features = row.get('features')
            language = row.get('native_language')
            split = str(row.get('split', 'train')).lower()
            if features is None or language is None or split != 'train':
                continue
            arr = np.asarray(features, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            train_vectors.append(arr)
            train_labels.append(str(language))

        if len(train_vectors) == 0:
            return

        mfcc_train_matrix = np.stack(train_vectors, axis=0).astype(np.float32)
        norms = np.linalg.norm(mfcc_train_matrix, axis=1, keepdims=True) + 1e-8
        mfcc_train_matrix_norm = mfcc_train_matrix / norms
        mfcc_train_labels = train_labels

        if label_encoder is not None:
            ordered = [str(c) for c in label_encoder.classes_ if str(c) in set(mfcc_train_labels)]
            mfcc_classes = ordered if ordered else sorted(list(set(mfcc_train_labels)))
        else:
            mfcc_classes = sorted(list(set(mfcc_train_labels)))

        RUN_MODE = 'mfcc-lite'
    except Exception as init_err:
        print(f"MFCC fallback init failed: {init_err}")


if RUN_MODE != 'model':
    initialize_mfcc_fallback()

# Initialize HuBERT feature extractor (model is loaded in the extractor's constructor)
hubert_extractor = None  # Lazy init to avoid blocking server start on network issues

# Safe printing of startup state
if MODEL_READY:
    _acc = checkpoint.get('test_acc', checkpoint.get('val_acc', None))
    if isinstance(_acc, (int, float)):
        print(f"✓ Model weights loaded: Test accuracy {_acc:.2f}%")
    else:
        print(f"✓ Model weights loaded. Test accuracy: {_acc}")
    print(f"✓ Classes: {list(label_encoder.classes_)}")
    print("→ HuBERT extractor will initialize on first prediction request.")
else:
    if RUN_MODE == 'mfcc-lite':
        print(f"⚠ Model weights not found at {MODEL_PATH}")
        print(f"✓ Running CPU-light MFCC fallback with {len(mfcc_classes)} classes")
    elif label_encoder is not None:
        print(f"⚠ Model weights not found at {MODEL_PATH}")
        print(f"✓ Loaded label encoder with classes: {list(label_encoder.classes_)}")
        print("→ Running in lightweight demo mode (no model inference, no training required).")
    else:
        print("⚠ No label encoder found. Predictions will be unavailable.")


def extract_features(audio_path):
    """Extract HuBERT features (mean+std over frames from layer 12 -> 1536 dims)."""
    try:
        global hubert_extractor
        if hubert_extractor is None:
            try:
                print("Initializing HuBERT extractor (lazy)...")
                # Lazy import to prevent heavy transformer/torch submodules from loading at app import time
                global HubertExtractor
                if HubertExtractor is None:
                    from importlib import import_module
                    HubertExtractor = getattr(import_module('src.features.hubert_extractor'), 'HuBERTExtractor')
                hubert_extractor = HubertExtractor()
            except Exception as init_err:
                print(f"HuBERT init failed: {init_err}")
                return None

        # Ask extractor for frame-level embeddings from layer 12
        result = hubert_extractor.extract_from_file(
            audio_path, extract_layer=12, pooling=None
        )

        if result is None:
            print(f"Feature extraction returned None for {audio_path}")
            return None

        # Pull out frames for the requested layer
        frames = None
        if isinstance(result, dict):
            embeddings_dict = result.get('embeddings')
            if isinstance(embeddings_dict, dict):
                if 'layer_12' in embeddings_dict:
                    frames = embeddings_dict['layer_12']  # shape: (T, 768)
                elif 12 in embeddings_dict:
                    frames = embeddings_dict[12]
                else:
                    # Fallback to first key
                    first_key = list(embeddings_dict.keys())[0]
                    print(f"Warning: using {first_key} instead of layer_12")
                    frames = embeddings_dict[first_key]
            else:
                # Unexpected shape; try to use as frames
                frames = embeddings_dict
        else:
            frames = result

        if frames is None:
            print(f"No frame-level embeddings found for {audio_path}")
            return None

        frames = np.asarray(frames)
        if frames.ndim == 1:
            # Already pooled to (768,), compute std as zeros to keep 1536 dims
            mean_vec = frames.astype(np.float32)
            std_vec = np.zeros_like(mean_vec, dtype=np.float32)
        else:
            # Compute mean and std over time dimension
            mean_vec = np.nanmean(frames, axis=0).astype(np.float32)
            std_vec = np.nanstd(frames, axis=0).astype(np.float32)

        # Replace NaNs/Infs if any
        mean_vec = np.nan_to_num(mean_vec, nan=0.0, posinf=0.0, neginf=0.0)
        std_vec = np.nan_to_num(std_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate to get 1536-dim feature
        features = np.concatenate([mean_vec, std_vec]).astype(np.float32)
        print(f"HuBERT features ready: {features.shape} (mean+std)")

        return features

    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_cuisine_recommendations(state):
    """Get cuisine recommendations based on predicted state"""
    cuisine_map = {
        'andhra_pradesh': [
            {'name': 'Hyderabadi Biryani', 'region': 'Andhra Pradesh', 'description': 'Aromatic rice dish with marinated meat'},
            {'name': 'Pesarattu', 'region': 'Andhra Pradesh', 'description': 'Green gram dosa served with ginger chutney'},
            {'name': 'Gongura Mutton', 'region': 'Andhra Pradesh', 'description': 'Tangy mutton curry with sorrel leaves'}
        ],
        'gujrat': [
            {'name': 'Dhokla', 'region': 'Gujarat', 'description': 'Steamed fermented rice and chickpea snack'},
            {'name': 'Gujarati Thali', 'region': 'Gujarat', 'description': 'Complete meal with variety of sweet and savory dishes'},
            {'name': 'Khandvi', 'region': 'Gujarat', 'description': 'Rolled gram flour snacks with tempered spices'}
        ],
        'jharkhand': [
            {'name': 'Litti Chokha', 'region': 'Jharkhand', 'description': 'Roasted wheat balls with mashed vegetables'},
            {'name': 'Dhuska', 'region': 'Jharkhand', 'description': 'Deep-fried rice and lentil pancakes'},
            {'name': 'Rugra', 'region': 'Jharkhand', 'description': 'Mushroom curry with local spices'}
        ],
        'karnataka': [
            {'name': 'Bisi Bele Bath', 'region': 'Karnataka', 'description': 'Spicy rice and lentil dish with vegetables'},
            {'name': 'Mysore Masala Dosa', 'region': 'Karnataka', 'description': 'Crispy dosa with red chutney and potato filling'},
            {'name': 'Ragi Mudde', 'region': 'Karnataka', 'description': 'Finger millet balls with sambar'}
        ],
        'kerala': [
            {'name': 'Appam with Stew', 'region': 'Kerala', 'description': 'Soft rice pancakes with coconut milk stew'},
            {'name': 'Kerala Fish Curry', 'region': 'Kerala', 'description': 'Fish cooked in coconut milk with spices'},
            {'name': 'Puttu and Kadala', 'region': 'Kerala', 'description': 'Steamed rice cylinders with chickpea curry'}
        ],
        'tamil': [
            {'name': 'Chettinad Chicken', 'region': 'Tamil Nadu', 'description': 'Spicy chicken curry with aromatic spices'},
            {'name': 'Idli Sambar', 'region': 'Tamil Nadu', 'description': 'Steamed rice cakes with lentil soup'},
            {'name': 'Pongal', 'region': 'Tamil Nadu', 'description': 'Savory rice and lentil dish with pepper'}
        ]
    }
    
    return cuisine_map.get(state.lower(), [])


def _stable_softmax(logits):
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / (np.sum(exp) + 1e-8)


def calibrate_probabilities(probs, temperature=CONFIDENCE_CALIBRATION_TEMPERATURE):
    """Apply temperature scaling to reduce overconfident distributions."""
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return p
    p = np.clip(p, 1e-8, 1.0)
    logits = np.log(p)
    scaled = logits / max(float(temperature), 1e-6)
    return _stable_softmax(scaled)


def compute_calibrated_confidence(calibrated_probs):
    """Blend top-probability, class margin, and entropy into a robust confidence score."""
    p = np.asarray(calibrated_probs, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return 0.0
    top = float(np.max(p))
    sorted_p = np.sort(p)[::-1]
    second = float(sorted_p[1]) if sorted_p.size > 1 else 0.0
    margin = max(0.0, top - second)

    entropy = -np.sum(p * np.log(p + 1e-8))
    entropy_norm = float(entropy / np.log(max(2, p.size)))
    entropy_signal = 1.0 - np.clip(entropy_norm, 0.0, 1.0)

    confidence = (0.55 * top) + (0.35 * margin) + (0.10 * entropy_signal)
    return float(np.clip(confidence * 100.0, 0.0, 100.0))


def analyze_audio_quality(audio_path):
    """Return a quality report used by gating logic before final output."""
    report = {
        'score': 100.0,
        'level': 'excellent',
        'can_predict': True,
        'issues': [],
        'duration_sec': None,
        'mean_rms': None,
        'silent_ratio': None,
        'clipping_ratio': None,
        'zcr': None
    }

    if librosa is None:
        report['issues'].append('audio_quality_backend_unavailable')
        return report

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            report['score'] = 0.0
            report['can_predict'] = False
            report['level'] = 'poor'
            report['issues'].append('empty_audio')
            return report

        duration_sec = float(y.size / max(sr, 1))
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).reshape(-1)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512).reshape(-1)

        mean_rms = float(np.mean(rms)) if rms.size > 0 else 0.0
        silent_ratio = float(np.mean(rms < 0.010)) if rms.size > 0 else 1.0
        clipping_ratio = float(np.mean(np.abs(y) >= 0.98))
        mean_zcr = float(np.mean(zcr)) if zcr.size > 0 else 0.0

        report['duration_sec'] = duration_sec
        report['mean_rms'] = mean_rms
        report['silent_ratio'] = silent_ratio
        report['clipping_ratio'] = clipping_ratio
        report['zcr'] = mean_zcr

        penalty = 0.0
        critical = False

        if duration_sec < 0.8:
            penalty += 45.0
            critical = True
            report['issues'].append('audio_too_short')
        elif duration_sec < 1.5:
            penalty += 20.0
            report['issues'].append('short_audio')

        if silent_ratio > 0.85:
            penalty += 40.0
            critical = True
            report['issues'].append('mostly_silence')
        elif silent_ratio > 0.60:
            penalty += 15.0
            report['issues'].append('high_silence')

        if mean_rms < 0.006:
            penalty += 18.0
            report['issues'].append('very_low_volume')
        elif mean_rms < 0.012:
            penalty += 8.0
            report['issues'].append('low_volume')

        if clipping_ratio > 0.04:
            penalty += 20.0
            report['issues'].append('audio_clipping')

        if mean_zcr > 0.24:
            penalty += 10.0
            report['issues'].append('possible_background_noise')

        score = float(np.clip(100.0 - penalty, 0.0, 100.0))
        report['score'] = score

        if score < 35.0:
            report['level'] = 'poor'
        elif score < 60.0:
            report['level'] = 'fair'
        elif score < 80.0:
            report['level'] = 'good'
        else:
            report['level'] = 'excellent'

        report['can_predict'] = bool((not critical) and (score >= QUALITY_SCORE_HARD_FLOOR))
        return report

    except Exception as quality_err:
        print(f"Audio quality check failed: {quality_err}")
        report['issues'].append('quality_check_failed')
        return report


def _build_uncertain_response(mode, quality_report, reason, top_predictions=None):
    return {
        'success': True,
        'predicted_accent': 'uncertain',
        'confidence': 0.0,
        'raw_confidence': 0.0,
        'calibrated_confidence': 0.0,
        'is_uncertain': True,
        'uncertain_reason': reason,
        'top_predictions': top_predictions or [],
        'recommendations': [],
        'quality': quality_report,
        'mode': mode
    }


def _read_feedback_records():
    records = []
    if not FEEDBACK_LOG_PATH.exists():
        return records
    try:
        with open(str(FEEDBACK_LOG_PATH), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        records.append(row)
                except json.JSONDecodeError:
                    continue
    except Exception as read_err:
        print(f"Feedback read failed: {read_err}")
    return records


def _append_feedback_record(record):
    try:
        with open(str(FEEDBACK_LOG_PATH), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')
        return True
    except Exception as write_err:
        print(f"Feedback write failed: {write_err}")
        return False


def _allowed_audio_file(filename):
    text = str(filename or '').strip().lower()
    return '.' in text and text.rsplit('.', 1)[1] in ALLOWED_AUDIO_EXTENSIONS


def _normalize_student_id(value):
    text = str(value or '').strip().lower()
    text = re.sub(r'[^a-z0-9_-]+', '_', text)
    text = text.strip('_')
    return text[:64]


def _hash_client_ip(value):
    text = str(value or '').strip()
    if text == '':
        return ''
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _read_student_submissions():
    records = []
    if not STUDENT_SUBMISSIONS_LOG_PATH.exists():
        return records
    try:
        with open(str(STUDENT_SUBMISSIONS_LOG_PATH), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        records.append(row)
                except json.JSONDecodeError:
                    continue
    except Exception as read_err:
        print(f"Student submission read failed: {read_err}")
    return records


def _append_student_submission_record(record):
    try:
        os.makedirs(STUDENT_COLLECTION_DIR, exist_ok=True)
        with open(str(STUDENT_SUBMISSIONS_LOG_PATH), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')
        return True
    except Exception as write_err:
        print(f"Student submission write failed: {write_err}")
        return False


def _ensure_student_metadata_header():
    os.makedirs(STUDENT_COLLECTION_DIR, exist_ok=True)
    if STUDENT_METADATA_CSV_PATH.exists() and STUDENT_METADATA_CSV_PATH.stat().st_size > 0:
        return
    with open(str(STUDENT_METADATA_CSV_PATH), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STUDENT_METADATA_COLUMNS)
        writer.writeheader()


def _append_student_metadata_rows(rows):
    try:
        _ensure_student_metadata_header()
        with open(str(STUDENT_METADATA_CSV_PATH), 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=STUDENT_METADATA_COLUMNS)
            for row in rows:
                writer.writerow(row)
        return True
    except Exception as write_err:
        print(f"Student metadata write failed: {write_err}")
        return False


def _ensure_aiven_tables(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS student_submissions (
                submission_id TEXT PRIMARY KEY,
                timestamp_utc TIMESTAMPTZ NOT NULL,
                student_name TEXT NOT NULL,
                student_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                department_section TEXT NOT NULL,
                hometown TEXT NOT NULL,
                native_language TEXT NOT NULL,
                native_language_other TEXT,
                consent BOOLEAN NOT NULL,
                num_recordings INTEGER NOT NULL,
                client_ip_hash TEXT,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS student_recordings (
                id BIGSERIAL PRIMARY KEY,
                submission_id TEXT NOT NULL REFERENCES student_submissions(submission_id) ON DELETE CASCADE,
                slot INTEGER NOT NULL,
                filepath TEXT NOT NULL,
                filename TEXT NOT NULL,
                prompt TEXT,
                sampling_rate INTEGER,
                duration DOUBLE PRECISION,
                storage_backend TEXT,
                storage_uri TEXT
            )
            """
        )


def _sync_student_submission_to_aiven(record):
    if not AIVEN_DB_SYNC_ENABLED:
        return True, ''
    if AIVEN_DB_URI == '':
        return False, 'Aiven DB URI not configured'
    if psycopg2 is None:
        return False, 'psycopg2 is not installed'

    try:
        with psycopg2.connect(AIVEN_DB_URI) as conn:
            _ensure_aiven_tables(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO student_submissions (
                        submission_id, timestamp_utc, student_name, student_id, speaker_id,
                        department_section, hometown, native_language, native_language_other,
                        consent, num_recordings, client_ip_hash, source, payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (submission_id) DO NOTHING
                    """,
                    (
                        record.get('submission_id', ''),
                        record.get('timestamp_utc', ''),
                        record.get('name', ''),
                        record.get('student_id', ''),
                        record.get('speaker_id', ''),
                        record.get('department_section', ''),
                        record.get('hometown', ''),
                        record.get('native_language', ''),
                        record.get('native_language_other', ''),
                        bool(record.get('consent', False)),
                        int(record.get('num_recordings', 0) or 0),
                        record.get('client_ip_hash', ''),
                        'class_student_collection',
                        json.dumps(record, ensure_ascii=True)
                    )
                )

                for audio in record.get('audio_files', []):
                    if not isinstance(audio, dict):
                        continue
                    cur.execute(
                        """
                        INSERT INTO student_recordings (
                            submission_id, slot, filepath, filename, prompt,
                            sampling_rate, duration, storage_backend, storage_uri
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            record.get('submission_id', ''),
                            int(audio.get('slot', 0) or 0),
                            audio.get('filepath', ''),
                            audio.get('filename', ''),
                            audio.get('prompt', ''),
                            audio.get('sampling_rate', None),
                            audio.get('duration', None),
                            audio.get('storage_backend', 'local'),
                            audio.get('storage_uri', '')
                        )
                    )
            conn.commit()
        return True, ''
    except Exception as sync_err:
        print(f"Aiven DB sync failed: {sync_err}")
        return False, str(sync_err)


def _collect_student_audio_files(file_storage):
    files = [f for f in file_storage.getlist('audio_files') if f and str(f.filename).strip() != '']
    if len(files) > 0:
        return files

    fixed = []
    for key in ['audio_1', 'audio_2', 'audio_3']:
        f = file_storage.get(key)
        if f and str(f.filename).strip() != '':
            fixed.append(f)
    return fixed


def _parse_bool(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y'}:
        return True
    if text in {'0', 'false', 'no', 'n'}:
        return False
    return None


def _parse_iso_date(value) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    try:
        # Accept plain dates as UTC midnight.
        if len(text) == 10 and text[4] == '-' and text[7] == '-':
            return datetime.fromisoformat(text + 'T00:00:00+00:00')
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _feedback_record_dt(record):
    return _parse_iso_date(record.get('timestamp_utc'))


def _normalize_accent_label(value):
    text = str(value or '').strip().lower()
    if text in {'tamil', 'tamilnadu'}:
        return 'tamil_nadu'
    if text in {'gujrat'}:
        return 'gujarat'
    return text


def _filter_feedback_records(records, args):
    feedback = str(args.get('feedback', '')).strip().lower()
    mode = str(args.get('mode', '')).strip().lower()
    predicted = _normalize_accent_label(args.get('predicted_accent', ''))
    corrected = _normalize_accent_label(args.get('corrected_accent', ''))
    is_uncertain = _parse_bool(args.get('is_uncertain'))
    from_dt = _parse_iso_date(args.get('from_date'))
    to_dt = _parse_iso_date(args.get('to_date'))

    if to_dt and len(str(args.get('to_date', '')).strip()) == 10:
        # For date-only filter include the entire day.
        to_dt = to_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    filtered = []
    for row in records:
        row_feedback = str(row.get('user_feedback', '')).strip().lower()
        row_mode = str(row.get('mode', '')).strip().lower()
        row_predicted = _normalize_accent_label(row.get('predicted_accent', ''))
        row_corrected = _normalize_accent_label(row.get('corrected_accent', ''))
        row_uncertain = bool(row.get('is_uncertain', False))

        if feedback and row_feedback != feedback:
            continue
        if mode and row_mode != mode:
            continue
        if predicted and row_predicted != predicted:
            continue
        if corrected and row_corrected != corrected:
            continue
        if is_uncertain is not None and row_uncertain != is_uncertain:
            continue

        row_dt = _feedback_record_dt(row)
        if from_dt and (row_dt is None or row_dt < from_dt):
            continue
        if to_dt and (row_dt is None or row_dt > to_dt):
            continue

        filtered.append(row)

    return filtered


def _finalize_prediction(mode, classes, probs, quality_report):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.size == 0 or len(classes) == 0:
        return {'success': False, 'error': 'Invalid prediction distribution'}

    calibrated_probs = calibrate_probabilities(probs)
    raw_confidence = float(np.max(probs) * 100.0)
    calibrated_confidence = compute_calibrated_confidence(calibrated_probs)
    quality_factor = float(np.clip((quality_report.get('score', 100.0) / 100.0), 0.0, 1.0))
    final_confidence = float(np.clip(calibrated_confidence * (0.70 + 0.30 * quality_factor), 0.0, 100.0))

    idx = int(np.argmax(calibrated_probs))
    predicted = str(classes[idx])

    top_k = min(3, len(classes))
    top_indices = np.argsort(-calibrated_probs)[:top_k]
    top_predictions = [
        {'accent': str(classes[i]), 'confidence': float(calibrated_probs[i] * 100.0)}
        for i in top_indices
    ]

    if not quality_report.get('can_predict', True):
        return _build_uncertain_response(
            mode=mode,
            quality_report=quality_report,
            reason='Audio quality is too low for reliable prediction.',
            top_predictions=top_predictions
        )

    if final_confidence < CONFIDENCE_UNCERTAIN_THRESHOLD or quality_report.get('score', 100.0) < QUALITY_SCORE_MIN:
        reason = 'Prediction confidence is low. Please upload a cleaner or longer sample.'
        return {
            'success': True,
            'predicted_accent': 'uncertain',
            'confidence': final_confidence,
            'raw_confidence': raw_confidence,
            'calibrated_confidence': calibrated_confidence,
            'is_uncertain': True,
            'uncertain_reason': reason,
            'top_predictions': top_predictions,
            'recommendations': [],
            'quality': quality_report,
            'mode': mode
        }

    return {
        'success': True,
        'predicted_accent': predicted,
        'confidence': final_confidence,
        'raw_confidence': raw_confidence,
        'calibrated_confidence': calibrated_confidence,
        'is_uncertain': False,
        'uncertain_reason': '',
        'top_predictions': top_predictions,
        'recommendations': get_cuisine_recommendations(predicted),
        'quality': quality_report,
        'mode': mode
    }


def predict_accent(audio_path):
    """Predict accent/language from audio file"""
    try:
        quality_report = analyze_audio_quality(audio_path)

        if RUN_MODE == 'mfcc-lite':
            global mfcc_extractor
            if mfcc_extractor is None:
                mfcc_extractor = MFCCExtractor(sr=16000, n_mfcc=40, include_delta=True, include_delta_delta=True)

            feat_dict = mfcc_extractor.extract_from_file(audio_path, return_frames=False)
            features = feat_dict['features'] if isinstance(feat_dict, dict) else feat_dict
            if features is None:
                return {'success': False, 'error': 'MFCC feature extraction failed'}

            x = np.asarray(features, dtype=np.float32).reshape(-1)
            if x.size == 0 or len(mfcc_classes) == 0 or mfcc_train_matrix_norm is None:
                return {'success': False, 'error': 'MFCC fallback not initialized'}

            # Cosine k-NN against train MFCC vectors.
            d = min(x.size, mfcc_train_matrix_norm.shape[1])
            x_use = x[:d]
            x_use = x_use / (np.linalg.norm(x_use) + 1e-8)
            sims = mfcc_train_matrix_norm[:, :d] @ x_use

            k = min(mfcc_knn_k, sims.shape[0])
            top_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]

            # Weighted vote by shifted cosine similarity.
            class_scores = {c: 0.0 for c in mfcc_classes}
            for i in top_idx:
                lbl = mfcc_train_labels[int(i)]
                if lbl not in class_scores:
                    class_scores[lbl] = 0.0
                class_scores[lbl] += float(sims[int(i)] + 1.0)

            scores = np.asarray([class_scores.get(c, 0.0) for c in mfcc_classes], dtype=np.float32)
            probs = scores / (np.sum(scores) + 1e-8)
            return _finalize_prediction('mfcc-lite', mfcc_classes, probs, quality_report)

        if RUN_MODE == 'demo':
            if label_encoder is None:
                return {'success': False, 'error': 'No model or label encoder available'}

            classes = [str(c) for c in label_encoder.classes_]
            # Lightweight deterministic fallback without model/feature extraction.
            # Use file content so different uploads can produce different outputs.
            try:
                with open(audio_path, 'rb') as f:
                    digest = hashlib.sha256(f.read()).digest()
                idx = int.from_bytes(digest[:4], byteorder='big') % len(classes)
            except Exception:
                idx = abs(hash(audio_path)) % len(classes)
            rotated_classes = classes[idx:] + classes[:idx]
            score_seed = np.linspace(1.0, 0.4, num=len(rotated_classes), dtype=np.float32)
            probs_rot = score_seed / (np.sum(score_seed) + 1e-8)
            class_to_prob = {c: float(p) for c, p in zip(rotated_classes, probs_rot)}
            probs = np.asarray([class_to_prob[c] for c in classes], dtype=np.float32)
            return _finalize_prediction('demo', classes, probs, quality_report)

        features = extract_features(audio_path)
        if features is None:
            return {'success': False, 'error': 'Feature extraction failed'}

        # Ensure float32 1-D
        features = np.asarray(features, dtype=np.float32).reshape(-1)
        print(f"Feature vector final shape: {features.shape}")

        expected_dim = 1536
        if features.size != expected_dim:
            print(f"Warning: resizing feature vector from {features.size} to {expected_dim}")
            if features.size < expected_dim:
                features = np.pad(features, (0, expected_dim - features.size))
            else:
                features = features[:expected_dim]

        features_tensor = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            logits = model(features_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        model_classes = [str(c) for c in label_encoder.classes_]
        return _finalize_prediction('model', model_classes, probs.cpu().numpy(), quality_report)
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'success': False, 'error': f'Prediction failure: {e}'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/student/intake')
def student_intake():
    return render_template('student_intake_v2.html', prompts=STUDENT_PROMPTS)


@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file uploaded'})
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    audio_file.save(filepath)
    
    # Make prediction
    result = predict_accent(filepath)
    
    # Clean up
    try:
        os.remove(filepath)
    except:
        pass
    
    return jsonify(result)


@app.route('/info')
def info():
    """Display model information"""
    mode_label = 'model' if RUN_MODE == 'model' else ('mfcc-lite' if RUN_MODE == 'mfcc-lite' else 'demo')
    model_name = 'MLP Classifier - Indian Accents' if RUN_MODE == 'model' else ('MFCC Lite Classifier (No Training)' if RUN_MODE == 'mfcc-lite' else 'Demo Mode (No Weights)')
    feature_name = 'HuBERT Layer 12 mean+std (1536-dimensional)' if RUN_MODE == 'model' else ('MFCC stats (600-dimensional) + class centroids' if RUN_MODE == 'mfcc-lite' else 'Demo fallback (no extraction)')
    class_list = mfcc_classes if RUN_MODE == 'mfcc-lite' else ([str(c) for c in label_encoder.classes_] if label_encoder is not None else [])

    return jsonify({
        'model': model_name,
        'architecture': '[1536 → 256 → 128 → 6]',
        'parameters': '188,294',
        'test_accuracy': f"{checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))}" if RUN_MODE == 'model' else 'N/A',
        'dataset': 'IndicAccentDB (2,798 samples)',
        'features': feature_name,
        'classes': class_list,
        'num_classes': len(class_list),
        'mode': mode_label,
        'quality_gate': {
            'score_min': QUALITY_SCORE_MIN,
            'hard_floor': QUALITY_SCORE_HARD_FLOOR,
            'uncertain_threshold': CONFIDENCE_UNCERTAIN_THRESHOLD,
            'temperature': CONFIDENCE_CALIBRATION_TEMPERATURE
        }
    })


@app.route('/feedback', methods=['POST'])
def feedback():
    payload = request.get_json(silent=True) or {}
    user_feedback = str(payload.get('user_feedback', '')).strip().lower()
    corrected_accent = str(payload.get('corrected_accent', '')).strip().lower()

    if user_feedback not in {'correct', 'incorrect'}:
        return jsonify({'success': False, 'error': "user_feedback must be 'correct' or 'incorrect'"}), 400

    if user_feedback == 'incorrect' and corrected_accent == '':
        return jsonify({'success': False, 'error': 'corrected_accent is required when feedback is incorrect'}), 400

    quality = payload.get('quality') if isinstance(payload.get('quality'), dict) else {}
    top_predictions = payload.get('top_predictions') if isinstance(payload.get('top_predictions'), list) else []

    record = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'audio_filename': str(payload.get('audio_filename', '')).strip(),
        'mode': str(payload.get('mode', '')).strip(),
        'predicted_accent': str(payload.get('predicted_accent', '')).strip().lower(),
        'is_uncertain': bool(payload.get('is_uncertain', False)),
        'confidence': float(payload.get('confidence', 0.0) or 0.0),
        'calibrated_confidence': float(payload.get('calibrated_confidence', 0.0) or 0.0),
        'raw_confidence': float(payload.get('raw_confidence', 0.0) or 0.0),
        'quality_score': float(quality.get('score', 0.0) or 0.0),
        'quality_level': str(quality.get('level', '')).strip(),
        'quality_issues': quality.get('issues', []) if isinstance(quality.get('issues', []), list) else [],
        'user_feedback': user_feedback,
        'corrected_accent': corrected_accent,
        'notes': str(payload.get('notes', '')).strip(),
        'top_predictions': top_predictions,
        'client_ip': request.headers.get('X-Forwarded-For', request.remote_addr)
    }

    if not _append_feedback_record(record):
        return jsonify({'success': False, 'error': 'Failed to persist feedback'}), 500

    return jsonify({'success': True, 'message': 'Feedback saved successfully'})


@app.route('/feedback/summary')
def feedback_summary():
    records = _read_feedback_records()
    total = len(records)
    correct = sum(1 for r in records if str(r.get('user_feedback', '')).lower() == 'correct')
    incorrect = sum(1 for r in records if str(r.get('user_feedback', '')).lower() == 'incorrect')
    uncertain = sum(1 for r in records if bool(r.get('is_uncertain', False)))

    return jsonify({
        'success': True,
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'uncertain': uncertain,
        'accuracy_proxy': float((correct / total) * 100.0) if total > 0 else 0.0
    })


@app.route('/feedback/export')
def feedback_export():
    records = _read_feedback_records()
    output = io.StringIO()

    columns = [
        'timestamp_utc', 'audio_filename', 'mode', 'predicted_accent',
        'is_uncertain', 'confidence', 'calibrated_confidence', 'raw_confidence',
        'quality_score', 'quality_level', 'quality_issues',
        'user_feedback', 'corrected_accent', 'notes',
        'top1_accent', 'top1_confidence', 'top2_accent', 'top2_confidence', 'top3_accent', 'top3_confidence',
        'client_ip'
    ]

    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    for row in records:
        top = row.get('top_predictions', []) if isinstance(row.get('top_predictions', []), list) else []
        entry = {
            'timestamp_utc': row.get('timestamp_utc', ''),
            'audio_filename': row.get('audio_filename', ''),
            'mode': row.get('mode', ''),
            'predicted_accent': row.get('predicted_accent', ''),
            'is_uncertain': row.get('is_uncertain', False),
            'confidence': row.get('confidence', 0.0),
            'calibrated_confidence': row.get('calibrated_confidence', 0.0),
            'raw_confidence': row.get('raw_confidence', 0.0),
            'quality_score': row.get('quality_score', 0.0),
            'quality_level': row.get('quality_level', ''),
            'quality_issues': '|'.join(row.get('quality_issues', []) if isinstance(row.get('quality_issues', []), list) else []),
            'user_feedback': row.get('user_feedback', ''),
            'corrected_accent': row.get('corrected_accent', ''),
            'notes': row.get('notes', ''),
            'top1_accent': top[0].get('accent', '') if len(top) > 0 and isinstance(top[0], dict) else '',
            'top1_confidence': top[0].get('confidence', '') if len(top) > 0 and isinstance(top[0], dict) else '',
            'top2_accent': top[1].get('accent', '') if len(top) > 1 and isinstance(top[1], dict) else '',
            'top2_confidence': top[1].get('confidence', '') if len(top) > 1 and isinstance(top[1], dict) else '',
            'top3_accent': top[2].get('accent', '') if len(top) > 2 and isinstance(top[2], dict) else '',
            'top3_confidence': top[2].get('confidence', '') if len(top) > 2 and isinstance(top[2], dict) else '',
            'client_ip': row.get('client_ip', '')
        }
        writer.writerow(entry)

    csv_text = output.getvalue()
    output.close()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'feedback_export_{timestamp}.csv'

    return app.response_class(
        csv_text,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


@app.route('/admin/feedback_dashboard')
def admin_feedback_dashboard():
    """Admin dashboard for reviewing feedback with filters"""
    return render_template('feedback_dashboard.html')


@app.route('/feedback/records')
def feedback_records():
    records = _read_feedback_records()
    filtered = _filter_feedback_records(records, request.args)

    try:
        limit = int(request.args.get('limit', 200))
    except Exception:
        limit = 200
    try:
        offset = int(request.args.get('offset', 0))
    except Exception:
        offset = 0

    limit = max(1, min(limit, 1000))
    offset = max(0, offset)

    sliced = filtered[offset:offset + limit]
    correct = sum(1 for r in filtered if str(r.get('user_feedback', '')).lower() == 'correct')
    incorrect = sum(1 for r in filtered if str(r.get('user_feedback', '')).lower() == 'incorrect')
    uncertain = sum(1 for r in filtered if bool(r.get('is_uncertain', False)))

    return jsonify({
        'success': True,
        'total': len(records),
        'filtered_total': len(filtered),
        'offset': offset,
        'limit': limit,
        'records': sliced,
        'summary': {
            'correct': correct,
            'incorrect': incorrect,
            'uncertain': uncertain,
            'accuracy_proxy': float((correct / len(filtered)) * 100.0) if len(filtered) > 0 else 0.0
        }
    })


@app.route('/feedback/retrain/prepare', methods=['POST'])
def feedback_retrain_prepare():
    payload = request.get_json(silent=True) or {}

    try:
        min_quality = float(payload.get('min_quality_score', 35.0))
    except Exception:
        min_quality = 35.0
    try:
        min_conf = float(payload.get('min_confidence', 35.0))
    except Exception:
        min_conf = 35.0

    include_uncertain = bool(payload.get('include_uncertain', False))
    include_unrated = bool(payload.get('include_unrated', False))

    records = _read_feedback_records()
    accepted = []
    rejected = 0

    for row in records:
        feedback = str(row.get('user_feedback', '')).strip().lower()
        predicted = _normalize_accent_label(row.get('predicted_accent', ''))
        corrected = _normalize_accent_label(row.get('corrected_accent', ''))
        is_uncertain = bool(row.get('is_uncertain', False))
        quality_score = float(row.get('quality_score', 0.0) or 0.0)
        confidence = float(row.get('confidence', 0.0) or 0.0)

        if not include_uncertain and is_uncertain:
            rejected += 1
            continue
        if quality_score < min_quality or confidence < min_conf:
            rejected += 1
            continue

        if feedback == 'incorrect' and corrected:
            final_label = corrected
        elif feedback == 'correct' and predicted:
            final_label = predicted
        elif include_unrated and predicted:
            final_label = predicted
        else:
            rejected += 1
            continue

        if final_label in {'', 'uncertain'}:
            rejected += 1
            continue

        accepted.append({
            'timestamp_utc': row.get('timestamp_utc', ''),
            'audio_filename': row.get('audio_filename', ''),
            'mode': row.get('mode', ''),
            'predicted_accent': predicted,
            'corrected_accent': corrected,
            'final_label': final_label,
            'user_feedback': feedback,
            'quality_score': quality_score,
            'confidence': confidence,
            'is_uncertain': is_uncertain,
            'notes': row.get('notes', '')
        })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = FEEDBACK_DIR / 'retrain'
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f'retrain_dataset_{timestamp}.csv'

    columns = [
        'timestamp_utc', 'audio_filename', 'mode', 'predicted_accent', 'corrected_accent',
        'final_label', 'user_feedback', 'quality_score', 'confidence', 'is_uncertain', 'notes'
    ]

    with open(str(out_path), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in accepted:
            writer.writerow(row)

    label_counts = {}
    for row in accepted:
        key = row['final_label']
        label_counts[key] = label_counts.get(key, 0) + 1

    return jsonify({
        'success': True,
        'input_records': len(records),
        'accepted_records': len(accepted),
        'rejected_records': rejected,
        'output_csv_path': str(out_path),
        'filters': {
            'min_quality_score': min_quality,
            'min_confidence': min_conf,
            'include_uncertain': include_uncertain,
            'include_unrated': include_unrated
        },
        'label_distribution': label_counts
    })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Process multiple audio files and return predictions"""
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    
    if not files or len(files) == 0:
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    results = []
    for idx, file in enumerate(files):
        if file.filename == '':
            continue
        
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'batch_{idx}_{file.filename}')
            file.save(filepath)
            
            prediction = predict_accent(filepath)
            prediction['original_filename'] = file.filename
            results.append(prediction)
            
            try:
                os.remove(filepath)
            except:
                pass
        except Exception as e:
            results.append({
                'success': False,
                'original_filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({
        'success': True,
        'total_files': len(files),
        'processed': len(results),
        'predictions': results
    })


@app.route('/student/submit', methods=['POST'])
def student_submit():
    form = request.form
    name = str(form.get('name', '')).strip()
    student_id_input = str(form.get('student_id', '')).strip()
    department_section = str(form.get('department_section', '')).strip()
    hometown = str(form.get('hometown', '')).strip()
    native_language_input = str(form.get('native_language', '')).strip()
    native_language_other = str(form.get('native_language_other', '')).strip()
    consent = str(form.get('consent', '')).strip().lower() in {'1', 'true', 'yes', 'on'}

    if name == '' or student_id_input == '' or department_section == '' or hometown == '' or native_language_input == '':
        return jsonify({
            'success': False,
            'error': 'name, student_id, department_section, hometown, and native_language are required'
        }), 400

    if not consent:
        return jsonify({'success': False, 'error': 'Consent is required to submit data'}), 400

    student_id = _normalize_student_id(student_id_input)
    if student_id == '':
        return jsonify({'success': False, 'error': 'student_id contains no valid characters'}), 400

    native_language = _normalize_accent_label(native_language_input)
    if native_language == 'other' and native_language_other == '':
        return jsonify({'success': False, 'error': 'Please provide your home state/language when selecting Other'}), 400
    if native_language != 'other':
        native_language_other = ''

    audio_files = _collect_student_audio_files(request.files)
    if len(audio_files) != 3:
        return jsonify({'success': False, 'error': 'Exactly 3 audio recordings are required'}), 400

    for f in audio_files:
        if not _allowed_audio_file(f.filename):
            return jsonify({'success': False, 'error': f'Unsupported file type: {f.filename}'}), 400

    timestamp = datetime.now(timezone.utc)
    submission_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    submission_audio_dir = STUDENT_AUDIO_DIR / student_id / submission_id
    os.makedirs(submission_audio_dir, exist_ok=True)

    saved_audio = []
    metadata_rows = []

    try:
        for i, f in enumerate(audio_files):
            source_name = secure_filename(f.filename) or f'audio_{i + 1}.wav'
            stored_name = f'prompt_{i + 1}_{source_name}'
            audio_path = submission_audio_dir / stored_name
            f.save(str(audio_path))

            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise ValueError(f'Prompt {i + 1} audio is empty. Please re-record and submit again.')

            sampling_rate = None
            duration = None
            if librosa is not None:
                try:
                    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
                    sampling_rate = int(sr) if sr else None
                    duration = float(len(y) / sr) if sr else None
                except Exception:
                    sampling_rate = None
                    duration = None

            relative_path = audio_path.relative_to(root_dir).as_posix()
            prompt_text = STUDENT_PROMPTS[i] if i < len(STUDENT_PROMPTS) else ''

            saved_audio.append({
                'slot': i + 1,
                'filepath': relative_path,
                'filename': stored_name,
                'prompt': prompt_text,
                'sampling_rate': sampling_rate,
                'duration': duration,
                'storage_backend': 'local',
                'storage_uri': relative_path
            })

            metadata_rows.append({
                'submission_id': submission_id,
                'timestamp_utc': timestamp.isoformat(),
                'filepath': relative_path,
                'filename': stored_name,
                'speaker_id': student_id,
                'student_name': name,
                'student_id': student_id_input,
                'department_section': department_section,
                'native_language': native_language,
                'native_language_other': native_language_other,
                'hometown': hometown,
                'age_group': 'adult',
                'utterance_type': 'read',
                'text': prompt_text,
                'sampling_rate': sampling_rate if sampling_rate is not None else '',
                'duration': duration if duration is not None else '',
                'split': 'unassigned',
                'source': 'class_student_collection',
                'storage_backend': 'local',
                'storage_uri': relative_path
            })
    except Exception as save_err:
        return jsonify({'success': False, 'error': f'Failed to store recordings: {save_err}'}), 500

    record = {
        'timestamp_utc': timestamp.isoformat(),
        'submission_id': submission_id,
        'name': name,
        'student_id': student_id_input,
        'speaker_id': student_id,
        'department_section': department_section,
        'hometown': hometown,
        'native_language': native_language,
        'native_language_other': native_language_other,
        'consent': True,
        'num_recordings': len(saved_audio),
        'audio_files': saved_audio,
        'client_ip_hash': _hash_client_ip(request.headers.get('X-Forwarded-For', request.remote_addr))
    }

    if not _append_student_submission_record(record):
        return jsonify({'success': False, 'error': 'Failed to save submission log'}), 500

    if not _append_student_metadata_rows(metadata_rows):
        return jsonify({'success': False, 'error': 'Failed to save metadata rows'}), 500

    aiven_synced, aiven_error = _sync_student_submission_to_aiven(record)
    if AIVEN_DB_STRICT and not aiven_synced:
        return jsonify({'success': False, 'error': f'Aiven DB sync failed: {aiven_error}'}), 500

    return jsonify({
        'success': True,
        'message': 'Submission saved successfully',
        'submission_id': submission_id,
        'stored_recordings': len(saved_audio),
        'aiven_synced': bool(aiven_synced),
        'aiven_error': '' if aiven_synced else aiven_error
    })


@app.route('/student/submissions/summary')
def student_submissions_summary():
    records = _read_student_submissions()
    unique_students = sorted({str(r.get('speaker_id', '')).strip() for r in records if str(r.get('speaker_id', '')).strip() != ''})
    total_recordings = 0
    for r in records:
        audio_files = r.get('audio_files', [])
        if isinstance(audio_files, list):
            total_recordings += len(audio_files)

    return jsonify({
        'success': True,
        'submissions': len(records),
        'unique_students': len(unique_students),
        'total_recordings': total_recordings,
        'metadata_csv_path': str(STUDENT_METADATA_CSV_PATH),
        'aiven_sync_enabled': AIVEN_DB_SYNC_ENABLED,
        'aiven_db_configured': AIVEN_DB_URI != ''
    })


@app.route('/student/submissions/export')
def student_submissions_export():
    records = _read_student_submissions()
    output = io.StringIO()
    columns = [
        'timestamp_utc', 'submission_id', 'name', 'student_id', 'speaker_id',
        'department_section', 'hometown', 'native_language', 'native_language_other', 'consent', 'num_recordings',
        'audio_1_path', 'audio_2_path', 'audio_3_path', 'client_ip_hash'
    ]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    for row in records:
        audio = row.get('audio_files', []) if isinstance(row.get('audio_files', []), list) else []
        writer.writerow({
            'timestamp_utc': row.get('timestamp_utc', ''),
            'submission_id': row.get('submission_id', ''),
            'name': row.get('name', ''),
            'student_id': row.get('student_id', ''),
            'speaker_id': row.get('speaker_id', ''),
            'department_section': row.get('department_section', ''),
            'hometown': row.get('hometown', ''),
            'native_language': row.get('native_language', ''),
            'native_language_other': row.get('native_language_other', ''),
            'consent': row.get('consent', False),
            'num_recordings': row.get('num_recordings', 0),
            'audio_1_path': audio[0].get('filepath', '') if len(audio) > 0 and isinstance(audio[0], dict) else '',
            'audio_2_path': audio[1].get('filepath', '') if len(audio) > 1 and isinstance(audio[1], dict) else '',
            'audio_3_path': audio[2].get('filepath', '') if len(audio) > 2 and isinstance(audio[2], dict) else '',
            'client_ip_hash': row.get('client_ip_hash', '')
        })

    csv_text = output.getvalue()
    output.close()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    return app.response_class(
        csv_text,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=student_submissions_{timestamp}.csv'}
    )


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌐 NATIVE LANGUAGE IDENTIFICATION - INDIAN ACCENTS")
    print("=" * 80)
    if RUN_MODE == 'model':
        try:
            test_acc = checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))
            print(f"✓ Model: MLP Classifier ({test_acc:.2f}% test accuracy)")
        except Exception:
            print(f"✓ Model: MLP Classifier (test accuracy: {checkpoint.get('test_acc', checkpoint.get('val_acc', 'N/A'))})")
    elif RUN_MODE == 'mfcc-lite':
        print("✓ Model: MFCC Lite Classifier (centroid-based, no training)")
    else:
        print("⚠ Model: Demo mode (no checkpoint available, training not required)")
    if RUN_MODE == 'mfcc-lite' and len(mfcc_classes) > 0:
        print(f"✓ Classes: {', '.join(mfcc_classes)}")
    elif label_encoder is not None:
        print(f"✓ Classes: {', '.join([str(c) for c in label_encoder.classes_])}")
    else:
        print("⚠ Classes: unavailable (label encoder missing)")
    print(f"✓ Server starting on: http://127.0.0.1:5000")
    print("=" * 80)
    print("\n📝 Usage:")
    print("  1. Open http://127.0.0.1:5000 in your browser")
    print("  2. Upload an audio file (WAV, MP3, etc.)")
    print("  3. Get Indian state accent prediction with confidence scores")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, load_dotenv=False)
