import csv
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import psycopg2
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT_DIR / 'app' / 'templates'

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}
STUDENT_PROMPTS = [
    'The quick brown fox jumps over the lazy dog near the village market. Please speak clearly at a natural pace for ten seconds.',
    'Today the teacher checked thirty thin notebooks, three thick textbooks, and thirteen blue pens before the class discussion began.',
    'Strong winds moved across green fields, bright roads, and small houses while children played, laughed, and shared stories after school.'
]

SUPPORTED_ACCENTS = [
    'andhra_pradesh',
    'gujarat',
    'jharkhand',
    'karnataka',
    'kerala',
    'tamil_nadu',
]

ACCENT_CUISINE_MAP = {
    'andhra_pradesh': [
        {'name': 'Pesarattu', 'region': 'Andhra Pradesh', 'description': 'Green gram dosa with ginger-forward flavor.'},
        {'name': 'Gongura Pachadi', 'region': 'Andhra Pradesh', 'description': 'Tangy sorrel leaves chutney, commonly served with rice.'},
    ],
    'gujarat': [
        {'name': 'Khaman Dhokla', 'region': 'Gujarat', 'description': 'Steamed gram flour snack with mustard tempering.'},
        {'name': 'Undhiyu', 'region': 'Gujarat', 'description': 'Seasonal mixed vegetable dish cooked with herbs and spices.'},
    ],
    'jharkhand': [
        {'name': 'Dhuska', 'region': 'Jharkhand', 'description': 'Deep-fried rice and lentil bread often paired with curry.'},
        {'name': 'Thekua', 'region': 'Jharkhand', 'description': 'Traditional jaggery and wheat flour festive snack.'},
    ],
    'karnataka': [
        {'name': 'Bisi Bele Bath', 'region': 'Karnataka', 'description': 'Lentil-rice dish with vegetables and aromatic spice blend.'},
        {'name': 'Neer Dosa', 'region': 'Karnataka', 'description': 'Thin rice crepes popular in coastal Karnataka cuisine.'},
    ],
    'kerala': [
        {'name': 'Appam with Stew', 'region': 'Kerala', 'description': 'Soft lacy rice pancakes served with coconut-based stew.'},
        {'name': 'Puttu and Kadala', 'region': 'Kerala', 'description': 'Steamed rice cylinders with black chickpea curry.'},
    ],
    'tamil_nadu': [
        {'name': 'Sambar Rice', 'region': 'Tamil Nadu', 'description': 'Lentil tamarind gravy mixed with rice and vegetables.'},
        {'name': 'Kothu Parotta', 'region': 'Tamil Nadu', 'description': 'Layered parotta chopped and tossed with spices.'},
    ],
}


def _extract_aiven_db_uri_from_text(raw_text):
    text = str(raw_text or '')
    if text.strip() == '':
        return ''

    psql_match = re.search(r"psql\s+['\"](postgres(?:ql)?://[^'\"]+)['\"]", text, flags=re.IGNORECASE)
    if psql_match:
        return psql_match.group(1).strip()

    uri_match = re.search(r'postgres(?:ql)?://[^\s]+', text, flags=re.IGNORECASE)
    if uri_match:
        return uri_match.group(0).strip().strip('"\'')

    lines = [line.strip() for line in text.splitlines() if line.strip()]
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
            return _extract_aiven_db_uri_from_text(value)

    env_file = ROOT_DIR / '.env'
    if env_file.exists():
        try:
            return _extract_aiven_db_uri_from_text(env_file.read_text(encoding='utf-8'))
        except Exception:
            return ''

    return ''


def _db_connect():
    uri = _get_aiven_db_uri()
    if uri == '':
        raise RuntimeError('Aiven DB URI is missing. Set AIVEN_DB_URI in Vercel Environment Variables.')
    return psycopg2.connect(uri)


def _ensure_tables(conn):
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
                storage_uri TEXT,
                mime_type TEXT,
                size_bytes INTEGER,
                audio_blob BYTEA
            )
            """
        )
        cur.execute("ALTER TABLE student_recordings ADD COLUMN IF NOT EXISTS mime_type TEXT")
        cur.execute("ALTER TABLE student_recordings ADD COLUMN IF NOT EXISTS size_bytes INTEGER")
        cur.execute("ALTER TABLE student_recordings ADD COLUMN IF NOT EXISTS audio_blob BYTEA")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS accent_feedback (
                id BIGSERIAL PRIMARY KEY,
                timestamp_utc TIMESTAMPTZ NOT NULL,
                audio_filename TEXT,
                mode TEXT,
                predicted_accent TEXT,
                corrected_accent TEXT,
                user_feedback TEXT,
                is_uncertain BOOLEAN,
                confidence DOUBLE PRECISION,
                calibrated_confidence DOUBLE PRECISION,
                raw_confidence DOUBLE PRECISION,
                quality_score DOUBLE PRECISION,
                quality_level TEXT,
                quality_issues TEXT,
                notes TEXT,
                top_predictions_json TEXT,
                client_ip TEXT
            )
            """
        )


def _normalize_student_id(value):
    text = str(value or '').strip().lower()
    text = re.sub(r'[^a-z0-9_-]+', '_', text)
    text = text.strip('_')
    return text[:64]


def _normalize_accent_label(value):
    text = str(value or '').strip().lower()
    if text in {'tamil', 'tamilnadu'}:
        return 'tamil_nadu'
    if text in {'gujrat'}:
        return 'gujarat'
    return text


def _allowed_audio_file(filename):
    text = str(filename or '').strip().lower()
    return '.' in text and text.rsplit('.', 1)[1] in ALLOWED_AUDIO_EXTENSIONS


def _hash_client_ip(value):
    text = str(value or '').strip()
    if text == '':
        return ''
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


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


def _state_title(value):
    text = str(value or '').strip().replace('_', ' ')
    return ' '.join(word.capitalize() for word in text.split())


def _predict_from_audio_bytes(filename, data_bytes):
    if not data_bytes:
        return {'success': False, 'error': 'Uploaded audio is empty'}

    # Hash only audio content so renaming a file does not change prediction output.
    digest = hashlib.sha256(data_bytes[:65536]).digest()

    raw_scores = []
    for idx, accent in enumerate(SUPPORTED_ACCENTS):
        chunk = digest[idx * 4:(idx + 1) * 4]
        score = (int.from_bytes(chunk, 'big') + 1) / 4294967297.0
        raw_scores.append((accent, score))

    total = sum(score for _, score in raw_scores)
    probs = [(accent, score / total) for accent, score in raw_scores]
    probs.sort(key=lambda x: x[1], reverse=True)

    predicted_accent = probs[0][0]
    raw_confidence = float(probs[0][1] * 100.0)

    size_bytes = len(data_bytes)
    quality_issues = []
    if size_bytes < 3000:
        quality_issues.append('very_short_audio')
    elif size_bytes < 10000:
        quality_issues.append('short_audio')

    quality_score = 78.0
    if 'very_short_audio' in quality_issues:
        quality_score = 24.0
    elif 'short_audio' in quality_issues:
        quality_score = 49.0

    quality_level = 'good'
    if quality_score < 35:
        quality_level = 'poor'
    elif quality_score < 60:
        quality_level = 'fair'

    calibrated_confidence = raw_confidence
    if quality_score < 60:
        calibrated_confidence = max(20.0, raw_confidence - 18.0)

    top_predictions = []
    for accent, prob in probs[:3]:
        top_predictions.append({
            'accent': accent,
            'confidence': round(float(prob * 100.0), 2)
        })

    # Demo mode is content-hash based and should always be treated as non-production confidence.
    is_uncertain = True
    uncertain_reason = 'Demo fallback mode is active (hash-based, not a trained classifier). Results are for pipeline checks only.'

    return {
        'success': True,
        'mode': 'demo',
        'demo_warning': 'Prediction is generated by deterministic fallback, not by trained model inference.',
        'predicted_accent': predicted_accent,
        'confidence': round(calibrated_confidence, 2),
        'calibrated_confidence': round(calibrated_confidence, 2),
        'raw_confidence': round(raw_confidence, 2),
        'is_uncertain': is_uncertain,
        'uncertain_reason': uncertain_reason,
        'top_predictions': top_predictions,
        'recommendations': ACCENT_CUISINE_MAP.get(predicted_accent, []),
        'quality': {
            'score': round(quality_score, 2),
            'level': quality_level,
            'duration_sec': round(size_bytes / 32000.0, 2),
            'silent_ratio': 0.08,
            'issues': quality_issues,
        }
    }


def _append_feedback_record(record):
    try:
        with _db_connect() as conn:
            _ensure_tables(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO accent_feedback (
                        timestamp_utc, audio_filename, mode, predicted_accent, corrected_accent,
                        user_feedback, is_uncertain, confidence, calibrated_confidence, raw_confidence,
                        quality_score, quality_level, quality_issues, notes, top_predictions_json, client_ip
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(timezone.utc),
                        str(record.get('audio_filename', '')).strip(),
                        str(record.get('mode', '')).strip(),
                        str(record.get('predicted_accent', '')).strip().lower(),
                        str(record.get('corrected_accent', '')).strip().lower(),
                        str(record.get('user_feedback', '')).strip().lower(),
                        bool(record.get('is_uncertain', False)),
                        float(record.get('confidence', 0.0) or 0.0),
                        float(record.get('calibrated_confidence', 0.0) or 0.0),
                        float(record.get('raw_confidence', 0.0) or 0.0),
                        float(record.get('quality_score', 0.0) or 0.0),
                        str(record.get('quality_level', '')).strip(),
                        '|'.join(record.get('quality_issues', []) if isinstance(record.get('quality_issues', []), list) else []),
                        str(record.get('notes', '')).strip(),
                        json.dumps(record.get('top_predictions', []), ensure_ascii=True),
                        str(record.get('client_ip', '')).strip(),
                    )
                )
            conn.commit()
        return True
    except Exception:
        return False


def _read_feedback_records():
    try:
        with _db_connect() as conn:
            _ensure_tables(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp_utc, audio_filename, mode, predicted_accent, corrected_accent,
                           user_feedback, is_uncertain, confidence, calibrated_confidence, raw_confidence,
                           quality_score, quality_level, quality_issues, notes, top_predictions_json, client_ip
                    FROM accent_feedback
                    ORDER BY timestamp_utc DESC
                    """
                )
                rows = cur.fetchall()

        records = []
        for row in rows:
            top_predictions = []
            try:
                top_predictions = json.loads(row[14] or '[]')
                if not isinstance(top_predictions, list):
                    top_predictions = []
            except Exception:
                top_predictions = []

            issues = []
            issues_raw = str(row[12] or '').strip()
            if issues_raw != '':
                issues = [s for s in issues_raw.split('|') if s]

            records.append({
                'timestamp_utc': row[0].isoformat() if row[0] else '',
                'audio_filename': row[1] or '',
                'mode': row[2] or '',
                'predicted_accent': row[3] or '',
                'corrected_accent': row[4] or '',
                'user_feedback': row[5] or '',
                'is_uncertain': bool(row[6]),
                'confidence': float(row[7] or 0.0),
                'calibrated_confidence': float(row[8] or 0.0),
                'raw_confidence': float(row[9] or 0.0),
                'quality_score': float(row[10] or 0.0),
                'quality_level': row[11] or '',
                'quality_issues': issues,
                'notes': row[13] or '',
                'top_predictions': top_predictions,
                'client_ip': row[15] or '',
            })

        return records
    except Exception:
        return []


def _filter_feedback_records(records, args):
    feedback_filter = str(args.get('feedback', '')).strip().lower()
    uncertain_filter = str(args.get('is_uncertain', '')).strip().lower()
    mode_filter = str(args.get('mode', '')).strip().lower()
    predicted_filter = str(args.get('predicted_accent', '')).strip().lower()
    from_date = str(args.get('from_date', '')).strip()
    to_date = str(args.get('to_date', '')).strip()

    filtered = []
    for row in records:
        if feedback_filter and str(row.get('user_feedback', '')).lower() != feedback_filter:
            continue

        if uncertain_filter in {'true', 'false'}:
            expected = uncertain_filter == 'true'
            if bool(row.get('is_uncertain', False)) != expected:
                continue

        if mode_filter and str(row.get('mode', '')).lower() != mode_filter:
            continue

        if predicted_filter and str(row.get('predicted_accent', '')).lower() != predicted_filter:
            continue

        ts_text = str(row.get('timestamp_utc', ''))
        row_date = ts_text[:10] if len(ts_text) >= 10 else ''
        if from_date and row_date and row_date < from_date:
            continue
        if to_date and row_date and row_date > to_date:
            continue

        filtered.append(row)

    return filtered


@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/student/intake')
def student_intake():
    return render_template('student_intake_v2.html', prompts=STUDENT_PROMPTS)


@app.route('/health')
def health():
    return jsonify({'success': True, 'service': 'vercel-intake'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    if not _allowed_audio_file(audio_file.filename):
        return jsonify({'success': False, 'error': f'Unsupported file type: {audio_file.filename}'}), 400

    data = audio_file.read()
    result = _predict_from_audio_bytes(audio_file.filename, data)
    status = 200 if result.get('success') else 400
    return jsonify(result), status


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'No files selected'}), 400

    predictions = []
    for file_obj in files:
        original_name = str(file_obj.filename or '').strip()
        if original_name == '':
            continue

        if not _allowed_audio_file(original_name):
            predictions.append({
                'success': False,
                'original_filename': original_name,
                'error': 'Unsupported file type'
            })
            continue

        data = file_obj.read()
        result = _predict_from_audio_bytes(original_name, data)
        result['original_filename'] = original_name
        predictions.append(result)

    return jsonify({
        'success': True,
        'total_files': len(files),
        'processed': len(predictions),
        'predictions': predictions,
    })


@app.route('/info')
def info():
    return jsonify({
        'model': 'Demo Accent Classifier (Vercel Runtime)',
        'architecture': 'Deterministic hash-based fallback',
        'parameters': 'N/A',
        'test_accuracy': 'N/A',
        'dataset': 'IndicAccentDB',
        'features': 'Audio byte signature fallback',
        'classes': SUPPORTED_ACCENTS,
        'num_classes': len(SUPPORTED_ACCENTS),
        'mode': 'demo',
        'quality_gate': {
            'score_min': 35,
            'hard_floor': 20,
            'uncertain_threshold': 43,
            'temperature': 1.0,
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
    record = {
        'audio_filename': str(payload.get('audio_filename', '')).strip(),
        'mode': str(payload.get('mode', '')).strip(),
        'predicted_accent': str(payload.get('predicted_accent', '')).strip().lower(),
        'corrected_accent': corrected_accent,
        'user_feedback': user_feedback,
        'is_uncertain': bool(payload.get('is_uncertain', False)),
        'confidence': float(payload.get('confidence', 0.0) or 0.0),
        'calibrated_confidence': float(payload.get('calibrated_confidence', 0.0) or 0.0),
        'raw_confidence': float(payload.get('raw_confidence', 0.0) or 0.0),
        'quality_score': float(quality.get('score', 0.0) or 0.0),
        'quality_level': str(quality.get('level', '')).strip(),
        'quality_issues': quality.get('issues', []) if isinstance(quality.get('issues', []), list) else [],
        'notes': str(payload.get('notes', '')).strip(),
        'top_predictions': payload.get('top_predictions', []) if isinstance(payload.get('top_predictions', []), list) else [],
        'client_ip': request.headers.get('X-Forwarded-For', request.remote_addr),
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
        'accuracy_proxy': float((correct / total) * 100.0) if total > 0 else 0.0,
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
        'client_ip',
    ]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    for row in records:
        top = row.get('top_predictions', []) if isinstance(row.get('top_predictions', []), list) else []
        writer.writerow({
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
            'client_ip': row.get('client_ip', ''),
        })

    csv_text = output.getvalue()
    output.close()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return app.response_class(
        csv_text,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=feedback_export_{timestamp}.csv'}
    )


@app.route('/admin/feedback_dashboard')
def admin_feedback_dashboard():
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
            'accuracy_proxy': float((correct / len(filtered)) * 100.0) if len(filtered) > 0 else 0.0,
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
        feedback_label = str(row.get('user_feedback', '')).strip().lower()
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

        if feedback_label == 'incorrect' and corrected:
            final_label = corrected
        elif feedback_label == 'correct' and predicted:
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
            'user_feedback': feedback_label,
            'quality_score': quality_score,
            'confidence': confidence,
            'is_uncertain': is_uncertain,
            'notes': row.get('notes', ''),
        })

    label_distribution = {}
    for row in accepted:
        label = row['final_label']
        label_distribution[label] = label_distribution.get(label, 0) + 1

    return jsonify({
        'success': True,
        'input_records': len(records),
        'accepted_records': len(accepted),
        'rejected_records': rejected,
        'output_csv_path': '/tmp/retrain_dataset_preview.csv',
        'filters': {
            'min_quality_score': min_quality,
            'min_confidence': min_conf,
            'include_uncertain': include_uncertain,
            'include_unrated': include_unrated,
        },
        'label_distribution': label_distribution,
    })


@app.route('/student/submissions/summary')
def student_submissions_summary():
    try:
        with _db_connect() as conn:
            _ensure_tables(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT speaker_id, native_language, native_language_other, num_recordings
                    FROM student_submissions
                    """
                )
                rows = cur.fetchall()

        submissions = len(rows)
        unique_students_set = set()
        total_recordings = 0
        state_breakdown = {}

        for speaker_id, native_language, native_language_other, num_recordings in rows:
            speaker_text = str(speaker_id or '').strip()
            if speaker_text != '':
                unique_students_set.add(speaker_text)

            record_count = int(num_recordings or 0)
            total_recordings += record_count

            raw_state = str(native_language or '').strip()
            if raw_state == '':
                raw_state = str(native_language_other or '').strip()

            state_key = _normalize_accent_label(raw_state)
            if state_key == '':
                state_key = 'unknown'

            if state_key not in state_breakdown:
                state_breakdown[state_key] = {
                    'submissions': 0,
                    'records': 0,
                    'unique_students': set()
                }

            state_breakdown[state_key]['submissions'] += 1
            state_breakdown[state_key]['records'] += record_count
            if speaker_text != '':
                state_breakdown[state_key]['unique_students'].add(speaker_text)

        state_breakdown_json = {}
        for state, stats in sorted(state_breakdown.items(), key=lambda x: x[0]):
            state_breakdown_json[state] = {
                'submissions': int(stats.get('submissions', 0)),
                'records': int(stats.get('records', 0)),
                'unique_students': len(stats.get('unique_students', set()))
            }

        return jsonify({
            'success': True,
            'submissions': submissions,
            'datasets_collected': submissions,
            'unique_students': len(unique_students_set),
            'total_recordings': total_recordings,
            'exact_records_collected': total_recordings,
            'unique_states': len(state_breakdown_json),
            'state_breakdown': state_breakdown_json,
            'aiven_sync_enabled': True,
            'aiven_db_configured': True,
            'storage_backend': 'aiven-postgres-bytea',
            'db_status': 'ok',
            'db_error': ''
        })
    except Exception as err:
        return jsonify({
            'success': True,
            'submissions': 0,
            'datasets_collected': 0,
            'unique_students': 0,
            'total_recordings': 0,
            'exact_records_collected': 0,
            'unique_states': 0,
            'state_breakdown': {},
            'aiven_sync_enabled': True,
            'aiven_db_configured': _get_aiven_db_uri() != '',
            'storage_backend': 'aiven-postgres-bytea',
            'db_status': 'error',
            'db_error': str(err)
        }), 200


@app.route('/student/submissions/export')
def student_submissions_export():
    try:
        with _db_connect() as conn:
            _ensure_tables(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp_utc, submission_id, student_name, student_id, speaker_id,
                           department_section, hometown, native_language, native_language_other,
                           consent, num_recordings, client_ip_hash
                    FROM student_submissions
                    ORDER BY timestamp_utc DESC
                    """
                )
                rows = cur.fetchall()

        output = io.StringIO()
        columns = [
            'timestamp_utc', 'submission_id', 'name', 'student_id', 'speaker_id',
            'department_section', 'hometown', 'native_language', 'native_language_other',
            'consent', 'num_recordings', 'client_ip_hash'
        ]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                'timestamp_utc': row[0],
                'submission_id': row[1],
                'name': row[2],
                'student_id': row[3],
                'speaker_id': row[4],
                'department_section': row[5],
                'hometown': row[6],
                'native_language': row[7],
                'native_language_other': row[8],
                'consent': row[9],
                'num_recordings': row[10],
                'client_ip_hash': row[11]
            })

        csv_text = output.getvalue()
        output.close()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return app.response_class(
            csv_text,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=student_submissions_{timestamp}.csv'}
        )
    except Exception as err:
        return jsonify({'success': False, 'error': f'Export failed: {err}'}), 500


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
    ip_hash = _hash_client_ip(request.headers.get('X-Forwarded-For', request.remote_addr))

    try:
        with _db_connect() as conn:
            _ensure_tables(conn)
            with conn.cursor() as cur:
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
                    'num_recordings': 3,
                    'client_ip_hash': ip_hash,
                    'source': 'vercel_student_collection'
                }

                cur.execute(
                    """
                    INSERT INTO student_submissions (
                        submission_id, timestamp_utc, student_name, student_id, speaker_id,
                        department_section, hometown, native_language, native_language_other,
                        consent, num_recordings, client_ip_hash, source, payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        submission_id,
                        timestamp,
                        name,
                        student_id_input,
                        student_id,
                        department_section,
                        hometown,
                        native_language,
                        native_language_other,
                        True,
                        3,
                        ip_hash,
                        'vercel_student_collection',
                        json.dumps(record, ensure_ascii=True)
                    )
                )

                for i, f in enumerate(audio_files):
                    source_name = secure_filename(f.filename) or f'prompt_{i + 1}.webm'
                    stored_name = f'prompt_{i + 1}_{source_name}'
                    data = f.read()
                    if not data:
                        raise ValueError(f'Prompt {i + 1} audio is empty. Please re-record and submit again.')

                    if len(data) > 16 * 1024 * 1024:
                        raise ValueError(f'Prompt {i + 1} exceeds 16MB file size limit.')

                    pseudo_path = f'aiven://student_audio/{submission_id}/{stored_name}'
                    prompt_text = STUDENT_PROMPTS[i] if i < len(STUDENT_PROMPTS) else ''

                    cur.execute(
                        """
                        INSERT INTO student_recordings (
                            submission_id, slot, filepath, filename, prompt,
                            sampling_rate, duration, storage_backend, storage_uri,
                            mime_type, size_bytes, audio_blob
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            submission_id,
                            i + 1,
                            pseudo_path,
                            stored_name,
                            prompt_text,
                            None,
                            None,
                            'aiven-postgres-bytea',
                            pseudo_path,
                            str(f.mimetype or ''),
                            len(data),
                            psycopg2.Binary(data)
                        )
                    )

            conn.commit()

        return jsonify({
            'success': True,
            'message': 'Submission saved successfully',
            'submission_id': submission_id,
            'stored_recordings': 3,
            'aiven_synced': True,
            'aiven_error': ''
        })

    except Exception as err:
        return jsonify({'success': False, 'error': f'Submission failed: {err}'}), 500
