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


@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def root():
    return render_template('student_intake_v2.html', prompts=STUDENT_PROMPTS)


@app.route('/student/intake')
def student_intake():
    return render_template('student_intake_v2.html', prompts=STUDENT_PROMPTS)


@app.route('/health')
def health():
    return jsonify({'success': True, 'service': 'vercel-intake'})


@app.route('/student/submissions/summary')
def student_submissions_summary():
    try:
        with _db_connect() as conn:
            _ensure_tables(conn)
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM student_submissions')
                submissions = int(cur.fetchone()[0])
                cur.execute('SELECT COUNT(DISTINCT speaker_id) FROM student_submissions')
                unique_students = int(cur.fetchone()[0])
                cur.execute('SELECT COALESCE(SUM(num_recordings), 0) FROM student_submissions')
                total_recordings = int(cur.fetchone()[0] or 0)

        return jsonify({
            'success': True,
            'submissions': submissions,
            'unique_students': unique_students,
            'total_recordings': total_recordings,
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
            'unique_students': 0,
            'total_recordings': 0,
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
