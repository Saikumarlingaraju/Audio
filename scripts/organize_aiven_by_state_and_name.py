"""
Export student recordings from Aiven and organize as:
state -> student_id__student_name -> submission_id -> audio files

Also writes training-oriented metadata and summary CSV files.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.index import _get_aiven_db_uri


def slugify(text: str, fallback: str = "unknown") -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or fallback


def pick_state(native_language: str, native_language_other: str) -> str:
    nl = str(native_language or "").strip().lower()
    other = str(native_language_other or "").strip().lower()
    if nl == "other":
        return other or "other"
    return nl or "unknown"


def infer_extension(filename: str, mime_type: str) -> str:
    mime_to_ext = {
        "audio/webm": "webm",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
    }

    mt = str(mime_type or "").strip().lower()
    if mt in mime_to_ext:
        return mime_to_ext[mt]

    name = str(filename or "").strip()
    if "." in name:
        ext = name.rsplit(".", 1)[1].lower()
        if ext:
            return ext

    return "webm"


def export_dataset(output_dir: Path, min_bytes: int) -> None:
    uri = _get_aiven_db_uri()
    if not uri:
        raise RuntimeError("Aiven DB URI is missing. Configure AIVEN_DB_URI or DATABASE_URL.")

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata_all.csv"
    summary_path = output_dir / "state_student_summary.csv"

    metadata_columns = [
        "filepath",
        "filename",
        "speaker_id",
        "student_name",
        "student_id",
        "native_language",
        "native_language_other",
        "hometown",
        "state_group",
        "submission_id",
        "slot",
        "text",
        "mime_type",
        "size_bytes",
        "is_tiny",
        "age_group",
        "utterance_type",
        "split",
        "source",
    ]

    summary_columns = [
        "state_group",
        "student_id",
        "student_name",
        "recordings",
        "tiny_recordings",
        "total_bytes",
    ]

    state_students = defaultdict(lambda: defaultdict(lambda: {"count": 0, "tiny": 0, "bytes": 0, "name": ""}))

    written = 0
    skipped_too_small = 0
    read_rows = 0

    with open(metadata_path, "w", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=metadata_columns)
        writer.writeheader()

        with psycopg2.connect(uri) as conn:
            with conn.cursor(name="aiven_export_cursor") as cur:
                cur.itersize = 100
                cur.execute(
                    """
                    SELECT
                        s.submission_id,
                        s.student_name,
                        s.student_id,
                        s.speaker_id,
                        s.native_language,
                        s.native_language_other,
                        s.hometown,
                        s.timestamp_utc,
                        r.slot,
                        r.filename,
                        r.prompt,
                        r.mime_type,
                        r.size_bytes,
                        r.audio_blob
                    FROM student_submissions s
                    JOIN student_recordings r ON r.submission_id = s.submission_id
                    WHERE COALESCE(r.size_bytes, 0) >= %s
                    ORDER BY s.timestamp_utc ASC, s.submission_id ASC, r.slot ASC
                    """,
                    (int(min_bytes),),
                )

                while True:
                    batch = cur.fetchmany(100)
                    if not batch:
                        break

                    for row in batch:
                        read_rows += 1
                        (
                            submission_id,
                            student_name,
                            student_id,
                            speaker_id,
                            native_language,
                            native_language_other,
                            hometown,
                            _timestamp_utc,
                            slot,
                            filename,
                            prompt,
                            mime_type,
                            size_bytes,
                            audio_blob,
                        ) = row

                        state_raw = pick_state(native_language, native_language_other)
                        state_key = slugify(state_raw)
                        student_id_text = str(student_id or "").strip()
                        name_slug = slugify(str(student_name or "unknown"), fallback="unknown")
                        student_dir = f"{slugify(student_id_text, fallback='unknown')}__{name_slug}"

                        size = int(size_bytes or 0)
                        is_tiny = size <= 10

                        ext = infer_extension(str(filename or ""), str(mime_type or ""))
                        out_name = f"prompt_{int(slot)}.{ext}"
                        rel_path = Path("organized_by_state") / state_key / student_dir / str(submission_id) / out_name
                        abs_path = output_dir / state_key / student_dir / str(submission_id) / out_name
                        abs_path.parent.mkdir(parents=True, exist_ok=True)

                        blob_bytes = bytes(audio_blob) if audio_blob is not None else b""
                        if len(blob_bytes) == 0:
                            skipped_too_small += 1
                            continue

                        abs_path.write_bytes(blob_bytes)
                        written += 1

                        writer.writerow(
                            {
                                "filepath": rel_path.as_posix(),
                                "filename": out_name,
                                "speaker_id": str(speaker_id or student_id_text),
                                "student_name": str(student_name or ""),
                                "student_id": student_id_text,
                                "native_language": str(native_language or ""),
                                "native_language_other": str(native_language_other or ""),
                                "hometown": str(hometown or ""),
                                "state_group": state_key,
                                "submission_id": str(submission_id),
                                "slot": int(slot),
                                "text": str(prompt or ""),
                                "mime_type": str(mime_type or ""),
                                "size_bytes": size,
                                "is_tiny": int(is_tiny),
                                "age_group": "adult",
                                "utterance_type": "read",
                                "split": "unassigned",
                                "source": "aiven_student_collection",
                            }
                        )

                        student_entry = state_students[state_key][student_id_text]
                        student_entry["count"] += 1
                        student_entry["tiny"] += int(is_tiny)
                        student_entry["bytes"] += size
                        student_entry["name"] = str(student_name or "")

                    print(f"Processed rows so far: {read_rows} | written: {written}")

    with open(summary_path, "w", encoding="utf-8", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=summary_columns)
        writer.writeheader()
        for state_group in sorted(state_students.keys()):
            for sid in sorted(state_students[state_group].keys()):
                info = state_students[state_group][sid]
                writer.writerow(
                    {
                        "state_group": state_group,
                        "student_id": sid,
                        "student_name": info["name"],
                        "recordings": info["count"],
                        "tiny_recordings": info["tiny"],
                        "total_bytes": info["bytes"],
                    }
                )

    print(f"Export complete: {written} recordings written")
    print(f"Skipped for size < {min_bytes} bytes: {skipped_too_small}")
    print(f"Metadata: {metadata_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize Aiven student data by state and student name")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/collections/class_students/organized_by_state",
        help="Root output directory for organized files",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1,
        help="Skip recordings smaller than this many bytes",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    export_dataset(output_dir=out_dir, min_bytes=max(1, int(args.min_bytes)))


if __name__ == "__main__":
    main()
