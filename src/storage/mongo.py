"""
src/storage/mongo.py
Saves parsed paper data and extracted documents to MongoDB.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pymongo import MongoClient
from datetime import datetime
from src.utils.logger import logging

# ── Connection ────────────────────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["arxiv_pipeline"]
collection = db["raw_papers"]
doc_collection = db["document_extractions"]
img_collection  = db["image_metadata"]
exif_collection = db["exif_metadata"]
transcript_collection = db["transcripts"]
pipeline_logs_collection = db["pipeline_logs"]


# ── Save paper ────────────────────────────────────────────────────────────────
def save_to_mongo(data, source="unknown"):
    """Insert a single paper document into MongoDB with metadata."""
    try:
        document = {
            **data,
            "_source": source,
            "_ingested_at": datetime.utcnow().isoformat()
        }
        result = collection.insert_one(document)
        logging.info(f"Saved to MongoDB: {result.inserted_id} (source: {source})")
        return result.inserted_id
    except Exception as e:
        logging.error(f"MongoDB insert failed: {e}")
        return None


# ── Save extracted document ───────────────────────────────────────────────────
def save_document_to_mongo(data, source="unknown"):
    """
    Insert an extracted document into the document_extractions collection.
    Expects data to already contain metadata fields (file_name, document_type, etc.).
    Skips duplicates based on file_name + extraction_timestamp.
    """
    try:
        # Guard: skip if a document with same file_name was already stored today
        existing = doc_collection.find_one({
            "file_name": data.get("file_name"),
            "extraction_timestamp": {"$regex": f"^{datetime.utcnow().strftime('%Y-%m-%d')}"}
        })
        if existing:
            logging.info(
                f"[MongoDB] Skipping duplicate: {data.get('file_name')} already stored today"
            )
            return existing["_id"]

        result = doc_collection.insert_one(data)
        logging.info(
            f"[MongoDB] Stored document extraction: {data.get('file_name')} "
            f"(type={data.get('document_type')}, id={result.inserted_id})"
        )
        return result.inserted_id
    except Exception as e:
        logging.error(f"[MongoDB] Document insert failed for {data.get('file_name', source)}: {e}")
        return None


# ── Save image metadata ───────────────────────────────────────────────────────
def save_image_metadata(data: dict) -> object:
    """
    Upsert image processing metadata into the image_metadata collection.
    Matches on filename; updates the record if it already exists so that
    re-running the pipeline always stores the latest enriched fields.
    """
    try:
        data["processed_at"] = datetime.utcnow().isoformat()
        result = img_collection.update_one(
            {"filename": data.get("filename")},
            {"$set": data},
            upsert=True,
        )
        action = "Updated" if result.matched_count else "Inserted"
        logging.info(
            f"[MongoDB] {action} image metadata: {data.get('filename')} "
            f"(movie={data.get('title')})"
        )
        return result.upserted_id
    except Exception as e:
        logging.error(f"[MongoDB] Image metadata upsert failed: {e}")
        return None


# ── Save EXIF metadata ────────────────────────────────────────────────────────
def save_exif_to_mongo(data: dict) -> object:
    """
    Insert EXIF summary into the exif_metadata collection.
    Skips if the same filename was already stored.
    """
    try:
        existing = exif_collection.find_one({"file": data.get("file")})
        if existing:
            logging.info(f"[MongoDB] Skipping duplicate EXIF: {data.get('file')}")
            return existing["_id"]

        data.setdefault("stored_at", datetime.utcnow().isoformat())
        result = exif_collection.insert_one(data)
        logging.info(
            f"[MongoDB] Stored EXIF metadata: {data.get('file')} "
            f"(camera={data.get('camera_make')} {data.get('camera_model')}, id={result.inserted_id})"
        )
        return result.inserted_id
    except Exception as e:
        logging.error(f"[MongoDB] EXIF insert failed: {e}")
        return None


# ── Save transcript ────────────────────────────────────────────────────────────
def save_transcript_to_mongo(result: dict) -> object:
    """
    Store a transcription result in the transcripts collection.
    Each document is keyed by source_path + transcribed_at to avoid duplicates.
    Expected keys in result: source_path, language, language_probability,
    duration_sec, segments, full_text, model, transcribed_at.
    """
    try:
        existing = transcript_collection.find_one({
            "source_path": result.get("source_path"),
            "model": result.get("model"),
            "transcribed_at": {"$regex": f"^{result.get('transcribed_at', '')[:10]}"},
        })
        if existing:
            logging.info(
                f"[MongoDB] Skipping duplicate transcript: {result.get('source_path')}"
            )
            return existing["_id"]

        doc = {
            **result,
            "stored_at": datetime.utcnow().isoformat(),
        }
        inserted = transcript_collection.insert_one(doc)
        logging.info(
            f"[MongoDB] Stored transcript: {result.get('source_path')} "
            f"(lang={result.get('language')}, model={result.get('model')}, "
            f"id={inserted.inserted_id})"
        )
        return inserted.inserted_id
    except Exception as e:
        logging.error(f"[MongoDB] Transcript insert failed: {e}")
        return None
