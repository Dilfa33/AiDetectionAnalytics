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
