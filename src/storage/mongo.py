"""
src/storage/mongo.py
Saves parsed paper data to MongoDB.
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


# ── Save ──────────────────────────────────────────────────────────────────────
def save_to_mongo(data, source="unknown"):
    """Insert a single document into MongoDB with metadata."""
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
