"""
run_pipeline.py
Extracts content from all PDF files in data/raw/pdf and stores to MongoDB.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pathlib import Path
from src.utils.logger import logging
from src.extraction.document_extractor import process_folder


if __name__ == "__main__":
    logging.info("Pipeline started")

    folder = Path("data/raw/pdf")
    logging.info(f"Processing PDF folder: {folder}")
    results = process_folder(folder, extensions=[".pdf"])

    logging.info("=" * 60)
    logging.info("Pipeline summary:")
    logging.info(f"  PDFs processed: {len(results)}")
    for r in results:
        logging.info(f"  - {r['file']} id={r['inserted_id']}")
    logging.info("Pipeline finished")
