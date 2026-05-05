"""
run_pipeline.py
Lab 9: Generate raw CSV from ArXiv JSON data, then run the cleaning pipeline.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import random
from src.utils.logger import logging
from src.pipeline.generate_raw_csv import generate_raw_csv
from src.cleaning.clean_pipeline import run_cleaning_pipeline


if __name__ == "__main__":
    logging.info("Pipeline started")

    random.seed(42)
    csv_rows = generate_raw_csv()
    logging.info(f"CSV stage complete: {csv_rows} rows written to data/raw/csv/papers_raw.csv")

    cleaned = run_cleaning_pipeline()
    logging.info(f"Cleaning stage complete: {len(cleaned)} clean rows saved")

    logging.info("Pipeline finished")
