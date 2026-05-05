"""
src/cleaning/clean_pipeline.py
Combine all cleaning steps into one reusable workflow.
"""

import pandas as pd
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logging

from .missing_handler import (
    drop_missing_ids,
    fill_missing_authors,
    fill_missing_abstracts,
    fill_missing_numeric,
    drop_high_missing_cols,
)
from .string_cleaner import (
    clean_titles,
    normalize_language,
    clean_abstract,
    extract_year,
    clean_categories,
)
from .deduplicator import drop_exact_duplicates, drop_key_duplicates
from .type_converter import convert_dates, convert_numerics, convert_categories
from .validator import run_all_validations

RAW_CSV     = Path("data/raw/csv/papers_raw.csv")
CLEANED_CSV = Path("data/processed/cleaned/cleaned_data.csv")


def run_cleaning_pipeline(
    df: pd.DataFrame | None = None,
    raw_path: Path = RAW_CSV,
    output_path: Path = CLEANED_CSV,
) -> pd.DataFrame:
    """
    Load → clean → validate → save.
    Pass a DataFrame directly or leave df=None to load from raw_path.
    Returns the cleaned DataFrame.
    """
    if df is None:
        logging.info("[CleanPipeline] Loading raw data from %s", raw_path)
        df = pd.read_csv(raw_path)
        logging.info("[CleanPipeline] Loaded %d rows, %d columns", *df.shape)

    df = df.copy()

    # ── 1. Drop columns with too many missing values ──────────────────────────
    logging.info("[CleanPipeline] Step 1: drop high-missing columns")
    df = drop_high_missing_cols(df, threshold=0.5)

    # ── 2. Handle missing values ──────────────────────────────────────────────
    logging.info("[CleanPipeline] Step 2: handle missing values")
    df = drop_missing_ids(df)
    df = fill_missing_authors(df)
    df = fill_missing_abstracts(df)
    df = fill_missing_numeric(df)

    # ── 3. String cleaning ────────────────────────────────────────────────────
    logging.info("[CleanPipeline] Step 3: string cleaning")
    df = clean_titles(df)
    df = normalize_language(df)
    df = clean_abstract(df)
    df = clean_categories(df)

    # ── 4. Extract year ───────────────────────────────────────────────────────
    logging.info("[CleanPipeline] Step 4: extract published year")
    df = extract_year(df)

    # ── 5. Remove duplicates ──────────────────────────────────────────────────
    logging.info("[CleanPipeline] Step 5: deduplication")
    df = drop_exact_duplicates(df)
    df = drop_key_duplicates(df)

    # ── 6. Type conversion ────────────────────────────────────────────────────
    logging.info("[CleanPipeline] Step 6: type conversion")
    df = convert_dates(df)
    df = convert_numerics(df)
    df = convert_categories(df)

    # ── 7. Validation ─────────────────────────────────────────────────────────
    logging.info("[CleanPipeline] Step 7: validation")
    results = run_all_validations(df)
    for check, outcome in results.items():
        logging.info("[CleanPipeline]   %s → %s", check, outcome)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("[CleanPipeline] Saved cleaned data (%d rows) to %s", len(df), output_path)

    return df


if __name__ == "__main__":
    cleaned = run_cleaning_pipeline()
    print(f"Cleaned dataset: {cleaned.shape[0]} rows × {cleaned.shape[1]} columns")
    print(cleaned.dtypes.to_string())
