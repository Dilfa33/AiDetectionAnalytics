"""
src/cleaning/deduplicator.py
Identify and remove duplicate records from the ArXiv papers dataset.
"""

import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logging


def count_duplicates(df: pd.DataFrame, col: str | None = None) -> int:
    """Return the number of duplicate rows (or duplicate values in a single column)."""
    if col:
        return int(df[col].duplicated().sum())
    return int(df.duplicated().sum())


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where every column is identical to another row."""
    before = len(df)
    df = df.drop_duplicates().copy()
    dropped = before - len(df)
    logging.info("[Deduplicator] Removed %d exact duplicate row(s)", dropped)
    return df


def drop_key_duplicates(
    df: pd.DataFrame,
    key: str = "paper_id",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove rows with a duplicated primary key, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=[key], keep=keep).copy()
    dropped = before - len(df)
    logging.info(
        "[Deduplicator] Removed %d key-duplicate row(s) on '%s'", dropped, key
    )
    return df
