"""
src/cleaning/validator.py
Validate cleaned data with assertions and logical checks.
"""

import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logging


def validate_ids(df: pd.DataFrame, col: str = "paper_id") -> bool:
    """Assert no null or empty paper IDs exist."""
    null_count = df[col].isnull().sum()
    empty_count = (df[col].astype(str).str.strip() == "").sum()
    assert null_count == 0, f"Found {null_count} null {col}(s)"
    assert empty_count == 0, f"Found {empty_count} empty {col}(s)"
    logging.info("[Validator] paper_id: OK (no nulls or empties)")
    return True


def validate_relevance(
    df: pd.DataFrame,
    col: str = "relevance_score",
    lo: float = 0.0,
    hi: float = 1.0,
) -> bool:
    """Assert all relevance scores are within [lo, hi]."""
    numeric = pd.to_numeric(df[col], errors="coerce").dropna()
    out_of_range = ((numeric < lo) | (numeric > hi)).sum()
    assert out_of_range == 0, (
        f"Found {out_of_range} relevance_score(s) outside [{lo}, {hi}]"
    )
    logging.info("[Validator] relevance_score: OK (all in [%.1f, %.1f])", lo, hi)
    return True


def validate_dates(df: pd.DataFrame, col: str = "published_date") -> bool:
    """Assert published_date column is datetime with no NaT values."""
    assert pd.api.types.is_datetime64_any_dtype(df[col]), (
        f"Column '{col}' is not datetime dtype (got {df[col].dtype})"
    )
    nat_count = df[col].isnull().sum()
    assert nat_count == 0, f"Found {nat_count} NaT value(s) in '{col}'"
    logging.info("[Validator] published_date: OK (datetime, no NaT)")
    return True


def validate_no_duplicates(df: pd.DataFrame, col: str = "paper_id") -> bool:
    """Assert no duplicate values exist in the key column."""
    dup_count = df[col].duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate {col}(s)"
    logging.info("[Validator] No duplicate %s values: OK", col)
    return True


def run_all_validations(df: pd.DataFrame) -> dict:
    """Run all validation checks and return a result dict (name -> passed/error message)."""
    checks = {
        "paper_id_not_null": validate_ids,
        "relevance_in_range": validate_relevance,
        "dates_are_datetime": validate_dates,
        "no_duplicate_ids":   validate_no_duplicates,
    }
    results = {}
    for name, fn in checks.items():
        try:
            fn(df)
            results[name] = "PASSED"
        except AssertionError as e:
            results[name] = f"FAILED: {e}"
            logging.warning("[Validator] %s — %s", name, e)
    return results
