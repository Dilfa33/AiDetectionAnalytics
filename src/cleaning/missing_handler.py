"""
src/cleaning/missing_handler.py
Detect and handle missing values in the ArXiv papers dataset.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logging


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of missing-value counts and ratios per column."""
    total = len(df)
    counts = df.isnull().sum()
    ratios = counts / total
    report = pd.DataFrame({
        "missing_count": counts,
        "missing_ratio": ratios.round(4),
        "dtype": df.dtypes,
    })
    logging.info("[MissingHandler] Missing-value report generated (%d columns)", len(report))
    return report


def drop_missing_ids(df: pd.DataFrame, id_col: str = "paper_id") -> pd.DataFrame:
    """Drop rows where the primary identifier is null or empty."""
    before = len(df)
    mask = df[id_col].isnull() | (df[id_col].astype(str).str.strip() == "")
    df = df[~mask].copy()
    dropped = before - len(df)
    logging.info("[MissingHandler] Dropped %d row(s) with missing %s", dropped, id_col)
    return df


def fill_missing_authors(df: pd.DataFrame, col: str = "authors") -> pd.DataFrame:
    """Fill missing author strings with 'Unknown'."""
    filled = df[col].isnull().sum()
    df = df.copy()
    df[col] = df[col].fillna("Unknown")
    logging.info("[MissingHandler] Filled %d missing author(s) with 'Unknown'", filled)
    return df


def fill_missing_abstracts(df: pd.DataFrame, col: str = "abstract") -> pd.DataFrame:
    """
    Replace empty, whitespace-only, or 'N/A' abstracts with NaN,
    then fill NaN with a placeholder.
    """
    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"": np.nan, "N/A": np.nan, "nan": np.nan})
    filled = df[col].isnull().sum()
    df[col] = df[col].fillna("Abstract not available")
    logging.info("[MissingHandler] Filled %d missing abstract(s)", filled)
    return df


def fill_missing_numeric(
    df: pd.DataFrame,
    citation_col: str = "citation_count",
    relevance_col: str = "relevance_score",
) -> pd.DataFrame:
    """
    - Convert citation_count to numeric (coerce), fill NaN with median.
    - Replace sentinel -1 in relevance_score with NaN, fill with median.
    """
    df = df.copy()

    df[citation_col] = pd.to_numeric(df[citation_col], errors="coerce")
    cite_median = df[citation_col].median()
    filled_cite = df[citation_col].isnull().sum()
    df[citation_col] = df[citation_col].fillna(cite_median)
    logging.info(
        "[MissingHandler] Filled %d missing citation_count(s) with median %.1f",
        filled_cite, cite_median,
    )

    df[relevance_col] = pd.to_numeric(df[relevance_col], errors="coerce")
    df[relevance_col] = df[relevance_col].replace(-1, np.nan)
    # Values outside [0, 1] are invalid — treat as missing
    df[relevance_col] = df[relevance_col].where(
        df[relevance_col].between(0.0, 1.0, inclusive="both"), other=np.nan
    )
    rel_median = df[relevance_col].median()
    filled_rel = df[relevance_col].isnull().sum()
    df[relevance_col] = df[relevance_col].fillna(rel_median)
    logging.info(
        "[MissingHandler] Filled %d missing/sentinel relevance_score(s) with median %.4f",
        filled_rel, rel_median,
    )

    return df


def drop_high_missing_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where more than `threshold` fraction of values are missing."""
    before_cols = list(df.columns)
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    logging.info(
        "[MissingHandler] Dropped %d high-missing column(s): %s", len(drop_cols), drop_cols
    )
    return df
