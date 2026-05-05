"""
src/cleaning/string_cleaner.py
Standardise and normalise text columns using vectorised pandas string operations.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logging

# Mapping of non-standard language codes to canonical 'en'
_LANG_MAP = {
    "en": "en",
    "english": "en",
    "eng": "en",
}


def clean_titles(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """Strip leading/trailing whitespace and apply title-case normalisation."""
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.title()
    logging.info("[StringCleaner] Cleaned title column")
    return df


def normalize_language(df: pd.DataFrame, col: str = "language") -> pd.DataFrame:
    """Lowercase, strip, and map language variants to a canonical code."""
    df = df.copy()
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: _LANG_MAP.get(x, x))
    )
    logging.info("[StringCleaner] Normalised language column")
    return df


def clean_abstract(df: pd.DataFrame, col: str = "abstract") -> pd.DataFrame:
    """Strip whitespace from abstracts; replace placeholder strings with NaN."""
    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace(
        {"N/A": np.nan, "Abstract not available": np.nan, "nan": np.nan, "": np.nan}
    )
    logging.info("[StringCleaner] Cleaned abstract column")
    return df


def extract_year(
    df: pd.DataFrame,
    date_col: str = "published_date",
    year_col: str = "published_year",
) -> pd.DataFrame:
    """
    Extract the four-digit year from published_date into a new column.
    Works on ISO dates (YYYY-MM-DD), slash dates (YYYY/MM/DD),
    compact (YYYYMMDD), and plain year strings.
    """
    import re
    df = df.copy()

    def _extract(val):
        s = str(val).strip()
        # compact YYYYMMDD — year is the leading 4 digits
        if re.fullmatch(r"\d{8}", s):
            return int(s[:4])
        m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
        return int(m.group(1)) if m else np.nan

    df[year_col] = df[date_col].apply(_extract)
    logging.info("[StringCleaner] Extracted year into '%s' column", year_col)
    return df


def clean_categories(df: pd.DataFrame, col: str = "primary_category") -> pd.DataFrame:
    """Strip whitespace from category codes."""
    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    logging.info("[StringCleaner] Cleaned primary_category column")
    return df
