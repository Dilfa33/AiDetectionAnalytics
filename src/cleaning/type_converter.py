"""
src/cleaning/type_converter.py
Convert dataset columns to appropriate data types.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logging


def convert_dates(
    df: pd.DataFrame,
    col: str = "published_date",
) -> pd.DataFrame:
    """
    Parse published_date to datetime.  Handles:
      YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD, and plain YYYY (treated as Jan 1).
    """
    import re

    df = df.copy()

    def _normalise(val):
        s = str(val).strip()
        # compact YYYYMMDD → YYYY-MM-DD
        if re.fullmatch(r"\d{8}", s):
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        # slash → dash
        s = s.replace("/", "-")
        # bare year → Jan 1
        if re.fullmatch(r"\d{4}", s):
            return f"{s}-01-01"
        return s

    df[col] = df[col].apply(_normalise)
    df[col] = pd.to_datetime(df[col], errors="coerce")
    invalid = df[col].isnull().sum()
    if invalid:
        logging.warning("[TypeConverter] %d unparseable date(s) set to NaT", invalid)
    logging.info("[TypeConverter] Converted '%s' to datetime", col)
    return df


def convert_numerics(
    df: pd.DataFrame,
    citation_col: str = "citation_count",
    relevance_col: str = "relevance_score",
) -> pd.DataFrame:
    """Convert citation_count to Int64 (nullable int) and relevance_score to float32."""
    df = df.copy()

    df[citation_col] = pd.to_numeric(df[citation_col], errors="coerce").astype("Int64")
    logging.info("[TypeConverter] Converted '%s' to Int64", citation_col)

    df[relevance_col] = pd.to_numeric(df[relevance_col], errors="coerce").astype("float32")
    logging.info("[TypeConverter] Converted '%s' to float32", relevance_col)

    return df


def convert_categories(
    df: pd.DataFrame,
    col: str = "primary_category",
) -> pd.DataFrame:
    """Convert low-cardinality text column to category dtype to save memory."""
    df = df.copy()
    df[col] = df[col].astype("category")
    logging.info("[TypeConverter] Converted '%s' to category dtype", col)
    return df


def memory_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """Compare per-column memory usage before and after type conversion."""
    before_mem = df_before.memory_usage(deep=True)
    after_mem  = df_after.memory_usage(deep=True)
    report = pd.DataFrame({
        "before_bytes": before_mem,
        "after_bytes":  after_mem,
    }).drop(index="Index", errors="ignore")
    report["saved_bytes"] = report["before_bytes"] - report["after_bytes"]
    total_saved = report["saved_bytes"].sum()
    logging.info("[TypeConverter] Memory report: saved %d bytes total", total_saved)
    return report
