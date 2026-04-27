"""
src/analytics/selector.py
Data selection and filtering using loc, iloc, boolean masks, isin, and between.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.utils.logger import logging


# ── Column selection ──────────────────────────────────────────────────────────

def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Select a subset of columns that actually exist in the DataFrame."""
    existing = [c for c in columns if c in df.columns]
    missing  = set(columns) - set(existing)
    if missing:
        logging.warning(f"[Selector] Columns not found (skipped): {missing}")
    result = df[existing]
    logging.info(f"[Selector] select_columns: {list(result.columns)}")
    return result


# ── loc – label-based ─────────────────────────────────────────────────────────

def filter_by_label(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Return rows where df[column] == value using loc."""
    if column not in df.columns:
        logging.warning(f"[Selector] Column '{column}' not in DataFrame")
        return df.iloc[0:0]
    result = df.loc[df[column] == value]
    logging.info(f"[Selector] filter_by_label({column}={value!r}): {len(result)} rows")
    return result


def loc_select(df: pd.DataFrame, row_labels, columns: list = None) -> pd.DataFrame:
    """loc access by row labels and optional column list."""
    if columns:
        existing_cols = [c for c in columns if c in df.columns]
        result = df.loc[row_labels, existing_cols]
    else:
        result = df.loc[row_labels]
    logging.info(f"[Selector] loc_select: {result.shape}")
    return result


# ── iloc – positional sampling ────────────────────────────────────────────────

def sample_rows(df: pd.DataFrame, start: int = 0, end: int = 5) -> pd.DataFrame:
    """Return rows [start:end] using iloc."""
    end    = min(end, len(df))
    result = df.iloc[start:end]
    logging.info(f"[Selector] sample_rows(iloc[{start}:{end}]): {len(result)} rows")
    return result


def iloc_select(df: pd.DataFrame, row_indices: list,
                col_indices: list = None) -> pd.DataFrame:
    """iloc access by row and optional column integer indices."""
    if col_indices:
        result = df.iloc[row_indices, col_indices]
    else:
        result = df.iloc[row_indices]
    logging.info(f"[Selector] iloc_select: {result.shape}")
    return result


# ── Boolean masks ─────────────────────────────────────────────────────────────

def filter_quality_and_popularity(df: pd.DataFrame,
                                   min_rating: float = 6.0,
                                   min_popularity: float = 5.0) -> pd.DataFrame:
    """
    Combined boolean mask: vote_average >= min_rating AND popularity >= min_popularity.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    if "vote_average" in df.columns:
        mask &= df["vote_average"].fillna(0) >= min_rating
    if "popularity" in df.columns:
        mask &= df["popularity"].fillna(0) >= min_popularity
    result = df[mask]
    logging.info(f"[Selector] filter_quality_and_popularity"
                 f"(rating>={min_rating}, pop>={min_popularity}): {len(result)} rows")
    return result


# ── isin filtering ────────────────────────────────────────────────────────────

def filter_by_language(df: pd.DataFrame, languages: list,
                        exclude: bool = False) -> pd.DataFrame:
    """
    Keep rows where original_language is in `languages`.
    Set exclude=True to remove those languages instead.
    """
    if "original_language" not in df.columns:
        logging.warning("[Selector] 'original_language' not in DataFrame")
        return df
    mask   = df["original_language"].isin(languages)
    result = df[~mask] if exclude else df[mask]
    action = "excluded" if exclude else "included"
    logging.info(f"[Selector] filter_by_language ({action} {languages}): {len(result)} rows")
    return result


def filter_by_isin(df: pd.DataFrame, column: str,
                    values: list, exclude: bool = False) -> pd.DataFrame:
    """Generic isin filter on any column."""
    if column not in df.columns:
        logging.warning(f"[Selector] Column '{column}' not in DataFrame")
        return df
    mask   = df[column].isin(values)
    result = df[~mask] if exclude else df[mask]
    logging.info(f"[Selector] filter_by_isin(column={column}, exclude={exclude}): {len(result)} rows")
    return result


# ── between (range) filtering ─────────────────────────────────────────────────

def filter_by_range(df: pd.DataFrame, column: str,
                     lo: float, hi: float) -> pd.DataFrame:
    """Return rows where df[column] is between lo and hi (inclusive)."""
    if column not in df.columns:
        logging.warning(f"[Selector] Column '{column}' not in DataFrame")
        return df
    result = df[df[column].between(lo, hi)]
    logging.info(f"[Selector] filter_by_range({column} in [{lo}, {hi}]): {len(result)} rows")
    return result


def run_selector_demo(df: pd.DataFrame) -> dict:
    """Demonstrate all selector functions and log results."""
    logging.info("[Selector] Running selector demo")
    return {
        "columns":        select_columns(df, ["title", "vote_average", "popularity", "original_language"]),
        "first_5":        sample_rows(df, 0, 5),
        "high_quality":   filter_quality_and_popularity(df, 6.0, 5.0),
        "english_only":   filter_by_language(df, ["en"]),
        "non_english":    filter_by_language(df, ["en"], exclude=True),
        "mid_rating":     filter_by_range(df, "vote_average", 6.0, 8.0),
    }


if __name__ == "__main__":
    from src.analytics.data_loader import load_from_mongo
    df = load_from_mongo()
    if not df.empty:
        results = run_selector_demo(df)
        for key, val in results.items():
            print(f"\n--- {key} ---")
            print(val[["title", "vote_average", "popularity"]].to_string()
                  if hasattr(val, "columns") and "title" in val.columns else val)
