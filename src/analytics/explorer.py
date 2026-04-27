"""
src/analytics/explorer.py
Exploratory Data Analysis: shape, info, describe, value_counts,
release year extraction, and distribution charts.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import logging

CHARTS_DIR = Path("data/processed/analytics/charts")


# ── Structure inspection ──────────────────────────────────────────────────────

def inspect_shape(df: pd.DataFrame) -> dict:
    """Return shape and column list."""
    info = {
        "rows":    df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
    }
    logging.info(f"[Explorer] Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    logging.info(f"[Explorer] Columns: {list(df.columns)}")
    return info


def display_info(df: pd.DataFrame) -> str:
    """Capture df.info() as a string and log it."""
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    logging.info(f"[Explorer] DataFrame info:\n{info_str}")
    return info_str


def describe_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return df.describe() for numeric columns."""
    stats = df.describe(include="all")
    logging.info(f"[Explorer] Descriptive statistics computed for {len(df.columns)} columns")
    return stats


# ── Categorical analysis ──────────────────────────────────────────────────────

def value_count_report(df: pd.DataFrame, columns: list = None) -> dict:
    """
    Return value_counts and nunique for each categorical column.
    If columns is None, auto-selects object + category columns.
    """
    if columns is None:
        columns = list(df.select_dtypes(include=["object", "category"]).columns)

    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        try:
            vc     = df[col].value_counts(dropna=False)
            n_uniq = df[col].nunique(dropna=False)
        except TypeError:
            logging.info(f"[Explorer] Skipping '{col}' (unhashable values)")
            continue
        report[col] = {"value_counts": vc, "nunique": n_uniq}
        logging.info(f"[Explorer] '{col}': {n_uniq} unique values — top={vc.index[0] if len(vc) else 'N/A'}")
    return report


# ── Release year extraction ───────────────────────────────────────────────────

def extract_release_year(df: pd.DataFrame,
                          date_col: str = "release_date") -> pd.DataFrame:
    """Parse YYYY from release_date and add a release_year column."""
    df = df.copy()
    if date_col not in df.columns:
        logging.warning(f"[Explorer] Column '{date_col}' not found — skipping year extraction")
        return df

    df["release_year"] = (
        pd.to_datetime(df[date_col], errors="coerce")
          .dt.year
          .astype("Int64")
    )
    valid = df["release_year"].notna().sum()
    logging.info(f"[Explorer] Extracted release_year: {valid}/{len(df)} valid dates")
    return df


# ── Charts ────────────────────────────────────────────────────────────────────

def _save_fig(name: str) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    logging.info(f"[Explorer] Chart saved: {path}")
    return path


def save_distribution_charts(df: pd.DataFrame) -> list[Path]:
    """
    Generate and save distribution charts for key movie columns.
    Returns list of saved file paths.
    """
    saved = []
    df = extract_release_year(df)

    # 1. Vote average distribution
    if "vote_average" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["vote_average"].dropna().hist(bins=20, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Vote Average Distribution")
        ax.set_xlabel("Vote Average")
        ax.set_ylabel("Count")
        saved.append(_save_fig("vote_average_dist"))

    # 2. Popularity distribution
    if "popularity" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["popularity"].dropna().hist(bins=20, ax=ax, color="tomato", edgecolor="white")
        ax.set_title("Popularity Distribution")
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Count")
        saved.append(_save_fig("popularity_dist"))

    # 3. Language bar chart
    if "original_language" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["original_language"].value_counts().plot(kind="bar", ax=ax, color="mediumseagreen")
        ax.set_title("Movies by Original Language")
        ax.set_xlabel("Language")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        saved.append(_save_fig("language_dist"))

    # 4. Release year distribution
    if "release_year" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["release_year"].dropna().astype(int).hist(bins=15, ax=ax, color="mediumpurple", edgecolor="white")
        ax.set_title("Release Year Distribution")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        saved.append(_save_fig("release_year_dist"))

    # 5. File size distribution
    if "file_size_kb" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["file_size_kb"].dropna().hist(bins=15, ax=ax, color="darkorange", edgecolor="white")
        ax.set_title("Poster File Size Distribution (KB)")
        ax.set_xlabel("File Size (KB)")
        ax.set_ylabel("Count")
        saved.append(_save_fig("file_size_dist"))

    logging.info(f"[Explorer] Saved {len(saved)} distribution charts to {CHARTS_DIR}")
    return saved


def run_eda(df: pd.DataFrame) -> dict:
    """Run full EDA and return a summary dict."""
    logging.info("[Explorer] Starting EDA")
    results = {
        "shape":   inspect_shape(df),
        "info":    display_info(df),
        "stats":   describe_stats(df),
        "cats":    value_count_report(df),
        "charts":  save_distribution_charts(df),
    }
    logging.info("[Explorer] EDA complete")
    return results


if __name__ == "__main__":
    from src.analytics.data_loader import load_from_mongo
    df = load_from_mongo()
    if not df.empty:
        run_eda(df)
