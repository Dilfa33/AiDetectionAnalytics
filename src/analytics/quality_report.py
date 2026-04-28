"""
src/analytics/quality_report.py
Systematic data quality assessment: missing values, zero-as-missing,
IQR outliers, duplicate IDs, rating validation, and a missing-data heatmap.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import logging

ANALYTICS_DIR  = Path("data/processed/analytics")
CHARTS_DIR     = ANALYTICS_DIR / "charts"
QUALITY_CSV    = ANALYTICS_DIR / "quality_report.csv"


# ── Missing value report ──────────────────────────────────────────────────────

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a missing value report with count, percentage, and severity.
    Severity: LOW < 5%, MEDIUM 5–20%, HIGH > 20%.
    """
    total  = len(df)
    report = []

    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct       = n_missing / total * 100 if total else 0
        severity  = "LOW" if pct < 5 else ("MEDIUM" if pct < 20 else "HIGH")
        report.append({
            "column":    col,
            "dtype":     str(df[col].dtype),
            "missing":   int(n_missing),
            "pct_missing": round(pct, 2),
            "severity":  severity,
        })

    result = pd.DataFrame(report).sort_values("pct_missing", ascending=False)
    high = (result["severity"] == "HIGH").sum()
    logging.info(f"[Quality] Missing value report: {high} HIGH-severity columns")
    return result


# ── Zero-as-missing detection ─────────────────────────────────────────────────

def detect_zero_as_missing(df: pd.DataFrame,
                             columns: list = None) -> pd.DataFrame:
    """
    Detect zeros that likely represent missing financial/metric data
    (e.g. vote_count=0 means no votes recorded, not actually zero).
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)

    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        n_zeros = (df[col] == 0).sum()
        pct     = n_zeros / len(df) * 100 if len(df) else 0
        rows.append({"column": col, "zero_count": int(n_zeros), "pct_zero": round(pct, 2)})
        logging.info(f"[Quality] Zero-as-missing: '{col}' has {n_zeros} zeros ({pct:.1f}%)")

    return pd.DataFrame(rows).sort_values("zero_count", ascending=False)


# ── IQR outlier detection ─────────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    IQR-based outlier detection for numeric columns.
    Outlier = value < Q1 - 1.5*IQR  or  value > Q3 + 1.5*IQR.
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)

    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        series  = df[col].dropna()
        q1, q3  = series.quantile(0.25), series.quantile(0.75)
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out   = ((series < lo) | (series > hi)).sum()
        rows.append({
            "column":        col,
            "q1":            round(float(q1), 3),
            "q3":            round(float(q3), 3),
            "iqr":           round(float(iqr), 3),
            "lower_fence":   round(float(lo),  3),
            "upper_fence":   round(float(hi),  3),
            "outlier_count": int(n_out),
            "pct_outliers":  round(n_out / len(series) * 100, 2) if len(series) else 0,
        })
        logging.info(f"[Quality] IQR outliers: '{col}' has {n_out} outliers")

    return pd.DataFrame(rows)


# ── Rating validation ─────────────────────────────────────────────────────────

def validate_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that vote_average is in [0, 10].
    Returns rows with out-of-range values.
    """
    rating_cols = [c for c in df.columns if "vote" in c.lower() or "rating" in c.lower()]
    rows = []
    for col in rating_cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid = numeric[(numeric < 0) | (numeric > 10)].dropna()
        rows.append({
            "column":        col,
            "invalid_count": len(invalid),
            "min":           float(numeric.min()) if not numeric.empty else None,
            "max":           float(numeric.max()) if not numeric.empty else None,
        })
        logging.info(f"[Quality] Rating validation: '{col}' has {len(invalid)} out-of-range values")
    return pd.DataFrame(rows)


# ── Duplicate detection ───────────────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame, column: str = "movie_id") -> dict:
    """Detect duplicate values in a key column."""
    if column not in df.columns:
        logging.warning(f"[Quality] Column '{column}' not found")
        return {}
    dupes = df[df.duplicated(subset=[column], keep=False)]
    logging.info(f"[Quality] Duplicates in '{column}': {len(dupes)} rows ({dupes[column].nunique()} IDs)")
    return {"column": column, "duplicate_rows": len(dupes), "duplicate_ids": dupes[column].nunique()}


# ── Missing data heatmap ──────────────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame) -> Path:
    """
    Visualise missing values as a heatmap and save to charts directory.
    Columns on x-axis, rows on y-axis; yellow = missing, purple = present.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / "missing_data_heatmap.png"

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.6), 5))
    missing_matrix = df.isnull().astype(int)

    ax.imshow(missing_matrix.T, aspect="auto", cmap="plasma", interpolation="none")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns, fontsize=8)
    ax.set_xlabel("Row index")
    ax.set_title("Missing Data Heatmap  (yellow = missing)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

    logging.info(f"[Quality] Missing data heatmap saved: {path}")
    return path


# ── Full quality audit ────────────────────────────────────────────────────────

def full_quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all quality checks and combine results into a single issues DataFrame.
    """
    logging.info("[Quality] Starting full quality audit")

    missing   = missing_value_report(df)
    zeros     = detect_zero_as_missing(df)
    outliers  = detect_outliers_iqr(df)
    dupes     = detect_duplicates(df, "movie_id")
    ratings   = validate_ratings(df)
    heatmap   = plot_missing_heatmap(df)

    # Combine into one audit summary (missing report is the primary output)
    audit = missing.copy()
    audit["zero_count"]     = audit["column"].map(
        zeros.set_index("column")["zero_count"] if not zeros.empty else {}
    )
    audit["outlier_count"]  = audit["column"].map(
        outliers.set_index("column")["outlier_count"] if not outliers.empty else {}
    )
    audit = audit.fillna(0)

    logging.info(f"[Quality] Audit complete: {len(audit)} columns assessed")
    logging.info(f"[Quality] Duplicates: {dupes}")
    logging.info(f"[Quality] Heatmap: {heatmap}")
    return audit


def save_quality_report(report: pd.DataFrame,
                         path: Path = QUALITY_CSV) -> Path:
    """Save the quality audit DataFrame as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    logging.info(f"[Quality] Quality report saved: {path} ({len(report)} rows)")
    return path


def run_quality_assessment(df: pd.DataFrame) -> Path:
    """Full pipeline: audit + save."""
    audit = full_quality_audit(df)
    return save_quality_report(audit)


if __name__ == "__main__":
    from src.analytics.data_loader import load_from_mongo
    df = load_from_mongo()
    if not df.empty:
        run_quality_assessment(df)
