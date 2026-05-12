"""
src/analytics/pivot_builder.py
Lab 10: Reshape data with melt/pivot and build pivot tables and crosstabs.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from pathlib import Path
from src.utils.logger import logging

ANALYTICS_DIR = Path("data/processed/analytics")


def extract_primary_category(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure primary_category column exists (already present in cleaned data)."""
    df = df.copy()
    if "primary_category" not in df.columns and "all_categories" in df.columns:
        df["primary_category"] = df["all_categories"].str.split("|").str[0].str.strip()
    return df


def wide_to_long(df: pd.DataFrame,
                 id_vars: list[str] | None = None,
                 value_vars: list[str] | None = None) -> pd.DataFrame:
    """Convert wide DataFrame to long format using melt()."""
    if id_vars is None:
        id_vars = ["paper_id", "title", "primary_category"]
    if value_vars is None:
        value_vars = ["citation_count", "relevance_score", "published_year"]

    id_vars    = [c for c in id_vars    if c in df.columns]
    value_vars = [c for c in value_vars if c in df.columns]

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="metric",
        value_name="value",
    )
    logging.info("[PivotBuilder] wide→long: %d rows × %d cols", *long_df.shape)
    return long_df


def long_to_wide(long_df: pd.DataFrame,
                 index: str = "paper_id",
                 columns: str = "metric",
                 values: str = "value") -> pd.DataFrame:
    """Convert long DataFrame back to wide format using pivot()."""
    wide_df = long_df.pivot(index=index, columns=columns, values=values)
    wide_df.columns.name = None
    wide_df = wide_df.reset_index()
    logging.info("[PivotBuilder] long→wide: %d rows × %d cols", *wide_df.shape)
    return wide_df


def build_pivot_table(df: pd.DataFrame,
                      index: str = "published_year",
                      columns: str = "primary_category",
                      values: str = "citation_count",
                      aggfunc: str = "mean") -> pd.DataFrame:
    """Build an aggregated pivot table with subtotals."""
    df = df.copy()
    df[index]   = df[index].astype(str)
    df[columns] = df[columns].astype(str)

    pivot = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        margins=True,
        margins_name="All",
    )
    logging.info("[PivotBuilder] Pivot table: %d rows × %d cols", *pivot.shape)
    return pivot


def build_crosstab(df: pd.DataFrame,
                   row_col: str = "language",
                   col_col: str = "published_year") -> pd.DataFrame:
    """Build a language × year cross-tabulation."""
    ct = pd.crosstab(df[row_col], df[col_col])
    logging.info("[PivotBuilder] Crosstab: %d rows × %d cols", *ct.shape)
    return ct


def save_pivot_csv(pivot: pd.DataFrame,
                   filename: str = "pivot_category_year.csv",
                   out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save a pivot table to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    pivot.to_csv(path)
    logging.info("[PivotBuilder] Saved pivot table to %s", path)
    return path
