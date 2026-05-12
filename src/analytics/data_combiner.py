"""
src/analytics/data_combiner.py
Lab 10: Merge MySQL financial data with CSV metadata using all four join types.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import logging

ANALYTICS_DIR = Path("data/processed/analytics")


def merge_on_paper_id(left: pd.DataFrame, right: pd.DataFrame,
                      how: str = "inner") -> pd.DataFrame:
    """Merge two DataFrames on paper_id using the specified join type."""
    left  = left.copy()
    right = right.copy()
    left["paper_id"]  = left["paper_id"].astype(str)
    right["paper_id"] = right["paper_id"].astype(str)

    merged = pd.merge(left, right, on="paper_id", how=how, suffixes=("_csv", "_db"))
    logging.info("[DataCombiner] %s join → %d rows", how.upper(), len(merged))
    return merged


def compare_join_types(csv_df: pd.DataFrame, db_df: pd.DataFrame) -> dict[str, int]:
    """Run all four join types and return a dict of join_type → row_count."""
    counts = {}
    for how in ("inner", "left", "right", "outer"):
        merged = merge_on_paper_id(csv_df, db_df, how=how)
        counts[how] = len(merged)
    return counts


def save_join_comparison_chart(counts: dict[str, int],
                                out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save a bar chart showing row counts per join type."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["steelblue"] * 4)
    ax.set_title("Row Count by Join Type")
    ax.set_xlabel("Join Type")
    ax.set_ylabel("Rows")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=11)
    plt.tight_layout()
    path = out_dir / "join_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    logging.info("[DataCombiner] Join comparison chart saved to %s", path)
    return path


def concatenate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack DataFrames with the same columns (row concatenation)."""
    combined = pd.concat(frames, ignore_index=True)
    logging.info("[DataCombiner] Concatenated %d frames → %d rows", len(frames), len(combined))
    return combined


def combine_sources(csv_df: pd.DataFrame, db_df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary entry point: inner-merge CSV metadata with DB metrics,
    deduplicate columns, and return the combined DataFrame.
    """
    combined = merge_on_paper_id(csv_df, db_df, how="inner")

    # Prefer CSV columns when duplicated
    for col in list(combined.columns):
        if col.endswith("_csv"):
            base = col[:-4]
            db_col = base + "_db"
            combined[base] = combined[col]
            combined.drop(columns=[col, db_col], errors="ignore", inplace=True)

    logging.info("[DataCombiner] Combined DataFrame: %d rows × %d cols",
                 *combined.shape)
    return combined
