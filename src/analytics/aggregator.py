"""
src/analytics/aggregator.py
Lab 10: GroupBy analysis — genre/category summaries, yearly trends, top-N per group.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import logging

ANALYTICS_DIR = Path("data/processed/analytics")


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by primary_category and compute four aggregations:
    mean, sum, count, and median for citation_count and relevance_score.
    """
    summary = df.groupby("primary_category").agg(
        mean_citations   =("citation_count",  "mean"),
        total_citations  =("citation_count",  "sum"),
        paper_count      =("citation_count",  "count"),
        median_citations =("citation_count",  "median"),
        mean_relevance   =("relevance_score", "mean"),
    ).reset_index()

    summary = summary.sort_values("total_citations", ascending=False)
    logging.info("[Aggregator] Category summary: %d categories", len(summary))
    return summary


def yearly_trends(df: pd.DataFrame,
                  start_year: int = 2020,
                  end_year: int = 2027) -> pd.DataFrame:
    """Compute paper count and total citations per year."""
    filtered = df[df["published_year"].between(start_year, end_year)].copy()
    trends = filtered.groupby("published_year").agg(
        paper_count     =("paper_id",       "count"),
        total_citations =("citation_count", "sum"),
        mean_relevance  =("relevance_score","mean"),
    ).reset_index()
    logging.info("[Aggregator] Yearly trends: %d years", len(trends))
    return trends


def top_n_per_group(df: pd.DataFrame,
                    group_col: str = "primary_category",
                    sort_col: str = "citation_count",
                    n: int = 3) -> pd.DataFrame:
    """Return the top N papers per group, sorted by sort_col descending."""
    sorted_df = df.sort_values(sort_col, ascending=False)
    top = (
        sorted_df.groupby(group_col)[sorted_df.columns.tolist()]
                 .apply(lambda g: g.head(n))
                 .reset_index(drop=True)
    )
    logging.info("[Aggregator] Top %d per '%s': %d rows", n, group_col, len(top))
    return top


def save_category_csv(summary: pd.DataFrame,
                      out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save category summary to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "category_analysis.csv"
    summary.to_csv(path, index=False)
    logging.info("[Aggregator] Saved category_analysis.csv")
    return path


def save_yearly_csv(trends: pd.DataFrame,
                    out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save yearly trends to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "yearly_trends.csv"
    trends.to_csv(path, index=False)
    logging.info("[Aggregator] Saved yearly_trends.csv")
    return path


def save_yearly_chart(trends: pd.DataFrame,
                      out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save a dual-axis chart: paper count and total citations over time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(trends["published_year"], trends["paper_count"],
            color="steelblue", alpha=0.7, label="Paper Count")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Paper Count", color="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(trends["published_year"], trends["total_citations"],
             color="darkorange", marker="o", label="Total Citations")
    ax2.set_ylabel("Total Citations", color="darkorange")

    ax1.set_title("Yearly Paper Count & Total Citations")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()

    path = out_dir / "yearly_trends.png"
    fig.savefig(path)
    plt.close(fig)
    logging.info("[Aggregator] Saved yearly_trends.png")
    return path
