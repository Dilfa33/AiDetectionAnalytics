"""
src/analytics/insight_reporter.py
Lab 10: Answer four analytical questions and save charts.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import logging

ANALYTICS_DIR = Path("data/processed/analytics")


# ── Q1: Top categories by average citations ───────────────────────────────────

def top_categories_by_citations(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Q1: Which categories attract the most citations on average?"""
    result = (
        df.groupby("primary_category")
          .agg(avg_citations=("citation_count", "mean"),
               paper_count  =("citation_count", "count"))
          .reset_index()
          .sort_values("avg_citations", ascending=False)
          .head(n)
    )
    top = result.iloc[0]
    logging.info("[Insights] Q1: Top category='%s' avg_citations=%.1f",
                 top["primary_category"], top["avg_citations"])
    return result


# ── Q2: ROI (citations per relevance_score unit) by category ─────────────────

def roi_by_category(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Q2: Which categories yield the most citations per relevance-score unit (ROI)?"""
    valid = df[df["relevance_score"] > 0].copy()
    valid["roi"] = valid["citation_count"] / valid["relevance_score"]

    result = (
        valid.groupby("primary_category")
             .agg(avg_roi   =("roi",           "mean"),
                  avg_rel   =("relevance_score","mean"),
                  avg_cit   =("citation_count", "mean"))
             .reset_index()
             .sort_values("avg_roi", ascending=False)
             .head(n)
    )
    top = result.iloc[0]
    logging.info("[Insights] Q2: Highest ROI category='%s' roi=%.2f",
                 top["primary_category"], top["avg_roi"])
    return result


# ── Q3: Yearly paper release volume ──────────────────────────────────────────

def yearly_paper_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Q3: How has the number of AI-detection papers grown year-over-year?"""
    result = (
        df.groupby("published_year")
          .agg(papers=("paper_id", "count"))
          .reset_index()
          .sort_values("published_year")
    )
    logging.info("[Insights] Q3: Year range %d–%d, %d data points",
                 result["published_year"].min(),
                 result["published_year"].max(),
                 len(result))
    return result


# ── Q4: Language distribution ─────────────────────────────────────────────────

def language_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Q4: What fraction of papers are published in each language?"""
    result = (
        df.groupby("language")
          .agg(count=("paper_id", "count"))
          .reset_index()
          .sort_values("count", ascending=False)
    )
    result["pct"] = (result["count"] / result["count"].sum() * 100).round(2)
    logging.info("[Insights] Q4: %d languages; dominant='%s' (%.1f%%)",
                 len(result),
                 result.iloc[0]["language"],
                 result.iloc[0]["pct"])
    return result


# ── Charts ────────────────────────────────────────────────────────────────────

def save_roi_chart(roi_df: pd.DataFrame,
                   out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save horizontal bar chart for ROI by category."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(roi_df["primary_category"], roi_df["avg_roi"], color="teal")
    for bar, val in zip(bars, roi_df["avg_roi"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)
    ax.set_title("Top Categories by Citation ROI (citations / relevance)")
    ax.set_xlabel("Avg ROI")
    ax.invert_yaxis()
    plt.tight_layout()
    path = out_dir / "category_roi.png"
    fig.savefig(path)
    plt.close(fig)
    logging.info("[Insights] Saved category_roi.png")
    return path


def save_volume_chart(vol_df: pd.DataFrame,
                      out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save yearly paper volume bar chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(vol_df["published_year"].astype(str), vol_df["papers"], color="steelblue")
    ax.set_title("Yearly AI-Detection Paper Volume")
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    plt.tight_layout()
    path = out_dir / "yearly_volume.png"
    fig.savefig(path)
    plt.close(fig)
    logging.info("[Insights] Saved yearly_volume.png")
    return path


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_questions(df: pd.DataFrame) -> dict:
    """Run all four analytical questions and print a formatted summary."""
    df = df.copy()
    if "primary_category" not in df.columns:
        df["primary_category"] = df.get("all_categories", pd.Series(dtype=str)) \
                                   .str.split("|").str[0].str.strip()

    q1 = top_categories_by_citations(df)
    q2 = roi_by_category(df)
    q3 = yearly_paper_volume(df)
    q4 = language_distribution(df)

    roi_chart = save_roi_chart(q2)
    vol_chart = save_volume_chart(q3)

    print("\n" + "=" * 60)
    print("ANALYTICAL FINDINGS — AI Detection Papers")
    print("=" * 60)

    print("\nQ1 — Top categories by avg citations:")
    print(q1[["primary_category", "avg_citations", "paper_count"]].to_string(index=False))

    print("\nQ2 — Categories with highest citation ROI:")
    print(q2[["primary_category", "avg_roi"]].to_string(index=False))

    print("\nQ3 — Papers published per year:")
    print(q3.to_string(index=False))

    print("\nQ4 — Language distribution:")
    print(q4.to_string(index=False))
    print("=" * 60)

    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4,
            "roi_chart": roi_chart, "vol_chart": vol_chart}
