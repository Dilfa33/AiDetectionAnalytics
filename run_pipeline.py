"""
run_pipeline.py
Lab 9 + Lab 10 + Lab 12: Generate raw CSV → clean → analytics → visualizations.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import random
from pathlib import Path
from src.utils.logger import logging
from src.pipeline.generate_raw_csv import generate_raw_csv
from src.cleaning.clean_pipeline import run_cleaning_pipeline

CLEANED_CSV   = Path("data/processed/cleaned/cleaned_data.csv")
ANALYTICS_DIR = Path("data/processed/analytics")


def run_analytics_pipeline(df):
    """Lab 10: run all analytics modules on the cleaned DataFrame."""
    import pandas as pd
    from src.analytics.aggregator    import (category_summary, yearly_trends,
                                             top_n_per_group,
                                             save_category_csv, save_yearly_csv,
                                             save_yearly_chart)
    from src.analytics.pivot_builder import (extract_primary_category,
                                             wide_to_long, build_pivot_table,
                                             build_crosstab, save_pivot_csv)
    from src.analytics.time_series   import run_time_series_pipeline
    from src.analytics.mongo_pipeline import run_mongo_pipeline
    from src.analytics.insight_reporter import run_all_questions

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure primary_category column exists before anything else
    df = extract_primary_category(df)

    # ── MySQL (optional — skipped if MySQL not installed) ─────────────────────
    try:
        from src.analytics.db_connector import run_db_pipeline
        db_df = run_db_pipeline(df)
        logging.info("[Pipeline] MySQL round-trip OK: %d rows", len(db_df))
    except Exception as e:
        logging.warning("[Pipeline] MySQL step skipped: %s", e)

    # ── Aggregation ───────────────────────────────────────────────────────────
    logging.info("[Pipeline] Running category summary")
    summary = category_summary(df)
    save_category_csv(summary)

    logging.info("[Pipeline] Running yearly trends")
    trends = yearly_trends(df)
    save_yearly_csv(trends)
    save_yearly_chart(trends)

    logging.info("[Pipeline] Top 3 papers per category by citations")
    top3 = top_n_per_group(df, n=3)
    logging.info("[Pipeline] Top-N sample:\n%s", top3[["primary_category", "title", "citation_count"]].head(9).to_string(index=False))

    # ── Pivot & reshape ───────────────────────────────────────────────────────
    logging.info("[Pipeline] Building pivot table")
    df = extract_primary_category(df)
    pivot = build_pivot_table(df)
    save_pivot_csv(pivot, filename="pivot_category_year.csv")

    logging.info("[Pipeline] Building crosstab")
    ct = build_crosstab(df)
    logging.info("[Pipeline] Crosstab shape: %s", ct.shape)

    # ── Time series ───────────────────────────────────────────────────────────
    logging.info("[Pipeline] Running time series analysis")
    ts_results = run_time_series_pipeline(df)
    logging.info("[Pipeline] Monthly series length: %d", len(ts_results["monthly"]))

    # ── MongoDB aggregation ───────────────────────────────────────────────────
    logging.info("[Pipeline] Running MongoDB aggregation pipeline")
    mongo_df = run_mongo_pipeline(df_fallback=df)
    logging.info("[Pipeline] Mongo pipeline result:\n%s", mongo_df.to_string(index=False))

    # ── Insight questions ─────────────────────────────────────────────────────
    logging.info("[Pipeline] Running analytical questions")
    findings = run_all_questions(df)

    logging.info("[Pipeline] Analytics complete — outputs in %s", ANALYTICS_DIR)
    return findings


def run_visualizations_pipeline(df):
    """Lab 12: generate all static and interactive charts."""
    from src.visualization.chart_generator import generate_all_charts
    logging.info("[Pipeline] Visualization step started")
    generate_all_charts(df=df)
    logging.info("[Pipeline] Visualization step complete")


if __name__ == "__main__":
    import shutil
    from src.api.client import fetch_papers

    logging.info("Pipeline started")

    random.seed(42)

    # ── Step 1: fetch 50 real papers from ArXiv (5 pages × 10 per page) ──────
    logging.info("Fetching 50 papers from ArXiv API ...")
    api_dir = Path("data/raw/api")
    # Clear old JSON pages so stale data doesn't mix in
    for old in api_dir.glob("papers_page_*.json"):
        old.unlink()
    fetched = fetch_papers(pages=5)
    logging.info("Fetch complete: %d papers retrieved", len(fetched))

    # ── Step 2: write raw CSV (real data — no artificial dirt) ───────────────
    csv_rows = generate_raw_csv(inject_dirt=False)
    logging.info("CSV stage complete: %d rows → data/raw/csv/papers_raw.csv", csv_rows)

    # ── Step 3: clean ─────────────────────────────────────────────────────────
    cleaned = run_cleaning_pipeline()
    logging.info("Cleaning stage complete: %d clean rows", len(cleaned))

    # ── Step 4: analytics ─────────────────────────────────────────────────────
    run_analytics_pipeline(cleaned)

    # ── Step 5: visualizations ────────────────────────────────────────────────
    run_visualizations_pipeline(cleaned)

    # ── Step 6: embeddings & vector search ───────────────────────────────────
    logging.info("Embedding step started")
    try:
        from src.embeddings.chroma_store import get_client, get_or_create_collection, add_papers
        client     = get_client()
        collection = get_or_create_collection(client)
        added      = add_papers(collection, cleaned)
        logging.info("Embedding step complete — %d new papers indexed (%d total)",
                     added, collection.count())
    except Exception as e:
        logging.warning("Embedding step skipped: %s", e)

    logging.info("Pipeline finished")
