"""
run_pipeline.py
Lab 8 – NumPy, pandas, and Data Exploration pipeline.

Stages
------
1. Re-seed MongoDB  – re-download posters so all movie fields are stored
2. NumPy analysis   – array creation + vectorized stats
3. Data loading     – MongoDB → CSV, chunked mean, per-language stats, dtype optimisation
4. EDA              – shape/info/describe/value_counts + distribution charts
5. Filtering        – loc / iloc / boolean / isin / between demos
6. Regex            – title, overview, genre pattern analysis
7. Quality report   – missing values, outliers, heatmap, CSV export
8. Drive upload     – upload charts to Google Drive (optional)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pathlib import Path
from src.utils.logger import logging

# ── Stage toggles ─────────────────────────────────────────────────────────────
RUN_RESEED  = True    # re-download + reprocess images to enrich MongoDB
RUN_DRIVE   = True    # set True after Google Drive credentials are configured


# ── Stage 1: re-seed MongoDB with enriched movie data ─────────────────────────
def run_reseed_stage():
    logging.info("=" * 60)
    logging.info("STAGE 1: Re-seeding MongoDB with enriched movie data")
    logging.info("=" * 60)
    from src.image_processing.downloader import download_posters
    from src.image_processing.batch import batch_process_images

    movies  = download_posters(count=10)
    lookup  = {m["filename"]: m for m in movies if m.get("filename")}
    results = batch_process_images(
        input_dir=Path("data/raw/images"),
        movie_lookup=lookup,
        upload_to_drive=False,
    )
    logging.info(f"Re-seed complete: {len(results)} records upserted into MongoDB")
    return results


# ── Stage 2: NumPy analysis ───────────────────────────────────────────────────
def run_numpy_stage(df):
    logging.info("=" * 60)
    logging.info("STAGE 2: NumPy array operations")
    logging.info("=" * 60)
    from src.analytics.numpy_ops import run_numpy_analysis
    stats = run_numpy_analysis(df)
    logging.info(f"NumPy stage complete: {stats}")
    return stats


# ── Stage 3: Data loading & memory optimisation ───────────────────────────────
def run_loader_stage():
    logging.info("=" * 60)
    logging.info("STAGE 3: Data loading, chunking, dtype optimisation")
    logging.info("=" * 60)
    from src.analytics.data_loader import (
        load_from_mongo, save_to_csv, load_from_csv,
        compute_mean_across_chunks, process_chunks_per_language,
        optimise_dtypes, memory_report,
    )

    df       = load_from_mongo("image_metadata")
    csv_path = save_to_csv(df)

    mean_rating  = compute_mean_across_chunks(csv_path, "vote_average")
    lang_summary = process_chunks_per_language(csv_path)
    logging.info(f"Per-language summary:\n{lang_summary.to_string()}")

    df_opt = optimise_dtypes(df)
    mem    = memory_report(df, df_opt)
    logging.info(f"Memory: {mem['before_mb']} MB -> {mem['after_mb']} MB "
                 f"({mem['reduction_pct']}% reduction)")
    return df, df_opt


# ── Stage 4: EDA ──────────────────────────────────────────────────────────────
def run_eda_stage(df):
    logging.info("=" * 60)
    logging.info("STAGE 4: Exploratory Data Analysis")
    logging.info("=" * 60)
    from src.analytics.explorer import run_eda
    results = run_eda(df)
    logging.info(f"EDA complete: {len(results['charts'])} charts saved")
    return results["charts"]


# ── Stage 5: Filtering ────────────────────────────────────────────────────────
def run_selector_stage(df):
    logging.info("=" * 60)
    logging.info("STAGE 5: loc / iloc / boolean / isin / between")
    logging.info("=" * 60)
    from src.analytics.selector import run_selector_demo
    results = run_selector_demo(df)
    for name, subset in results.items():
        if hasattr(subset, "__len__"):
            logging.info(f"  {name}: {len(subset)} rows")
    return results


# ── Stage 6: Regex ────────────────────────────────────────────────────────────
def run_regex_stage(df):
    logging.info("=" * 60)
    logging.info("STAGE 6: Regular expression operations")
    logging.info("=" * 60)
    from src.analytics.regex_ops import run_regex_demo
    results = run_regex_demo(df)
    logging.info(f"Top genres: {results.get('top_genres')}")
    return results


# ── Stage 7: Quality report ───────────────────────────────────────────────────
def run_quality_stage(df):
    logging.info("=" * 60)
    logging.info("STAGE 7: Data quality assessment")
    logging.info("=" * 60)
    from src.analytics.quality_report import full_quality_audit, save_quality_report, CHARTS_DIR
    audit        = full_quality_audit(df)
    report_path  = save_quality_report(audit)
    logging.info(f"Quality report saved: {report_path}")
    heatmap_path = CHARTS_DIR / "missing_data_heatmap.png"
    return report_path, heatmap_path


# ── Stage 8: Google Drive upload ──────────────────────────────────────────────
def run_drive_stage(chart_paths):
    logging.info("=" * 60)
    logging.info("STAGE 8: Uploading charts to Google Drive")
    logging.info("=" * 60)
    try:
        from src.utils.upload_utils import get_drive_service, upload_batch
        service = get_drive_service()
        if service:
            results  = upload_batch(service, chart_paths)
            uploaded = sum(1 for r in results if r["drive_id"])
            logging.info(f"Drive upload: {uploaded}/{len(chart_paths)} charts uploaded")
            return results
    except Exception as e:
        logging.error(f"Drive upload failed: {e}")
    return []


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("Pipeline started  (Lab 8 – Analytics)")

    if RUN_RESEED:
        run_reseed_stage()

    df, df_opt = run_loader_stage()

    if df.empty:
        logging.error("No data loaded from MongoDB — aborting analytics stages")
    else:
        run_numpy_stage(df)
        chart_paths          = run_eda_stage(df)
        run_selector_stage(df)
        run_regex_stage(df)
        report_path, heatmap = run_quality_stage(df)

        if RUN_DRIVE:
            all_charts = list(chart_paths) + ([heatmap] if heatmap.exists() else [])
            if all_charts:
                run_drive_stage(all_charts)

    logging.info("=" * 60)
    logging.info("Pipeline finished")
