"""
src/analytics/data_loader.py
Load movie data from MongoDB and CSV, chunked processing,
dtype optimisation, and memory reporting.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from pathlib import Path
from pymongo import MongoClient
from src.utils.logger import logging

ANALYTICS_DIR = Path("data/processed/analytics")
RAW_CSV       = ANALYTICS_DIR / "movies_raw.csv"
MONGO_URI     = "mongodb://localhost:27017/"
DB_NAME       = "arxiv_pipeline"


# ── MongoDB → DataFrame ───────────────────────────────────────────────────────

def load_from_mongo(collection: str = "image_metadata") -> pd.DataFrame:
    """Load an entire MongoDB collection into a pandas DataFrame."""
    try:
        client  = MongoClient(MONGO_URI)
        db      = client[DB_NAME]
        records = list(db[collection].find({}, {"_id": 0}))
        client.close()

        if not records:
            logging.warning(f"[DataLoader] No records found in '{collection}'")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logging.info(f"[DataLoader] Loaded {len(df)} rows from '{collection}' "
                     f"({len(df.columns)} columns)")
        return df
    except Exception as e:
        logging.error(f"[DataLoader] MongoDB load failed: {e}")
        return pd.DataFrame()


# ── CSV save / load ───────────────────────────────────────────────────────────

def save_to_csv(df: pd.DataFrame, path: Path = RAW_CSV) -> Path:
    """Save a DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logging.info(f"[DataLoader] Saved {len(df)} rows to {path}")
    return path


def load_from_csv(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        logging.warning(f"[DataLoader] CSV not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8")
    logging.info(f"[DataLoader] Loaded {len(df)} rows from {path}")
    return df


# ── Chunked processing ────────────────────────────────────────────────────────

def load_in_chunks(path: Path = RAW_CSV, chunksize: int = 3):
    """Yield DataFrame chunks from a CSV file."""
    path = Path(path)
    if not path.exists():
        logging.warning(f"[DataLoader] CSV not found for chunked load: {path}")
        return
    logging.info(f"[DataLoader] Starting chunked load: chunksize={chunksize}")
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, encoding="utf-8")):
        logging.info(f"[DataLoader] Chunk {i + 1}: {len(chunk)} rows")
        yield chunk


def compute_mean_across_chunks(path: Path = RAW_CSV,
                                column: str = "vote_average",
                                chunksize: int = 3) -> float:
    """
    Compute the global mean of `column` across all CSV chunks
    without loading the full file into memory at once.
    """
    total_sum   = 0.0
    total_count = 0

    for chunk in load_in_chunks(path, chunksize):
        if column not in chunk.columns:
            continue
        valid = chunk[column].dropna()
        total_sum   += valid.sum()
        total_count += len(valid)

    global_mean = total_sum / total_count if total_count else float("nan")
    logging.info(f"[DataLoader] Global mean of '{column}' across chunks: {global_mean:.4f} "
                 f"(n={total_count})")
    return global_mean


def process_chunks_per_language(path: Path = RAW_CSV,
                                 chunksize: int = 3) -> pd.DataFrame:
    """
    Accumulate per-language stats (mean vote_average, count, mean popularity)
    across chunks and combine into a single summary DataFrame.
    """
    accumulators: dict = {}   # lang -> {sum_vote, sum_pop, count}

    for chunk in load_in_chunks(path, chunksize):
        if "original_language" not in chunk.columns:
            continue
        for lang, grp in chunk.groupby("original_language"):
            if lang not in accumulators:
                accumulators[lang] = {"sum_vote": 0.0, "sum_pop": 0.0, "count": 0}
            acc = accumulators[lang]
            acc["sum_vote"] += grp["vote_average"].fillna(0).sum()
            acc["sum_pop"]  += grp["popularity"].fillna(0).sum() if "popularity" in grp else 0
            acc["count"]    += len(grp)

    rows = []
    for lang, acc in accumulators.items():
        n = acc["count"] or 1
        rows.append({
            "language":        lang,
            "mean_vote_avg":   round(acc["sum_vote"] / n, 3),
            "mean_popularity": round(acc["sum_pop"]  / n, 3),
            "count":           acc["count"],
        })

    summary = pd.DataFrame(rows).sort_values("count", ascending=False)
    logging.info(f"[DataLoader] Per-language summary: {len(summary)} languages")
    return summary


# ── dtype optimisation ────────────────────────────────────────────────────────

def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns and convert low-cardinality string
    columns to category dtype to reduce memory usage.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Convert low-cardinality string columns to category
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            n_unique = df[col].nunique()
        except TypeError:
            # Column contains unhashable types (list, dict) — skip
            logging.info(f"[DataLoader] Skipping '{col}' (contains unhashable values)")
            continue
        if n_unique / max(len(df), 1) < 0.5:   # less than 50% unique → category
            df[col] = df[col].astype("category")
            logging.info(f"[DataLoader] Converted '{col}' to category ({n_unique} unique)")

    return df


def memory_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Log and return memory usage before and after optimisation."""
    mb_before = df_before.memory_usage(deep=True).sum() / 1024 ** 2
    mb_after  = df_after.memory_usage(deep=True).sum()  / 1024 ** 2
    reduction = (1 - mb_after / mb_before) * 100 if mb_before else 0

    logging.info(f"[DataLoader] Memory before optimisation: {mb_before:.4f} MB")
    logging.info(f"[DataLoader] Memory after  optimisation: {mb_after:.4f} MB")
    logging.info(f"[DataLoader] Memory reduction: {reduction:.1f}%")

    return {
        "before_mb":   round(mb_before, 4),
        "after_mb":    round(mb_after,  4),
        "reduction_pct": round(reduction, 1),
    }


if __name__ == "__main__":
    df = load_from_mongo()
    if not df.empty:
        csv_path = save_to_csv(df)
        mean_rating = compute_mean_across_chunks(csv_path, "vote_average")
        lang_summary = process_chunks_per_language(csv_path)
        print(lang_summary)
        df_opt = optimise_dtypes(df)
        memory_report(df, df_opt)
