"""
src/analytics/numpy_ops.py
NumPy array creation, vectorized operations, and statistical analysis
on movie data (ratings, popularity, file sizes).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from src.utils.logger import logging


# ── Array creation demo (4+ methods) ─────────────────────────────────────────

def create_arrays_demo() -> dict:
    """
    Create NumPy arrays using 6 different methods.
    Logs shape / dtype / ndim for each.
    """
    # Method 1: np.array from a Python list
    sample_ratings = np.array([7.2, 6.8, 8.1, 5.5, 7.9, 6.3, 8.5, 7.1], dtype=np.float32)

    # Method 2: np.zeros
    zeros = np.zeros(8, dtype=np.float32)

    # Method 3: np.ones
    ones = np.ones(8, dtype=np.float32)

    # Method 4: np.linspace
    scale = np.linspace(0, 10, 8)

    # Method 5: np.arange
    indices = np.arange(8)

    # Method 6: np.random.uniform
    noise = np.random.uniform(-0.5, 0.5, 8)

    arrays = {
        "sample_ratings": sample_ratings,
        "zeros":          zeros,
        "ones":           ones,
        "linspace_scale": scale,
        "arange_indices": indices,
        "random_noise":   noise,
    }

    for name, arr in arrays.items():
        logging.info(
            f"[NumPy] {name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"ndim={arr.ndim}, size={arr.size}, itemsize={arr.itemsize}B"
        )

    return arrays


# ── Vectorized operations (zero Python loops) ─────────────────────────────────

def vectorized_ops(df: pd.DataFrame) -> dict:
    """
    Perform vectorized arithmetic on movie DataFrame columns.
    No Python loops used — all operations are NumPy broadcasts.
    """
    def _col(name, fill=0.0, dtype=np.float32):
        if name in df.columns:
            return df[name].fillna(fill).to_numpy(dtype=dtype)
        return np.zeros(len(df), dtype=dtype)

    vote_avg   = _col("vote_average")
    popularity = _col("popularity")
    file_kb    = _col("file_size_kb")
    vote_count = _col("vote_count", dtype=np.int32)

    # Vectorized arithmetic
    denom           = vote_avg.max() - vote_avg.min() + 1e-8
    normalized      = (vote_avg - vote_avg.min()) / denom
    log_popularity  = np.log1p(popularity)
    weighted_score  = vote_avg * np.log1p(vote_count)
    above_mean_mask = vote_avg > vote_avg.mean()

    # 2D matrix: each row is [vote_avg, popularity] for one movie
    matrix_2d = np.column_stack([vote_avg, popularity])

    logging.info(f"[NumPy] 2D matrix: shape={matrix_2d.shape}, ndim={matrix_2d.ndim}, "
                 f"size={matrix_2d.size}, itemsize={matrix_2d.itemsize}B")

    stats = {
        "vote_average": {
            "mean":              float(np.mean(vote_avg)),
            "std":               float(np.std(vote_avg)),
            "min":               float(np.min(vote_avg)),
            "max":               float(np.max(vote_avg)),
            "median":            float(np.median(vote_avg)),
            "percentile_25":     float(np.percentile(vote_avg, 25)),
            "percentile_75":     float(np.percentile(vote_avg, 75)),
            "above_mean_count":  int(np.sum(above_mean_mask)),
        },
        "popularity": {
            "mean":     float(np.mean(popularity)),
            "max":      float(np.max(popularity)),
            "log_mean": float(np.mean(log_popularity)),
        },
        "file_size_kb": {
            "mean":     float(np.mean(file_kb)),
            "total_mb": float(np.sum(file_kb) / 1024),
        },
        "weighted_score": {
            "mean": float(np.mean(weighted_score)),
            "max":  float(np.max(weighted_score)),
        },
    }

    for col_name, col_stats in stats.items():
        logging.info(f"[NumPy] {col_name}: {col_stats}")

    return stats, normalized


def run_numpy_analysis(df: pd.DataFrame) -> dict:
    """Full NumPy analysis: array creation + vectorized ops."""
    logging.info("[NumPy] Starting NumPy analysis")
    create_arrays_demo()
    stats, _ = vectorized_ops(df)
    logging.info("[NumPy] NumPy analysis complete")
    return stats


if __name__ == "__main__":
    from src.analytics.data_loader import load_from_mongo
    df = load_from_mongo()
    run_numpy_analysis(df)
