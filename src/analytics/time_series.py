"""
src/analytics/time_series.py
Lab 10: Parse dates, extract components, resample, and compute rolling averages.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import logging

ANALYTICS_DIR = Path("data/processed/analytics")


def parse_dates(df: pd.DataFrame, date_col: str = "published_date") -> pd.DataFrame:
    """Parse date column to datetime64 and report valid/missing counts."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    valid   = df[date_col].notna().sum()
    missing = df[date_col].isna().sum()
    logging.info("[TimeSeries] Parsed '%s': %d valid, %d missing", date_col, valid, missing)
    return df


def extract_date_components(df: pd.DataFrame,
                             date_col: str = "published_date") -> pd.DataFrame:
    """Add year, month, weekday, and quarter columns from the date column."""
    df = df.copy()
    dt = df[date_col]
    df["ts_year"]    = dt.dt.year
    df["ts_month"]   = dt.dt.month
    df["ts_weekday"] = dt.dt.day_name()
    df["ts_quarter"] = dt.dt.quarter
    logging.info("[TimeSeries] Extracted date components")
    return df


def monthly_citation_series(df: pd.DataFrame,
                              date_col: str = "published_date") -> pd.Series:
    """Build a monthly total-citations time series indexed by month."""
    df = df.copy()
    df = parse_dates(df, date_col)
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    monthly = df["citation_count"].resample("ME").sum()
    logging.info("[TimeSeries] Monthly series: %d months", len(monthly))
    return monthly


def resample_yearly(monthly: pd.Series) -> pd.Series:
    """Resample monthly series to yearly totals."""
    yearly = monthly.resample("YE").sum()
    logging.info("[TimeSeries] Yearly series: %d years", len(yearly))
    return yearly


def rolling_averages(monthly: pd.Series) -> pd.DataFrame:
    """Compute 3-, 6-, and 12-month rolling averages."""
    ra = pd.DataFrame({"monthly": monthly})
    ra["rolling_3"]  = monthly.rolling(window=3,  min_periods=1).mean()
    ra["rolling_6"]  = monthly.rolling(window=6,  min_periods=1).mean()
    ra["rolling_12"] = monthly.rolling(window=12, min_periods=1).mean()
    logging.info("[TimeSeries] Rolling averages computed")
    return ra


def save_rolling_chart(ra: pd.DataFrame,
                       out_dir: Path = ANALYTICS_DIR) -> Path:
    """Save monthly citations chart with 3/6/12-month rolling averages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(ra.index, ra["monthly"], color="lightgrey", label="Monthly Citations", alpha=0.6)
    ax.plot(ra.index, ra["rolling_3"],  color="steelblue",  linewidth=1.5, label="3-month avg")
    ax.plot(ra.index, ra["rolling_6"],  color="darkorange", linewidth=1.5, label="6-month avg")
    ax.plot(ra.index, ra["rolling_12"], color="darkgreen",  linewidth=1.5, label="12-month avg")

    ax.set_title("Monthly Citations with Rolling Averages")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Citations")
    ax.legend()
    plt.tight_layout()

    path = out_dir / "rolling_citations.png"
    fig.savefig(path)
    plt.close(fig)
    logging.info("[TimeSeries] Saved rolling_citations.png")
    return path


def run_time_series_pipeline(df: pd.DataFrame) -> dict:
    """Full time-series pipeline: parse → extract → monthly → rolling → chart."""
    df = parse_dates(df)
    df = extract_date_components(df)
    monthly = monthly_citation_series(df)
    yearly  = resample_yearly(monthly)
    ra      = rolling_averages(monthly)
    chart   = save_rolling_chart(ra)
    return {
        "parsed_df": df,
        "monthly":   monthly,
        "yearly":    yearly,
        "rolling":   ra,
        "chart":     chart,
    }
