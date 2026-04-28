"""
src/analytics/regex_ops.py
Regular expression operations on movie text fields:
title, overview, genre_ids.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
import pandas as pd
from collections import Counter
from src.utils.logger import logging

# ── Pre-compiled patterns ─────────────────────────────────────────────────────

RE_YEAR_IN_TITLE  = re.compile(r"\((\d{4})\)")
RE_NUMBER         = re.compile(r"\d+")
RE_CRIME          = re.compile(
    r"\b(crime|murder|detective|thriller|heist|robbery|criminal|detective|suspect|gang)\b",
    re.IGNORECASE
)
RE_SEQUEL_NUMBER  = re.compile(r"\b([2-9]|I{2,}|Part\s+\d+)\b", re.IGNORECASE)
RE_TMDB_ID        = re.compile(r"^\d{1,10}$")
RE_STARTS_WITH    = {}   # populated lazily

# TMDb genre ID → name mapping
TMDB_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}


# ── Title operations ──────────────────────────────────────────────────────────

def extract_year_from_title(series: pd.Series) -> pd.Series:
    """Extract (YYYY) from titles like 'Dune (2021)'. Returns int Series (NaN if absent)."""
    result = series.str.extract(RE_YEAR_IN_TITLE.pattern, expand=False).astype(float)
    found  = result.notna().sum()
    logging.info(f"[Regex] extract_year_from_title: {found}/{len(series)} titles contain a year")
    return result


def filter_titles_by_prefix(series: pd.Series, prefix: str) -> pd.Series:
    """Return only titles that start with `prefix` (case-insensitive)."""
    pat    = re.compile(f"^{re.escape(prefix)}", re.IGNORECASE)
    result = series[series.str.match(pat, na=False)]
    logging.info(f"[Regex] filter_titles_by_prefix('{prefix}'): {len(result)} matches")
    return result


def find_sequel_titles(series: pd.Series) -> pd.Series:
    """Return titles that appear to be sequels (contain 2, II, Part 2, etc.)."""
    result = series[series.str.contains(RE_SEQUEL_NUMBER, na=False)]
    logging.info(f"[Regex] find_sequel_titles: {len(result)} potential sequels")
    return result


# ── Overview operations ───────────────────────────────────────────────────────

def count_crime_terms(series: pd.Series) -> pd.Series:
    """Count crime-related terms in each overview string."""
    result = series.fillna("").apply(lambda t: len(RE_CRIME.findall(t)))
    total  = result.sum()
    logging.info(f"[Regex] count_crime_terms: {total} total crime-related terms across {len(series)} overviews")
    return result


def find_short_overviews(series: pd.Series, max_len: int = 60) -> pd.Series:
    """Identify overviews that are unusually short (possible data quality issue)."""
    result = series[series.fillna("").str.len().between(1, max_len)]
    logging.info(f"[Regex] find_short_overviews(max_len={max_len}): {len(result)} short overviews")
    return result


def extract_numbers_from_overview(series: pd.Series) -> pd.Series:
    """Extract all numbers found in each overview (years, counts, etc.)."""
    result = series.fillna("").apply(lambda t: RE_NUMBER.findall(t))
    logging.info(f"[Regex] extract_numbers_from_overview: done for {len(series)} overviews")
    return result


# ── Genre parsing ─────────────────────────────────────────────────────────────

def parse_genres(df: pd.DataFrame) -> pd.Series:
    """
    Map genre_ids (list of ints) to genre name strings using TMDB_GENRES.
    Returns a Series of comma-separated genre strings.
    """
    if "genre_ids" not in df.columns:
        logging.warning("[Regex] 'genre_ids' column not found")
        return pd.Series([""] * len(df), index=df.index)

    def _map(ids):
        if not isinstance(ids, (list, tuple)):
            return ""
        return ", ".join(TMDB_GENRES.get(int(i), str(i)) for i in ids if i)

    result = df["genre_ids"].apply(_map)
    logging.info(f"[Regex] parse_genres: mapped genre_ids for {result.str.len().gt(0).sum()} rows")
    return result


def top_genres(df: pd.DataFrame, n: int = 10) -> list[tuple]:
    """Return the n most common genres across all movies."""
    genres_series = parse_genres(df)
    counter: Counter = Counter()
    for genre_str in genres_series:
        for g in genre_str.split(", "):
            g = g.strip()
            if g:
                counter[g] += 1
    top = counter.most_common(n)
    logging.info(f"[Regex] top_genres: {top}")
    return top


# ── ID validation ─────────────────────────────────────────────────────────────

def validate_movie_ids(series: pd.Series) -> pd.Series:
    """Return boolean mask: True where the value is a valid numeric TMDB ID."""
    result = series.astype(str).str.match(RE_TMDB_ID)
    valid  = result.sum()
    logging.info(f"[Regex] validate_movie_ids: {valid}/{len(series)} valid IDs")
    return result


def run_regex_demo(df: pd.DataFrame) -> dict:
    """Run all regex operations and return results."""
    logging.info("[Regex] Running regex demo")
    results = {}

    if "title" in df.columns:
        results["years_in_titles"]  = extract_year_from_title(df["title"])
        results["the_titles"]       = filter_titles_by_prefix(df["title"], "The")
        results["sequel_titles"]    = find_sequel_titles(df["title"])

    if "overview" in df.columns:
        results["crime_counts"]     = count_crime_terms(df["overview"])
        results["short_overviews"]  = find_short_overviews(df["overview"])
        results["overview_numbers"] = extract_numbers_from_overview(df["overview"])

    results["top_genres"]           = top_genres(df, 10)
    results["genre_strings"]        = parse_genres(df)

    if "movie_id" in df.columns:
        results["valid_ids"]        = validate_movie_ids(df["movie_id"])

    return results


if __name__ == "__main__":
    from src.analytics.data_loader import load_from_mongo
    df = load_from_mongo()
    if not df.empty:
        res = run_regex_demo(df)
        print("Top genres:", res.get("top_genres"))
        print("Sequel titles:\n", res.get("sequel_titles"))
