"""
tests/test_cleaning.py
Unit tests for the src/cleaning package.
Run: pytest tests/test_cleaning.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cleaning.missing_handler import (
    report_missing,
    drop_missing_ids,
    fill_missing_authors,
    fill_missing_abstracts,
    fill_missing_numeric,
    drop_high_missing_cols,
)
from src.cleaning.string_cleaner import (
    clean_titles,
    normalize_language,
    clean_abstract,
    extract_year,
    clean_categories,
)
from src.cleaning.deduplicator import (
    count_duplicates,
    drop_exact_duplicates,
    drop_key_duplicates,
)
from src.cleaning.type_converter import (
    convert_dates,
    convert_numerics,
    convert_categories,
    memory_report,
)
from src.cleaning.validator import (
    validate_ids,
    validate_relevance,
    validate_dates,
    validate_no_duplicates,
    run_all_validations,
)


# ── shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def dirty_df():
    """Small DataFrame with every type of quality issue used in tests."""
    return pd.DataFrame({
        "paper_id": ["2603.001", "", "2603.003", "2603.001", np.nan],
        "title": [
            "  deep learning survey  ",
            "NEURAL NETWORKS",
            "  transformer models  ",
            "  deep learning survey  ",
            "reinforcement learning",
        ],
        "authors": ["Alice; Bob", np.nan, "Carol", "Alice; Bob", "Dave"],
        "abstract": [
            "A thorough survey of deep learning.",
            "",
            "N/A",
            "A thorough survey of deep learning.",
            "   ",
        ],
        "published_date": [
            "2026-03-20",
            "2026/03/21",
            "20260322",
            "2026-03-20",
            "2026",
        ],
        "primary_category": ["cs.AI", " cs.LG ", "CS.CV  ", "cs.AI", "cs.CL"],
        "all_categories": ["cs.AI", "cs.LG", "cs.CV", "cs.AI", "cs.CL"],
        "pdf_url": [
            "https://arxiv.org/pdf/2603.001",
            "https://arxiv.org/pdf/2603.002",
            "https://arxiv.org/pdf/2603.003",
            "https://arxiv.org/pdf/2603.001",
            "https://arxiv.org/pdf/2603.004",
        ],
        "citation_count": ["10", "unknown", "5", "10", np.nan],
        "relevance_score": ["0.9", "-1", "0.7", "0.9", "1.5"],
        "language": ["en", "EN", " en ", "en", "English"],
    })


# ── missing_handler tests ─────────────────────────────────────────────────────

class TestMissingHandler:
    def test_report_missing_returns_dataframe(self, dirty_df):
        report = report_missing(dirty_df)
        assert isinstance(report, pd.DataFrame)
        assert "missing_count" in report.columns
        assert "missing_ratio" in report.columns

    def test_report_missing_counts_correct(self, dirty_df):
        report = report_missing(dirty_df)
        assert report.loc["authors", "missing_count"] == 1
        assert report.loc["paper_id", "missing_count"] == 1

    def test_drop_missing_ids_removes_null_and_empty(self, dirty_df):
        result = drop_missing_ids(dirty_df)
        assert result["paper_id"].isnull().sum() == 0
        assert (result["paper_id"].astype(str).str.strip() == "").sum() == 0
        assert len(result) == 3  # 5 rows minus empty and NaN

    def test_fill_missing_authors(self, dirty_df):
        result = fill_missing_authors(dirty_df)
        assert result["authors"].isnull().sum() == 0
        assert "Unknown" in result["authors"].values

    def test_fill_missing_abstracts_replaces_empty_and_na(self, dirty_df):
        result = fill_missing_abstracts(dirty_df)
        assert result["abstract"].isnull().sum() == 0
        assert "N/A" not in result["abstract"].values
        assert "" not in result["abstract"].values

    def test_fill_missing_numeric_handles_bad_values(self, dirty_df):
        result = fill_missing_numeric(dirty_df)
        assert pd.to_numeric(result["citation_count"], errors="coerce").isnull().sum() == 0
        scores = pd.to_numeric(result["relevance_score"], errors="coerce")
        assert (scores == -1).sum() == 0

    def test_drop_high_missing_cols_removes_columns(self):
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan, np.nan],
            "b": [1, 2, 3, 4],
        })
        result = drop_high_missing_cols(df, threshold=0.5)
        assert "a" not in result.columns
        assert "b" in result.columns


# ── string_cleaner tests ──────────────────────────────────────────────────────

class TestStringCleaner:
    def test_clean_titles_strips_whitespace(self, dirty_df):
        result = clean_titles(dirty_df)
        for title in result["title"]:
            assert title == title.strip()

    def test_clean_titles_applies_title_case(self, dirty_df):
        result = clean_titles(dirty_df)
        assert result["title"].iloc[1] == "Neural Networks"

    def test_normalize_language_maps_variants(self, dirty_df):
        result = normalize_language(dirty_df)
        assert set(result["language"].unique()) == {"en"}

    def test_normalize_language_strips_whitespace(self, dirty_df):
        result = normalize_language(dirty_df)
        for lang in result["language"]:
            assert lang == lang.strip()

    def test_clean_abstract_replaces_na(self, dirty_df):
        result = clean_abstract(dirty_df)
        assert "N/A" not in result["abstract"].dropna().values

    def test_extract_year_iso_date(self, dirty_df):
        result = extract_year(dirty_df)
        assert "published_year" in result.columns
        assert result["published_year"].iloc[0] == 2026

    def test_extract_year_compact_date(self, dirty_df):
        result = extract_year(dirty_df)
        assert result["published_year"].iloc[2] == 2026

    def test_extract_year_plain_year(self, dirty_df):
        result = extract_year(dirty_df)
        assert result["published_year"].iloc[4] == 2026

    def test_clean_categories_strips_whitespace(self, dirty_df):
        result = clean_categories(dirty_df)
        for cat in result["primary_category"]:
            assert cat == cat.strip()


# ── deduplicator tests ────────────────────────────────────────────────────────

class TestDeduplicator:
    def test_count_duplicates_full_rows(self, dirty_df):
        assert count_duplicates(dirty_df) == 1  # row 3 is exact dup of row 0

    def test_count_duplicates_by_column(self, dirty_df):
        assert count_duplicates(dirty_df, col="paper_id") >= 1

    def test_drop_exact_duplicates_removes_copies(self, dirty_df):
        result = drop_exact_duplicates(dirty_df)
        assert result.duplicated().sum() == 0
        assert len(result) == len(dirty_df) - 1

    def test_drop_key_duplicates_keeps_first(self, dirty_df):
        df_clean = drop_missing_ids(dirty_df)
        result = drop_key_duplicates(df_clean, key="paper_id")
        assert result["paper_id"].duplicated().sum() == 0

    def test_drop_key_duplicates_reduces_rows(self, dirty_df):
        df_clean = drop_missing_ids(dirty_df)
        before = len(df_clean)
        result = drop_key_duplicates(df_clean, key="paper_id")
        assert len(result) < before


# ── type_converter tests ──────────────────────────────────────────────────────

class TestTypeConverter:
    def test_convert_dates_iso(self, dirty_df):
        result = convert_dates(dirty_df)
        assert pd.api.types.is_datetime64_any_dtype(result["published_date"])

    def test_convert_dates_handles_formats(self, dirty_df):
        result = convert_dates(dirty_df)
        not_nat = result["published_date"].dropna()
        assert len(not_nat) >= 4  # most rows should parse

    def test_convert_numerics_citation_count(self, dirty_df):
        result = fill_missing_numeric(dirty_df)
        result = convert_numerics(result)
        assert str(result["citation_count"].dtype) == "Int64"

    def test_convert_numerics_relevance_score(self, dirty_df):
        result = fill_missing_numeric(dirty_df)
        result = convert_numerics(result)
        assert result["relevance_score"].dtype == np.float32

    def test_convert_categories_dtype(self, dirty_df):
        result = convert_categories(dirty_df)
        assert str(result["primary_category"].dtype) == "category"

    def test_memory_report_returns_dataframe(self, dirty_df):
        after = convert_categories(dirty_df)
        report = memory_report(dirty_df, after)
        assert isinstance(report, pd.DataFrame)
        assert "before_bytes" in report.columns
        assert "after_bytes" in report.columns


# ── validator tests ───────────────────────────────────────────────────────────

class TestValidator:
    @pytest.fixture
    def clean_df(self, dirty_df):
        """Run the full pipeline to get a clean DataFrame for validation tests."""
        df = drop_missing_ids(dirty_df)
        df = fill_missing_authors(df)
        df = fill_missing_abstracts(df)
        df = fill_missing_numeric(df)
        df = drop_exact_duplicates(df)
        df = drop_key_duplicates(df)
        df = convert_dates(df)
        df = convert_numerics(df)
        # Clamp out-of-range relevance scores
        df["relevance_score"] = df["relevance_score"].clip(0.0, 1.0)
        return df

    def test_validate_ids_passes_on_clean_data(self, clean_df):
        assert validate_ids(clean_df) is True

    def test_validate_ids_fails_on_null(self, dirty_df):
        with pytest.raises(AssertionError):
            validate_ids(dirty_df)

    def test_validate_relevance_passes_on_clean_data(self, clean_df):
        assert validate_relevance(clean_df) is True

    def test_validate_relevance_fails_on_out_of_range(self):
        # Build a DataFrame that bypasses fill_missing_numeric so invalid value stays
        df = pd.DataFrame({
            "paper_id": ["x.001", "x.002"],
            "relevance_score": [0.8, 1.9],  # 1.9 is out of range
        })
        df["relevance_score"] = df["relevance_score"].astype("float32")
        with pytest.raises(AssertionError):
            validate_relevance(df)

    def test_validate_dates_passes_on_clean_data(self, clean_df):
        assert validate_dates(clean_df) is True

    def test_validate_dates_fails_on_string_col(self, dirty_df):
        with pytest.raises(AssertionError):
            validate_dates(dirty_df)

    def test_validate_no_duplicates_passes_on_clean_data(self, clean_df):
        assert validate_no_duplicates(clean_df) is True

    def test_run_all_validations_returns_dict(self, clean_df):
        results = run_all_validations(clean_df)
        assert isinstance(results, dict)
        assert all(v == "PASSED" for v in results.values()), results
