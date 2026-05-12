"""
src/pipeline/generate_raw_csv.py

Reads the ArXiv papers JSON files collected by the pipeline and writes a
raw CSV to data/raw/csv/papers_raw.csv.  The CSV intentionally contains
the kinds of quality issues that Lab 9 (Data Cleaning) is designed to fix:

  - Missing values in several columns
  - Duplicate rows (exact and key-based)
  - Inconsistent string formatting (extra whitespace, mixed case)
  - Dates stored in mixed formats (ISO-8601, slash-separated, plain year)
  - A numeric column that contains non-numeric strings
  - A float column with out-of-range sentinel values (-1)
  - A low-cardinality text column with case inconsistencies
"""

import json
import csv
import random
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import logging

RAW_API_DIR = Path("data/raw/api")
OUTPUT_CSV   = Path("data/raw/csv/papers_raw.csv")

FIELDNAMES = [
    "paper_id",
    "title",
    "authors",
    "abstract",
    "published_date",
    "primary_category",
    "all_categories",
    "pdf_url",
    "citation_count",   # intentionally dirty numeric field
    "relevance_score",  # intentionally dirty float field
    "language",         # low-cardinality, intentionally inconsistent
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _authors_str(authors: list) -> str:
    return "; ".join(authors)


def _clean_record(paper: dict, idx: int) -> dict:
    """Baseline clean record from a JSON paper object."""
    cats = paper.get("categories", [])
    return {
        "paper_id":        paper["id"],
        "title":           paper["title"],
        "authors":         _authors_str(paper.get("authors", [])),
        "abstract":        paper.get("abstract", ""),
        "published_date":  paper.get("published", "")[:10],   # YYYY-MM-DD
        "primary_category": cats[0] if cats else "",
        "all_categories":  "|".join(cats),
        "pdf_url":         paper.get("pdf_url", ""),
        "citation_count":  str(random.randint(0, 120)),
        "relevance_score": f"{random.uniform(0.5, 1.0):.4f}",
        "language":        "en",
    }


def _inject_dirt(records: list[dict]) -> list[dict]:
    """Add realistic data-quality problems to a copy of the record list."""
    dirty = [r.copy() for r in records]

    # 1. Missing critical identifier (paper_id)
    dirty[2]["paper_id"] = ""

    # 2. Missing authors
    dirty[4]["authors"] = ""
    dirty[9]["authors"] = ""

    # 3. Missing / very short abstracts
    dirty[1]["abstract"] = ""
    dirty[6]["abstract"] = "N/A"
    dirty[11]["abstract"] = "  "

    # 4. Titles with extra whitespace and mixed case
    dirty[0]["title"] = "  " + dirty[0]["title"].upper() + "  "
    dirty[3]["title"] = dirty[3]["title"].lower()
    dirty[7]["title"] = "   " + dirty[7]["title"] + "   "

    # 5. Date format inconsistencies
    dirty[5]["published_date"]  = dirty[5]["published_date"].replace("-", "/")   # YYYY/MM/DD
    dirty[10]["published_date"] = dirty[10]["published_date"][:4]                 # YYYY only
    dirty[14]["published_date"] = "20260320"                                      # YYYYMMDD compact

    # 6. Primary category case / whitespace inconsistencies
    dirty[0]["primary_category"]  = dirty[0]["primary_category"].lower()
    dirty[8]["primary_category"]  = " " + dirty[8]["primary_category"] + " "
    dirty[12]["primary_category"] = dirty[12]["primary_category"].upper() + "  "

    # 7. Non-numeric citation_count
    dirty[3]["citation_count"]  = "unknown"
    dirty[7]["citation_count"]  = "N/A"
    dirty[13]["citation_count"] = ""

    # 8. Out-of-range relevance_score (sentinel -1 for missing)
    dirty[5]["relevance_score"]  = "-1"
    dirty[9]["relevance_score"]  = "-1"
    dirty[13]["relevance_score"] = "1.9500"   # > 1.0 — out of valid range

    # 9. Language field inconsistencies
    dirty[1]["language"]  = "EN"
    dirty[4]["language"]  = "English"
    dirty[8]["language"]  = " en "
    dirty[11]["language"] = "ENG"

    # 10. Exact duplicate rows (rows 0 and 2 duplicated at the end)
    dirty.append(dirty[0].copy())
    dirty.append(dirty[2].copy())

    # 11. Key duplicate: same paper_id as record 6 but slightly different title
    key_dup = dirty[6].copy()
    key_dup["title"] = key_dup["title"] + " (preprint)"
    key_dup["citation_count"] = "5"
    dirty.append(key_dup)

    return dirty


# ── main ─────────────────────────────────────────────────────────────────────

def generate_raw_csv(
    api_dir: Path = RAW_API_DIR,
    output: Path = OUTPUT_CSV,
    inject_dirt: bool = False,
) -> int:
    logging.info("[GenerateRawCSV] Reading JSON files from %s", api_dir)

    papers: list[dict] = []
    for json_file in sorted(api_dir.glob("papers_page_*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            batch = json.load(f)
            papers.extend(batch)
            logging.info("[GenerateRawCSV] Loaded %d papers from %s", len(batch), json_file.name)

    if not papers:
        logging.error("[GenerateRawCSV] No papers found — is data/raw/api/ populated?")
        return 0

    records = [_clean_record(p, i) for i, p in enumerate(papers)]
    rows = _inject_dirt(records) if inject_dirt else records

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("[GenerateRawCSV] Wrote %d rows to %s", len(rows), output)
    return len(rows)


if __name__ == "__main__":
    random.seed(42)
    count = generate_raw_csv()
    print(f"Generated {count} rows → {OUTPUT_CSV}")
