"""
src/parsing/parsers.py
Parses JSON, CSV, and XML data files and saves to MongoDB.
Run: python parsers.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from src.utils.logger import logging
from src.storage.mongo import save_to_mongo


# ── JSON Parsing ──────────────────────────────────────────────────────────────
def parse_json_files():
    """Parse all JSON files from data/raw/api/ and save to MongoDB."""
    api_dir = Path("data/raw/api")
    files = list(api_dir.glob("*.json")) if api_dir.exists() else []

    if not files:
        logging.warning("No JSON files found in data/raw/api/ — run client.py first")
        return []

    all_papers = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                papers = json.load(f)
            logging.info(f"Parsed JSON: {file.name} ({len(papers)} records)")
            for paper in papers:
                parsed = extract_paper_fields(paper)
                save_to_mongo(parsed, source=file.name)
                all_papers.append(parsed)
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in file: {file}")
        except FileNotFoundError:
            logging.error(f"File not found: {file}")

    return all_papers


# ── CSV Parsing ───────────────────────────────────────────────────────────────
def parse_csv_file(file_path):
    """Parse a CSV file and save each row to MongoDB."""
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        logging.info(f"Parsed CSV: {file_path} ({len(rows)} rows)")
        for row in rows:
            save_to_mongo(row, source="csv")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"CSV parse error: {e}")
    return rows


# ── XML Parsing ───────────────────────────────────────────────────────────────
def parse_xml_file(file_path):
    """Parse an XML file and save each entry to MongoDB."""
    records = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for paper in root.findall("paper"):
            record = {child.tag: child.text for child in paper}
            save_to_mongo(record, source="xml")
            records.append(record)
        logging.info(f"Parsed XML: {file_path} ({len(records)} records)")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except ET.ParseError as e:
        logging.error(f"XML parse error in {file_path}: {e}")
    return records


# ── Field Extractor ───────────────────────────────────────────────────────────
def extract_paper_fields(paper):
    """Extract and normalize key fields from a raw ArXiv paper dict."""
    return {
        "id": paper.get("id", ""),
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "published": paper.get("published", ""),
        "categories": paper.get("categories", []),
        "pdf_url": paper.get("pdf_url", ""),
        "abs_url": paper.get("abs_url", "")
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parse_json_files()
    parse_csv_file("data/raw/csv/sample.csv")
    parse_xml_file("data/raw/xml/sample.xml")
