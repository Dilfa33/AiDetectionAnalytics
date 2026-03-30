"""
src/extraction/document_extractor.py
Extracts text, tables, and metadata from PDF, Word, and Excel files.
Handles encoding detection and stores results in MongoDB.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
from pathlib import Path
from datetime import datetime

import chardet
import pdfplumber
from docx import Document
import openpyxl

from src.utils.logger import logging
from src.storage.mongo import save_document_to_mongo


# ── Encoding helpers ──────────────────────────────────────────────────────────

def detect_encoding(file_path):
    """Detect the encoding of a file using chardet."""
    with open(file_path, "rb") as f:
        raw = f.read(65536)  # read up to 64 KB for detection
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    confidence = result.get("confidence", 0)
    logging.info(f"Detected encoding for {Path(file_path).name}: {encoding} (confidence: {confidence:.2f})")
    return encoding


def safe_decode(text):
    """Normalize and clean extracted text: collapse whitespace, strip nulls."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── PDF extraction ────────────────────────────────────────────────────────────

def _is_two_column(page):
    """Heuristic: if the page has words clustered in two x-bands, treat as two-column."""
    words = page.extract_words()
    if not words:
        return False
    midpoint = page.width / 2
    left = sum(1 for w in words if w["x0"] < midpoint - 20)
    right = sum(1 for w in words if w["x0"] > midpoint + 20)
    total = len(words)
    if total == 0:
        return False
    # Both columns should each hold at least 20% of words
    return (left / total > 0.2) and (right / total > 0.2)


def _extract_two_column_text(page):
    """Extract text from a two-column page by splitting at the page midpoint."""
    mid = page.width / 2
    left_bbox  = (0,       page.bbox[1], mid,          page.bbox[3])
    right_bbox = (mid,     page.bbox[1], page.bbox[2], page.bbox[3])

    left_text  = page.within_bbox(left_bbox).extract_text()  or ""
    right_text = page.within_bbox(right_bbox).extract_text() or ""
    return left_text + "\n" + right_text


def extract_pdf(file_path):
    """
    Extract text and tables from a PDF file.
    Detects single vs. two-column layout per page.
    Returns a list of page-level dicts.
    """
    file_path = Path(file_path)
    logging.info(f"[PDF] Starting extraction: {file_path.name}")
    pages_data = []

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            total = len(pdf.pages)
            logging.info(f"[PDF] {total} pages found in {file_path.name}")

            for i, page in enumerate(pdf.pages, start=1):
                page_info = {
                    "page_number": i,
                    "layout": "unknown",
                    "text": "",
                    "tables": []
                }

                # Detect layout
                two_col = _is_two_column(page)
                page_info["layout"] = "two-column" if two_col else "single-column"

                # Extract text
                if two_col:
                    raw_text = _extract_two_column_text(page)
                else:
                    raw_text = page.extract_text() or ""

                page_info["text"] = safe_decode(raw_text)

                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for t_idx, table in enumerate(tables):
                        cleaned = [
                            [safe_decode(cell) if cell else "" for cell in row]
                            for row in table
                        ]
                        page_info["tables"].append({
                            "table_index": t_idx,
                            "rows": cleaned
                        })
                    logging.info(f"[PDF] Page {i}: {len(tables)} table(s) found")

                pages_data.append(page_info)

        logging.info(f"[PDF] Extraction complete: {file_path.name} ({total} pages)")

    except Exception as e:
        logging.error(f"[PDF] Failed to extract {file_path.name}: {e}")

    return pages_data


# ── Word extraction ───────────────────────────────────────────────────────────

def extract_word(file_path):
    """
    Extract paragraphs, runs, and tables from a .docx file.
    Returns a dict with paragraphs list and tables list.
    """
    file_path = Path(file_path)
    logging.info(f"[WORD] Starting extraction: {file_path.name}")
    result = {"paragraphs": [], "tables": []}

    try:
        doc = Document(str(file_path))

        # Paragraphs
        for para in doc.paragraphs:
            text = safe_decode(para.text)
            if not text:
                continue
            runs_text = [safe_decode(run.text) for run in para.runs if run.text.strip()]
            result["paragraphs"].append({
                "style": para.style.name,
                "text": text,
                "runs": runs_text
            })

        # Tables
        for t_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [safe_decode(cell.text) for cell in row.cells]
                rows.append(cells)
            result["tables"].append({
                "table_index": t_idx,
                "rows": rows
            })

        logging.info(
            f"[WORD] Extraction complete: {file_path.name} "
            f"({len(result['paragraphs'])} paragraphs, {len(result['tables'])} tables)"
        )

    except Exception as e:
        logging.error(f"[WORD] Failed to extract {file_path.name}: {e}")

    return result


# ── Excel extraction ──────────────────────────────────────────────────────────

def extract_excel(file_path):
    """
    Extract data from all sheets in an Excel file using openpyxl.
    Reads cell values (not formula strings) where possible.
    Returns a dict keyed by sheet name.
    """
    file_path = Path(file_path)
    logging.info(f"[EXCEL] Starting extraction: {file_path.name}")
    result = {}

    try:
        # data_only=True returns cached formula results instead of formula strings
        wb = openpyxl.load_workbook(str(file_path), data_only=True)

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cleaned = [safe_decode(str(cell)) if cell is not None else "" for cell in row]
                rows.append(cleaned)

            result[sheet_name] = {
                "dimensions": ws.dimensions,
                "rows": rows
            }
            logging.info(
                f"[EXCEL] Sheet '{sheet_name}': {ws.max_row} rows × {ws.max_column} cols"
            )

        logging.info(f"[EXCEL] Extraction complete: {file_path.name} ({len(result)} sheets)")

    except Exception as e:
        logging.error(f"[EXCEL] Failed to extract {file_path.name}: {e}")

    return result


# ── Encoding-safe text file extraction ───────────────────────────────────────

def extract_text_file(file_path):
    """Read a plain-text file with automatic encoding detection."""
    file_path = Path(file_path)
    enc = detect_encoding(file_path)
    try:
        with open(file_path, "r", encoding=enc, errors="replace") as f:
            text = f.read()
        return safe_decode(text)
    except Exception as e:
        logging.error(f"[TEXT] Failed to read {file_path.name}: {e}")
        return ""


# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract_document(file_path):
    """
    Detect file type, extract content, attach metadata, and save to MongoDB.
    Returns the MongoDB inserted_id or None on failure.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        doc_type = "pdf"
        content = {"pages": extract_pdf(file_path)}
    elif suffix in (".docx", ".doc"):
        doc_type = "word"
        content = extract_word(file_path)
    elif suffix in (".xlsx", ".xls", ".xlsm"):
        doc_type = "excel"
        content = extract_excel(file_path)
    else:
        logging.warning(f"Unsupported file type: {suffix} ({file_path.name})")
        return None

    # Attach metadata
    document = {
        "file_name": file_path.name,
        "document_type": doc_type,
        "source": str(file_path),
        "extraction_timestamp": datetime.utcnow().isoformat(),
        "extraction_library": {
            "pdf":   "pdfplumber",
            "word":  "python-docx",
            "excel": "openpyxl"
        }.get(doc_type, "unknown"),
        "content": content
    }

    inserted_id = save_document_to_mongo(document, source=file_path.name)
    return inserted_id


# ── Bulk folder processing ────────────────────────────────────────────────────

def process_folder(folder_path, extensions=None):
    """
    Process all supported documents in a folder.
    extensions: list of suffixes to include, e.g. ['.pdf', '.docx']
    """
    folder = Path(folder_path)
    if not folder.exists():
        logging.warning(f"Folder not found: {folder}")
        return []

    default_exts = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".xlsm"}
    exts = set(extensions) if extensions else default_exts

    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts]
    logging.info(f"Found {len(files)} document(s) in {folder}")

    results = []
    for f in files:
        inserted_id = extract_document(f)
        results.append({"file": f.name, "inserted_id": str(inserted_id)})

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = Path("data/raw")
    for subfolder in ("pdf", "word", "excel"):
        folder = base / subfolder
        logging.info(f"=== Processing folder: {folder} ===")
        process_folder(folder)
