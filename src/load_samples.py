import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from io_utils import read_json, setup_logging, read_text


if __name__ == "__main__":
    setup_logging("pipeline.log")

    # Read the raw ArXiv XML response saved in data/json/
    # Note: .atom is XML text so we read it as plain text
    atom_text = read_text("data/json/response.atom")

    # Read the first PDF path from data/raw/pdf/
    pdf_dir = "data/raw/pdf"
    pdf_files = list(Path(pdf_dir).glob("*.pdf")) if Path(pdf_dir).exists() else []

    if atom_text:
        print("ArXiv response preview (first 100 characters):")
        print(atom_text[:100])

    if pdf_files:
        pdf = pdf_files[0]
        size_kb = pdf.stat().st_size // 1024
        print(f"\nPDF on disk: {pdf.name} ({size_kb} KB)")
    else:
        print("\nNo PDFs found in data/raw/pdf/")