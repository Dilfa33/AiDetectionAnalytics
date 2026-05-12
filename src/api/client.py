"""
src/api/client.py
Fetches AI papers from ArXiv API with pagination, error handling, and retry logic.
Run: python client.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import requests
import xml.etree.ElementTree as ET
import json
import time
from pathlib import Path
from src.utils.logger import logging

# ── Config ────────────────────────────────────────────────────────────────────
ARXIV_API_URL = "https://export.arxiv.org/api/query"
RESULTS_PER_PAGE = 10
SEARCH_TOPIC = "AI generated content detection"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom"
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_page(topic, page=0, page_size=RESULTS_PER_PAGE, retries=3):
    """
    Fetch a single page of ArXiv results.
    Implements retry logic with exponential backoff for rate limits.
    """
    params = {
        "search_query": f"all:{topic.replace(' ', '+')}",
        "start": page * page_size,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    for attempt in range(retries):
        try:
            logging.info(f"Fetching page {page + 1} (attempt {attempt + 1})")
            response = requests.get(ARXIV_API_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error on page {page + 1}: {e}")
            if response.status_code == 429:
                wait = 2 ** attempt
                logging.warning(f"Rate limited — waiting {wait}s before retry")
                time.sleep(wait)
        except requests.exceptions.ConnectionError:
            logging.error("Connection error — check your internet")
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            logging.error(f"Timeout on page {page + 1}")
            time.sleep(2 ** attempt)

    logging.error(f"Failed to fetch page {page + 1} after {retries} attempts")
    return None


def parse_page(xml_text):
    """Parse ArXiv Atom XML into list of paper dicts."""
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall("atom:entry", NS):
        raw_id = entry.find("atom:id", NS).text.strip()
        clean_id = raw_id.split("/abs/")[-1].split("v")[0]

        authors = [
            a.find("atom:name", NS).text
            for a in entry.findall("atom:author", NS)
        ]
        categories = [t.get("term") for t in entry.findall("atom:category", NS)]

        papers.append({
            "id": clean_id,
            "title": entry.find("atom:title", NS).text.strip().replace("\n", " "),
            "authors": authors,
            "abstract": entry.find("atom:summary", NS).text.strip().replace("\n", " "),
            "published": entry.find("atom:published", NS).text.strip(),
            "categories": categories,
            "pdf_url": f"https://arxiv.org/pdf/{clean_id}",
            "abs_url": f"https://arxiv.org/abs/{clean_id}"
        })

    return papers


def save_page(papers, page_num):
    """Save a page of results as JSON to data/raw/api/"""
    out_dir = Path("data/raw/api")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"papers_page_{page_num}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)

    logging.info(f"Saved {len(papers)} papers to {out_path}")


def fetch_papers(pages=3):
    """
    Fetch multiple pages of ArXiv papers.
    Saves each page as a separate JSON file.
    Returns all papers as a flat list.
    """
    all_papers = []

    for page in range(pages):
        xml_text = fetch_page(SEARCH_TOPIC, page=page)
        if not xml_text:
            logging.warning(f"Skipping page {page + 1} — no data returned")
            continue

        papers = parse_page(xml_text)
        save_page(papers, page + 1)
        all_papers.extend(papers)

        # Be polite to ArXiv API — 3 second delay between pages
        if page < pages - 1:
            logging.info("Waiting 3s before next page (ArXiv rate limit)")
            time.sleep(3)

    logging.info(f"Total papers fetched: {len(all_papers)}")
    return all_papers


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    papers = fetch_papers(pages=3)
    print(f"Fetched {len(papers)} papers.")
    if papers:
        print("Here are some paper titles:")
        for paper in papers[:5]:
            print(f"  - {paper['title'][:80]}")
