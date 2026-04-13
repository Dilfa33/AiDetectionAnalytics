"""
src/image_processing/downloader.py
Fetches popular movie posters from the TMDb API and saves them locally.
Requires TMDB_API_KEY in your .env file.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import requests
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import logging

load_dotenv()

TMDB_API_KEY   = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL  = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"
RAW_IMAGE_DIR  = Path("data/raw/images")


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_popular_movies(page: int = 1) -> list[dict]:
    """Return a list of popular movie dicts from TMDb."""
    if not TMDB_API_KEY:
        logging.error("[Downloader] TMDB_API_KEY not set in .env")
        return []

    url = f"{TMDB_BASE_URL}/movie/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        movies = resp.json().get("results", [])
        logging.info(f"[Downloader] Fetched {len(movies)} movies from TMDb (page {page})")
        return movies
    except requests.RequestException as e:
        logging.error(f"[Downloader] Failed to fetch movies: {e}")
        return []


def download_poster(poster_path: str, save_dir: Path, filename: str) -> Path | None:
    """
    Download a single poster image from TMDb.
    poster_path: value of movie['poster_path'], e.g. '/abc123.jpg'
    Returns the saved file path or None on failure.
    """
    if not poster_path:
        logging.warning(f"[Downloader] No poster_path for {filename}")
        return None

    url = f"{TMDB_IMAGE_URL}{poster_path}"
    save_dir.mkdir(parents=True, exist_ok=True)
    dest = save_dir / filename

    if dest.exists():
        logging.info(f"[Downloader] Already exists, skipping: {filename}")
        return dest

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logging.info(f"[Downloader] Downloaded: {filename} ({len(resp.content) / 1024:.1f} KB)")
        return dest
    except requests.RequestException as e:
        logging.error(f"[Downloader] Failed to download {filename}: {e}")
        return None


def download_posters(count: int = 10, save_dir: Path = RAW_IMAGE_DIR) -> list[dict]:
    """
    Download `count` popular movie posters from TMDb.
    Returns a list of dicts with movie metadata + local path.
    """
    logging.info(f"[Downloader] Downloading {count} movie posters")
    movies   = fetch_popular_movies(page=1)
    results  = []

    for movie in movies[:count]:
        movie_id    = movie.get("id")
        title       = movie.get("title", "unknown").replace(" ", "_")
        poster_path = movie.get("poster_path")
        filename    = f"{movie_id}_{title}.jpg"

        local_path = download_poster(poster_path, save_dir, filename)

        results.append({
            "movie_id":     movie_id,
            "title":        movie.get("title"),
            "poster_path":  poster_path,
            "local_path":   str(local_path) if local_path else None,
            "filename":     filename,
            "overview":     movie.get("overview", ""),
            "release_date": movie.get("release_date", ""),
            "vote_average": movie.get("vote_average", 0),
        })

    downloaded = sum(1 for r in results if r["local_path"])
    logging.info(f"[Downloader] {downloaded}/{count} posters downloaded to {save_dir}")
    return results


if __name__ == "__main__":
    results = download_posters(count=10)
    for r in results:
        print(f"  {r['title']} -> {r['local_path']}")
