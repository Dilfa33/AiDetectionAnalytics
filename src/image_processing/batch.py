"""
src/image_processing/batch.py
Batch image processing pipeline with tqdm progress tracking.
Processes all images in data/raw/images/:
  - Resize  -> data/processed/resized/
  - Thumbnail -> data/processed/thumbnails/
  - WebP    -> data/processed/webp/
  - Crop    -> data/processed/cropped/
Stores metadata in MongoDB and optionally uploads to Google Drive.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.utils.logger import logging
from src.image_processing.processor import (
    inspect_image, save_resized, save_thumbnail,
    save_cropped, convert_to_webp,
    apply_filter, apply_enhancement,
)
from src.storage.mongo import save_image_metadata

RAW_IMAGE_DIR = Path("data/raw/images")
OUTPUT_DIRS = {
    "resized":    Path("data/processed/resized"),
    "thumbnails": Path("data/processed/thumbnails"),
    "webp":       Path("data/processed/webp"),
    "cropped":    Path("data/processed/cropped"),
}


# ── Single image ──────────────────────────────────────────────────────────────

def process_single(image_path: Path, movie_metadata: dict = None) -> dict | None:
    """
    Process one image file: resize, thumbnail, crop, and WebP conversion.
    Returns a metadata dict ready for MongoDB storage.
    """
    image_path = Path(image_path)

    # Ensure output dirs exist
    for d in OUTPUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    try:
        props = inspect_image(image_path)

        with Image.open(image_path) as img:
            resized_path   = save_resized(img, image_path.name)
            thumb_path     = save_thumbnail(img, image_path.name)
            cropped_path   = save_cropped(img, image_path.name)
            webp_path      = convert_to_webp(img, image_path.name)

        meta = movie_metadata or {}
        record = {
            # Image identity
            "movie_id":          meta.get("movie_id"),
            "title":             meta.get("title", image_path.stem),
            "source":            "tmdb_api",
            "type":              "poster",
            "filename":          image_path.name,
            # Movie metadata from TMDb
            "overview":          meta.get("overview", ""),
            "release_date":      meta.get("release_date", ""),
            "vote_average":      meta.get("vote_average", 0.0),
            "vote_count":        meta.get("vote_count", 0),
            "popularity":        meta.get("popularity", 0.0),
            "original_language": meta.get("original_language", ""),
            "genre_ids":         meta.get("genre_ids", []),
            # Image paths
            "original_path":     str(image_path),
            "resized_path":      str(resized_path),
            "thumbnail_path":    str(thumb_path),
            "webp_path":         str(webp_path),
            "cropped_path":      str(cropped_path),
            # Image properties
            "format":            props.get("format"),
            "mode":              props.get("mode"),
            "width":             props.get("width"),
            "height":            props.get("height"),
            "aspect_ratio":      props.get("aspect_ratio"),
            "file_size_bytes":   props.get("file_size_bytes"),
            "file_size_kb":      props.get("file_size_kb"),
            "exif":              {"camera_make": None, "gps": None},
        }

        logging.info(f"[Batch] Processed: {image_path.name}")
        return record

    except Exception as e:
        logging.error(f"[Batch] Failed to process {image_path.name}: {e}")
        return None


# ── Batch ─────────────────────────────────────────────────────────────────────

def batch_process_images(
    input_dir: Path = RAW_IMAGE_DIR,
    movie_lookup: dict = None,
    upload_to_drive: bool = False,
) -> list[dict]:
    """
    Process all JPEG/PNG images in input_dir.
    movie_lookup: optional dict mapping filename -> movie metadata dict.
    upload_to_drive: if True, upload WebP and thumbnail to Google Drive.
    Returns list of processed metadata records.
    """
    input_dir  = Path(input_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images     = [f for f in input_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in extensions]

    if not images:
        logging.warning(f"[Batch] No images found in {input_dir}")
        return []

    logging.info(f"[Batch] Starting batch processing: {len(images)} image(s)")
    results = []

    for img_path in tqdm(images, desc="Processing images", unit="img"):
        meta   = (movie_lookup or {}).get(img_path.name)
        record = process_single(img_path, meta)

        if record:
            # Store metadata in MongoDB
            save_image_metadata(record)
            results.append(record)

            # Optional Google Drive upload
            if upload_to_drive:
                try:
                    from src.utils.upload_utils import get_drive_service, upload_batch
                    service = get_drive_service()
                    files_to_upload = [
                        Path(record["webp_path"]),
                        Path(record["thumbnail_path"]),
                    ]
                    upload_batch(service, files_to_upload)
                except Exception as e:
                    logging.error(f"[Batch] Google Drive upload failed for {img_path.name}: {e}")

    logging.info(f"[Batch] Complete: {len(results)}/{len(images)} images processed successfully")
    return results


if __name__ == "__main__":
    results = batch_process_images()
    print(f"\nBatch complete: {len(results)} images processed")
    for r in results:
        print(f"  {r['filename']} -> resized={r['resized_path']}")
