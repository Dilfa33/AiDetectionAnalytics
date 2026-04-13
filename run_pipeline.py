"""
run_pipeline.py
Full data pipeline:
  1. Extract content from PDF files in data/raw/pdf/
  2. Download movie posters from TMDb API
  3. Batch process images (resize, thumbnail, crop, WebP) -> MongoDB
  4. Extract EXIF metadata from camera photos in data/raw/exif_samples/
  5. Upload processed images to Google Drive
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pathlib import Path
from src.utils.logger import logging
from src.extraction.document_extractor import process_folder
from src.image_processing.downloader import download_posters
from src.image_processing.batch import batch_process_images
from src.image_processing.exif_utils import get_exif_summary
from src.storage.mongo import save_exif_to_mongo


def run_pdf_stage():
    logging.info("=" * 60)
    logging.info("STAGE 1: Extracting PDF documents")
    logging.info("=" * 60)
    folder  = Path("data/raw/pdf")
    results = process_folder(folder, extensions=[".pdf"])
    logging.info(f"PDF stage complete: {len(results)} file(s) processed")
    return results


def run_download_stage(count: int = 10):
    logging.info("=" * 60)
    logging.info("STAGE 2: Downloading movie posters from TMDb")
    logging.info("=" * 60)
    movies     = download_posters(count=count)
    downloaded = sum(1 for m in movies if m["local_path"])
    logging.info(f"Download stage complete: {downloaded}/{count} posters saved")
    return movies


def run_image_stage(movies: list):
    logging.info("=" * 60)
    logging.info("STAGE 3: Batch image processing")
    logging.info("=" * 60)
    lookup  = {m["filename"]: m for m in movies if m.get("filename")}
    results = batch_process_images(
        input_dir=Path("data/raw/images"),
        movie_lookup=lookup,
        upload_to_drive=True,
    )
    logging.info(f"Image stage complete: {len(results)} image(s) processed")
    return results


def run_exif_stage():
    logging.info("=" * 60)
    logging.info("STAGE 4: Extracting EXIF metadata from camera photos")
    logging.info("=" * 60)
    exif_dir = Path("data/raw/exif_samples")
    photos   = list(exif_dir.glob("*.jpg")) + list(exif_dir.glob("*.jpeg"))

    if not photos:
        logging.warning("[EXIF] No JPEG photos found in data/raw/exif_samples/")
        return []

    summaries = []
    for photo in photos:
        summary = get_exif_summary(photo)
        save_exif_to_mongo(summary)
        summaries.append(summary)
        logging.info(
            f"[EXIF] {photo.name}: camera={summary.get('camera_make')} {summary.get('camera_model')}, "
            f"date={summary.get('datetime_original')}, GPS={summary.get('gps')}"
        )

    logging.info(f"EXIF stage complete: {len(summaries)} photo(s) processed")
    return summaries


def run_drive_upload(image_results: list):
    logging.info("=" * 60)
    logging.info("STAGE 5: Uploading processed images to Google Drive")
    logging.info("=" * 60)
    try:
        from src.utils.upload_utils import get_drive_service, upload_batch
        service = get_drive_service()
        if not service:
            logging.error("[Drive] Could not authenticate — skipping upload")
            return []

        files_to_upload = []
        for r in image_results:
            for key in ("webp_path", "thumbnail_path"):
                p = Path(r.get(key, ""))
                if p.exists():
                    files_to_upload.append(p)

        results = upload_batch(service, files_to_upload)
        uploaded = sum(1 for r in results if r["drive_id"])
        logging.info(f"Drive stage complete: {uploaded}/{len(files_to_upload)} files uploaded")
        return results
    except Exception as e:
        logging.error(f"Drive stage failed: {e}")
        return []


if __name__ == "__main__":
    logging.info("Pipeline started")

    pdf_results    = run_pdf_stage()
    movies         = run_download_stage(count=10)
    image_results  = run_image_stage(movies)
    exif_results   = run_exif_stage()
    drive_results  = run_drive_upload(image_results)

    logging.info("=" * 60)
    logging.info("Pipeline summary:")
    logging.info(f"  PDFs processed:     {len(pdf_results)}")
    logging.info(f"  Posters downloaded: {sum(1 for m in movies if m['local_path'])}")
    logging.info(f"  Images processed:   {len(image_results)}")
    logging.info(f"  EXIF photos:        {len(exif_results)}")
    logging.info(f"  Drive uploads:      {sum(1 for r in drive_results if r['drive_id'])}")
    logging.info("Pipeline finished")
