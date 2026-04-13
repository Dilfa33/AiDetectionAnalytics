"""
src/image_processing/exif_utils.py
EXIF metadata extraction helpers using Pillow.
Works on JPEG images taken with cameras or smartphones.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from src.utils.logger import logging


# ── Raw EXIF ──────────────────────────────────────────────────────────────────

def extract_exif(path: str | Path) -> dict:
    """
    Extract all EXIF tags from a JPEG image.
    Returns a dict of {tag_name: value} or empty dict if no EXIF.
    """
    path = Path(path)
    try:
        with Image.open(path) as img:
            raw = img._getexif()

        if not raw:
            logging.warning(f"[EXIF] No EXIF data found in {path.name}")
            return {}

        exif = {}
        for tag_id, value in raw.items():
            tag = TAGS.get(tag_id, str(tag_id))
            # Convert bytes to string for readability
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = repr(value)
            exif[tag] = value

        logging.info(f"[EXIF] Extracted {len(exif)} tags from {path.name}")
        return exif

    except Exception as e:
        logging.error(f"[EXIF] Failed to extract from {path.name}: {e}")
        return {}


# ── GPS helpers ───────────────────────────────────────────────────────────────

def _convert_to_degrees(value) -> float:
    """Convert GPS coordinate stored as (degrees, minutes, seconds) to decimal degrees."""
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return 0.0


def get_gps_coordinates(exif_data: dict) -> dict | None:
    """
    Parse GPS latitude and longitude from raw EXIF data.
    Returns {'latitude': float, 'longitude': float} or None if not present.
    """
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None

    try:
        # Decode GPS sub-IFD tags
        gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

        lat  = _convert_to_degrees(gps["GPSLatitude"])
        lon  = _convert_to_degrees(gps["GPSLongitude"])

        if gps.get("GPSLatitudeRef",  "N") != "N":
            lat = -lat
        if gps.get("GPSLongitudeRef", "E") != "E":
            lon = -lon

        return {"latitude": round(lat, 6), "longitude": round(lon, 6)}

    except (KeyError, TypeError, ZeroDivisionError) as e:
        logging.warning(f"[EXIF] Could not parse GPS: {e}")
        return None


# ── Clean summary ─────────────────────────────────────────────────────────────

def get_exif_summary(path: str | Path) -> dict:
    """
    Return a cleaned EXIF summary with the most useful fields.
    Includes GPS coordinates when available.
    """
    path     = Path(path)
    raw_exif = extract_exif(path)

    if not raw_exif:
        return {"file": path.name, "exif_available": False}

    gps = get_gps_coordinates(raw_exif)

    summary = {
        "file":              path.name,
        "exif_available":    True,
        "camera_make":       raw_exif.get("Make"),
        "camera_model":      raw_exif.get("Model"),
        "lens_model":        raw_exif.get("LensModel"),
        "datetime_original": raw_exif.get("DateTimeOriginal"),
        "datetime":          raw_exif.get("DateTime"),
        "exposure_time":     str(raw_exif.get("ExposureTime", "")),
        "f_number":          str(raw_exif.get("FNumber", "")),
        "iso":               raw_exif.get("ISOSpeedRatings"),
        "focal_length":      str(raw_exif.get("FocalLength", "")),
        "flash":             raw_exif.get("Flash"),
        "orientation":       raw_exif.get("Orientation"),
        "software":          raw_exif.get("Software"),
        "gps":               gps,
    }

    logging.info(
        f"[EXIF] Summary for {path.name}: "
        f"camera={summary['camera_make']} {summary['camera_model']}, "
        f"date={summary['datetime_original']}, GPS={gps}"
    )
    return summary


# ── Strip EXIF ────────────────────────────────────────────────────────────────

def strip_exif(input_path: str | Path, output_path: str | Path) -> Path:
    """
    Save a clean copy of the image with all EXIF metadata removed.
    Useful for privacy-safe publishing.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(input_path) as img:
            # Re-encode via pixel data — drops all metadata
            clean = Image.new(img.mode, img.size)
            clean.putdata(list(img.getdata()))
            clean.save(output_path)
        logging.info(f"[EXIF] Stripped EXIF: saved clean copy to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"[EXIF] strip_exif failed for {input_path.name}: {e}")
        return input_path


if __name__ == "__main__":
    import json
    samples = list(Path("data/raw/exif_samples").glob("*.jpg")) + \
              list(Path("data/raw/exif_samples").glob("*.jpeg"))

    if not samples:
        print("No JPEG files found in data/raw/exif_samples/")
        print("Add camera photos (JPG) to that folder and re-run.")
    else:
        for photo in samples:
            summary = get_exif_summary(photo)
            print(f"\n=== {photo.name} ===")
            print(json.dumps(summary, indent=2, default=str))
