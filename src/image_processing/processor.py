"""
src/image_processing/processor.py
Image processing functions: inspect, resize, thumbnail, crop,
format conversion, filters, and enhancements using Pillow.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from src.utils.logger import logging

RESIZED_DIR    = Path("data/processed/resized")
THUMBNAIL_DIR  = Path("data/processed/thumbnails")
WEBP_DIR       = Path("data/processed/webp")
CROPPED_DIR    = Path("data/processed/cropped")


# ── Inspect ───────────────────────────────────────────────────────────────────

def inspect_image(path: str | Path) -> dict:
    """Return a dict of core image properties without loading pixel data."""
    path = Path(path)
    try:
        with Image.open(path) as img:
            info = {
                "filename":        path.name,
                "format":          img.format,
                "mode":            img.mode,
                "width":           img.size[0],
                "height":          img.size[1],
                "aspect_ratio":    round(img.size[0] / img.size[1], 3),
                "file_size_bytes": path.stat().st_size,
                "file_size_kb":    round(path.stat().st_size / 1024, 1),
            }
        logging.info(
            f"[Processor] Inspected {path.name}: "
            f"{info['width']}x{info['height']} {info['format']} {info['mode']} "
            f"({info['file_size_kb']} KB)"
        )
        return info
    except Exception as e:
        logging.error(f"[Processor] inspect_image failed for {path}: {e}")
        return {}


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_image(img: Image.Image, width: int, height: int,
                 resample=Image.LANCZOS) -> Image.Image:
    """Resize to exact dimensions. Does not preserve aspect ratio."""
    return img.resize((width, height), resample)


def resize_proportional(img: Image.Image, max_size: int = 500,
                         resample=Image.LANCZOS) -> Image.Image:
    """
    Resize so the longest side equals max_size, preserving aspect ratio.
    Saves result to data/processed/resized/.
    """
    w, h  = img.size
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), resample)


def save_resized(img: Image.Image, filename: str, max_size: int = 500) -> Path:
    """Resize proportionally and save to resized folder."""
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    resized = resize_proportional(img, max_size)
    stem    = Path(filename).stem
    dest    = RESIZED_DIR / f"{stem}_resized.jpg"

    # Ensure RGB before saving as JPEG
    out = resized.convert("RGB") if resized.mode == "RGBA" else resized
    out.save(dest, "JPEG", quality=85)
    logging.info(f"[Processor] Saved resized: {dest}")
    return dest


# ── Thumbnails ────────────────────────────────────────────────────────────────

def generate_thumbnail(img: Image.Image, max_size: tuple = (150, 150)) -> Image.Image:
    """
    Generate a thumbnail preserving aspect ratio using Pillow's thumbnail().
    Works on a copy so the original is unchanged.
    """
    copy = img.copy()
    copy.thumbnail(max_size, Image.LANCZOS)
    return copy


def generate_fixed_thumbnail(img: Image.Image, size: tuple = (150, 150),
                              method: str = "contain") -> Image.Image:
    """
    Generate a fixed-size thumbnail using ImageOps helpers.
    method: 'contain' | 'cover' | 'fit' | 'pad'
    """
    ops = {
        "contain": lambda: ImageOps.contain(img, size, Image.LANCZOS),
        "cover":   lambda: ImageOps.cover(img, size, Image.LANCZOS),
        "fit":     lambda: ImageOps.fit(img, size, Image.LANCZOS),
        "pad":     lambda: ImageOps.pad(img, size, Image.LANCZOS, color=(0, 0, 0)),
    }
    if method not in ops:
        logging.warning(f"[Processor] Unknown thumbnail method '{method}', using 'contain'")
        method = "contain"
    return ops[method]()


def save_thumbnail(img: Image.Image, filename: str,
                   size: tuple = (150, 150)) -> Path:
    """Generate and save a fixed thumbnail."""
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumb = generate_fixed_thumbnail(img, size, method="contain")
    stem  = Path(filename).stem
    dest  = THUMBNAIL_DIR / f"{stem}_thumb.jpg"
    out   = thumb.convert("RGB") if thumb.mode == "RGBA" else thumb
    out.save(dest, "JPEG", quality=85)
    logging.info(f"[Processor] Saved thumbnail: {dest}")
    return dest


# ── Crop ──────────────────────────────────────────────────────────────────────

def crop_image(img: Image.Image, box: tuple) -> Image.Image:
    """
    Crop using a (left, upper, right, lower) bounding box.
    (0, 0) is top-left corner.
    """
    return img.crop(box)


def crop_banner(img: Image.Image) -> Image.Image:
    """Crop the top half of the image — typical banner crop for movie posters."""
    w, h = img.size
    return crop_image(img, (0, 0, w, h // 2))


def save_cropped(img: Image.Image, filename: str) -> Path:
    """Banner-crop and save."""
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    cropped = crop_banner(img)
    stem    = Path(filename).stem
    dest    = CROPPED_DIR / f"{stem}_banner.jpg"
    out     = cropped.convert("RGB") if cropped.mode == "RGBA" else cropped
    out.save(dest, "JPEG", quality=85)
    logging.info(f"[Processor] Saved cropped: {dest}")
    return dest


# ── Format conversion ─────────────────────────────────────────────────────────

def convert_to_webp(img: Image.Image, filename: str, quality: int = 80) -> Path:
    """Convert image to WebP and save to data/processed/webp/."""
    WEBP_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    dest = WEBP_DIR / f"{stem}.webp"
    out  = img.convert("RGB") if img.mode == "RGBA" else img
    out.save(dest, "WEBP", quality=quality)
    logging.info(f"[Processor] Saved WebP: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def convert_to_grayscale(img: Image.Image) -> Image.Image:
    """Convert image to grayscale (mode 'L')."""
    return img.convert("L")


def save_optimised_jpeg(img: Image.Image, output_path: Path,
                         quality: int = 75) -> Path:
    """Save as optimised progressive JPEG."""
    output_path = Path(output_path)
    out = img.convert("RGB") if img.mode == "RGBA" else img
    out.save(output_path, "JPEG", quality=quality, optimize=True, progressive=True)
    logging.info(f"[Processor] Saved optimised JPEG: {output_path}")
    return output_path


# ── Filters & Enhancements ────────────────────────────────────────────────────

FILTERS = {
    "blur":        ImageFilter.BLUR,
    "gaussian":    ImageFilter.GaussianBlur(radius=3),
    "sharpen":     ImageFilter.SHARPEN,
    "find_edges":  ImageFilter.FIND_EDGES,
    "contour":     ImageFilter.CONTOUR,
    "emboss":      ImageFilter.EMBOSS,
    "median":      ImageFilter.MedianFilter(size=3),
}


def apply_filter(img: Image.Image, filter_name: str) -> Image.Image:
    """
    Apply a named filter. Available: blur, gaussian, sharpen,
    find_edges, contour, emboss, median.
    """
    f = FILTERS.get(filter_name.lower())
    if f is None:
        logging.warning(f"[Processor] Unknown filter '{filter_name}'")
        return img
    result = img.filter(f)
    logging.info(f"[Processor] Applied filter: {filter_name}")
    return result


def apply_enhancement(img: Image.Image, enhancer_name: str,
                       factor: float = 1.5) -> Image.Image:
    """
    Apply an enhancement. enhancer_name: brightness | contrast | color | sharpness.
    factor 1.0 = original, <1 reduces, >1 increases.
    """
    enhancers = {
        "brightness": ImageEnhance.Brightness,
        "contrast":   ImageEnhance.Contrast,
        "color":      ImageEnhance.Color,
        "sharpness":  ImageEnhance.Sharpness,
    }
    cls = enhancers.get(enhancer_name.lower())
    if cls is None:
        logging.warning(f"[Processor] Unknown enhancer '{enhancer_name}'")
        return img
    result = cls(img).enhance(factor)
    logging.info(f"[Processor] Applied enhancement: {enhancer_name} x{factor}")
    return result


if __name__ == "__main__":
    import sys
    test_images = list(Path("data/raw/images").glob("*.jpg"))
    if not test_images:
        print("No images in data/raw/images/ — run downloader.py first")
        sys.exit(0)

    path = test_images[0]
    print(f"Testing with: {path.name}")
    print(inspect_image(path))

    with Image.open(path) as img:
        save_resized(img, path.name)
        save_thumbnail(img, path.name)
        save_cropped(img, path.name)
        convert_to_webp(img, path.name)
