"""
src/video_processing/loader.py
Load video files, inspect properties, and extract audio tracks.
Uses moviepy. Always closes VideoFileClip to prevent file handle leaks.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from moviepy import VideoFileClip
from src.utils.logger import logging

PROCESSED_AUDIO_DIR = Path("data/processed/audio")
PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def load_video(file_path: str | Path) -> VideoFileClip:
    """Load a video file and return a VideoFileClip. Caller must call .close()."""
    file_path = Path(file_path)
    clip = VideoFileClip(str(file_path))
    logging.info(f"[VideoLoader] Loaded {file_path.name} ({clip.duration:.2f}s)")
    return clip


def inspect_video(file_path: str | Path) -> dict:
    """Return a dict of video properties."""
    file_path = Path(file_path)
    clip = None
    try:
        clip = load_video(file_path)
        w, h = clip.size
        props = {
            "filename": file_path.name,
            "duration_sec": round(clip.duration, 2),
            "fps": clip.fps,
            "resolution": f"{w}x{h}",
            "width_px": w,
            "height_px": h,
            "has_audio": clip.audio is not None,
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        }
        logging.info(
            f"[VideoLoader] {file_path.name}: "
            f"{props['duration_sec']}s, {props['fps']}fps, {props['resolution']}"
        )
        return props
    finally:
        if clip:
            clip.close()


def extract_audio(
    file_path: str | Path,
    output_path: str | Path | None = None,
    bitrate: str = "192k",
) -> Path:
    """
    Extract the audio track from a video file and save it as an MP3.
    Returns the path to the extracted MP3.
    """
    file_path = Path(file_path)
    if output_path is None:
        output_path = PROCESSED_AUDIO_DIR / f"{file_path.stem}_audio.mp3"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip = None
    try:
        clip = load_video(file_path)
        if clip.audio is None:
            raise ValueError(f"Video file has no audio track: {file_path.name}")
        clip.audio.write_audiofile(str(output_path), bitrate=bitrate, logger=None)
        size_kb = output_path.stat().st_size / 1024.0
        logging.info(
            f"[VideoLoader] Audio extracted → {output_path.name} ({size_kb:.1f} KB)"
        )
        return output_path
    finally:
        if clip:
            clip.close()


def inspect_all_videos(video_dir: str | Path) -> list[dict]:
    """Inspect every supported video file in a directory."""
    video_dir = Path(video_dir)
    results = []
    files = [f for f in video_dir.iterdir() if f.suffix.lower() in SUPPORTED_VIDEO_FORMATS]
    if not files:
        logging.warning(f"[VideoLoader] No supported video files in {video_dir}")
        return results

    for f in sorted(files):
        try:
            props = inspect_video(f)
            print(f"\n  filename     : {props['filename']}")
            print(f"  duration_sec : {props['duration_sec']}")
            print(f"  fps          : {props['fps']}")
            print(f"  resolution   : {props['resolution']}")
            print(f"  has_audio    : {props['has_audio']}")
            print(f"  file_size_mb : {props['file_size_mb']}")
            results.append(props)
        except Exception as e:
            logging.error(f"[VideoLoader] Failed to inspect {f.name}: {e}")

    return results


if __name__ == "__main__":
    import sys
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/video")
    props_list = inspect_all_videos(folder)

    # Also extract audio from each video
    for props in props_list:
        video_path = folder / props["filename"]
        if props["has_audio"]:
            extract_audio(video_path)
