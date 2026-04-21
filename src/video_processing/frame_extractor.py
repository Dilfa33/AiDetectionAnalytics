"""
src/video_processing/frame_extractor.py
Extract keyframes from video files at regular intervals.
Memory warning: 1 min @ 30fps = 1,800 frames. Only extract what you need.
"""

import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from moviepy import VideoFileClip
import cv2
from src.utils.logger import logging

FRAMES_DIR = Path("data/processed/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def save_frame(
    clip: VideoFileClip,
    t_seconds: float,
    output_path: str | Path,
) -> Path:
    """
    Save a single frame from the video at timestamp t_seconds as a PNG image.
    clip must already be open; this function does NOT close it.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = clip.get_frame(t_seconds)          # numpy array (H, W, 3) RGB
    # cv2 expects BGR
    import cv2 as _cv2
    bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
    _cv2.imwrite(str(output_path), bgr)
    return output_path


def extract_keyframes(
    video_path: str | Path,
    interval_seconds: float = 10.0,
    output_dir: str | Path | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    """
    Extract one frame every interval_seconds seconds from the video.

    Args:
        video_path: Path to the video file.
        interval_seconds: Time between extracted frames (default 10s).
        output_dir: Directory to save frames. Defaults to data/processed/frames/<stem>/.
        max_frames: Cap on number of frames to extract (avoids memory issues).

    Returns:
        List of dicts with keys: timestamp_sec, frame_index, path.
        Also writes a JSON index file alongside the frames.
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = FRAMES_DIR / video_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"[FrameExtractor] Extracting keyframes from {video_path.name} "
        f"(interval={interval_seconds}s)"
    )

    clip = None
    try:
        clip = VideoFileClip(str(video_path))
        duration = clip.duration

        timestamps = []
        t = 0.0
        while t <= duration:
            timestamps.append(round(t, 3))
            t += interval_seconds

        if max_frames and len(timestamps) > max_frames:
            timestamps = timestamps[:max_frames]
            logging.warning(
                f"[FrameExtractor] Capped to {max_frames} frames "
                f"(would have been {len(timestamps)} total)"
            )

        frame_records = []
        for idx, ts in enumerate(timestamps):
            fname = f"frame_{idx:04d}_{ts:.1f}s.png"
            out_path = output_dir / fname
            try:
                save_frame(clip, ts, out_path)
                size_kb = out_path.stat().st_size / 1024.0
                frame_records.append({
                    "timestamp_sec": ts,
                    "frame_index": idx,
                    "path": str(out_path),
                    "size_kb": round(size_kb, 1),
                })
                logging.info(f"[FrameExtractor] Frame {idx+1}/{len(timestamps)}: t={ts}s → {fname}")
            except Exception as e:
                logging.error(f"[FrameExtractor] Failed to save frame at t={ts}s: {e}")

        # Write JSON index
        index_path = output_dir / "frames_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": str(video_path),
                    "interval_seconds": interval_seconds,
                    "total_frames": len(frame_records),
                    "frames": frame_records,
                },
                f,
                indent=2,
            )
        logging.info(
            f"[FrameExtractor] {len(frame_records)} frames saved to {output_dir}. "
            f"Index: {index_path}"
        )
        return frame_records

    finally:
        if clip:
            clip.close()


if __name__ == "__main__":
    import sys
    video_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/video")
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    from src.video_processing.loader import SUPPORTED_VIDEO_FORMATS
    for video_file in sorted(video_dir.iterdir()):
        if video_file.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
            records = extract_keyframes(video_file, interval_seconds=interval)
            print(f"\n{video_file.name}: {len(records)} keyframes extracted")
