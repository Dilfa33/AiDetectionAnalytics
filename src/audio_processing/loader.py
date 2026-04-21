"""
src/audio_processing/loader.py
Load audio files in various formats and inspect their properties.
Supports: WAV, MP3, FLAC, OGG, AAC
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from pydub import AudioSegment
from src.utils.logger import logging


SUPPORTED_FORMATS = {
    ".wav": "WAV",
    ".mp3": "MP3",
    ".flac": "FLAC",
    ".ogg": "OGG",
    ".aac": "AAC",
    ".m4a": "AAC",
}


def load_audio(file_path: str | Path) -> AudioSegment:
    """Load an audio file into a pydub AudioSegment."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {ext}")

    fmt = SUPPORTED_FORMATS[ext]
    audio = AudioSegment.from_file(str(file_path), format=fmt.lower())
    logging.info(f"[AudioLoader] Loaded {file_path.name} ({fmt}, {len(audio)/1000:.2f}s)")
    return audio


def inspect_audio(file_path: str | Path) -> dict:
    """Return a dict of technical properties for an audio file."""
    file_path = Path(file_path)
    audio = load_audio(file_path)

    ext = file_path.suffix.lower()
    fmt = SUPPORTED_FORMATS.get(ext, ext.upper().lstrip("."))
    channels = audio.channels
    channel_type = "Mono" if channels == 1 else "Stereo"
    duration_sec = len(audio) / 1000.0
    bit_depth = audio.sample_width * 8
    file_size_kb = file_path.stat().st_size / 1024.0

    props = {
        "filename": file_path.name,
        "format": fmt,
        "duration_sec": round(duration_sec, 2),
        "channels": channels,
        "channel_type": channel_type,
        "frame_rate_hz": audio.frame_rate,
        "bit_depth": bit_depth,
        "file_size_kb": round(file_size_kb, 1),
    }
    return props


def print_audio_info(props: dict) -> None:
    """Pretty-print audio properties."""
    for key, value in props.items():
        print(f"  {key:<22}: {value}")


def inspect_all_audio(audio_dir: str | Path) -> list[dict]:
    """Load and inspect every supported audio file in a directory."""
    audio_dir = Path(audio_dir)
    results = []

    files = [f for f in audio_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]
    if not files:
        logging.warning(f"[AudioLoader] No supported audio files found in {audio_dir}")
        return results

    for f in sorted(files):
        try:
            props = inspect_audio(f)
            print_audio_info(props)
            print()
            results.append(props)
        except Exception as e:
            logging.error(f"[AudioLoader] Failed to inspect {f.name}: {e}")

    return results


if __name__ == "__main__":
    import sys
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/audio")
    inspect_all_audio(folder)
