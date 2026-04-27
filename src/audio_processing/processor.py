"""
src/audio_processing/processor.py
Audio manipulation: trim, concatenate, volume, fade, convert.
All AudioSegment operations return new objects (immutable).
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from pydub import AudioSegment
from src.audio_processing.loader import load_audio, inspect_all_audio, SUPPORTED_FORMATS
from src.utils.logger import logging


OUTPUT_DIR = Path("data/processed/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Trim ──────────────────────────────────────────────────────────────────────
def trim_audio(audio: AudioSegment, start_ms: int, end_ms: int) -> AudioSegment:
    """Return a trimmed slice [start_ms, end_ms] of the audio."""
    trimmed = audio[start_ms:end_ms]
    logging.info(
        f"[AudioProcessor] Trimmed audio: {start_ms}ms–{end_ms}ms "
        f"({len(trimmed)/1000:.2f}s)"
    )
    return trimmed


# ── Concatenate ───────────────────────────────────────────────────────────────
def concatenate_audio(clips: list[AudioSegment]) -> AudioSegment:
    """Concatenate a list of AudioSegment objects into one."""
    combined = clips[0]
    for clip in clips[1:]:
        combined = combined + clip
    logging.info(
        f"[AudioProcessor] Concatenated {len(clips)} clips → "
        f"{len(combined)/1000:.2f}s total"
    )
    return combined


# ── Volume ────────────────────────────────────────────────────────────────────
def adjust_volume(audio: AudioSegment, db: float) -> AudioSegment:
    """Increase (+) or decrease (-) volume by db decibels."""
    adjusted = audio + db
    logging.info(f"[AudioProcessor] Volume adjusted by {db:+.1f} dB")
    return adjusted


# ── Fade ──────────────────────────────────────────────────────────────────────
def apply_fade(
    audio: AudioSegment,
    fade_in_ms: int = 1000,
    fade_out_ms: int = 1000,
) -> AudioSegment:
    """Apply fade-in and fade-out effects."""
    faded = audio.fade_in(fade_in_ms).fade_out(fade_out_ms)
    logging.info(
        f"[AudioProcessor] Fades applied: in={fade_in_ms}ms, out={fade_out_ms}ms"
    )
    return faded


# ── Export / Convert ──────────────────────────────────────────────────────────
def export_audio(
    audio: AudioSegment,
    output_path: str | Path,
    fmt: str | None = None,
    bitrate: str = "192k",
) -> Path:
    """
    Export an AudioSegment to a file.
    fmt is inferred from the extension if not given (e.g. 'mp3', 'wav', 'flac').
    bitrate only applies to lossy formats (MP3, OGG, AAC).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        fmt = output_path.suffix.lstrip(".").lower()

    params = {}
    if fmt in ("mp3", "ogg", "aac"):
        params["bitrate"] = bitrate

    audio.export(str(output_path), format=fmt, **params)
    size_kb = output_path.stat().st_size / 1024.0
    logging.info(
        f"[AudioProcessor] Exported → {output_path.name} "
        f"(format={fmt.upper()}, size={size_kb:.1f} KB)"
    )
    return output_path


def convert_audio(
    input_path: str | Path,
    output_format: str,
    bitrate: str = "192k",
) -> Path:
    """
    Load an audio file and re-export it in a different format.
    output_format: 'mp3', 'wav', 'flac', 'ogg'
    """
    input_path = Path(input_path)
    audio = load_audio(input_path)
    stem = input_path.stem
    output_path = OUTPUT_DIR / f"{stem}_converted.{output_format}"
    export_audio(audio, output_path, fmt=output_format, bitrate=bitrate)
    logging.info(
        f"[AudioProcessor] Converted {input_path.name} → {output_path.name}"
    )
    return output_path


# ── Demo / main ───────────────────────────────────────────────────────────────
def run_demo(audio_dir: str | Path = "data/raw/audio") -> None:
    """
    Demonstrate all processor operations on files found in audio_dir.
    Outputs are saved to data/processed/audio/.
    """
    audio_dir = Path(audio_dir)
    files = sorted([f for f in audio_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS])

    if not files:
        logging.warning("[AudioProcessor] No audio files found for demo.")
        return

    # ── Load first file ───────────────────────────────────────────────────────
    primary = load_audio(files[0])
    stem = files[0].stem

    # 1. Trim first 30 seconds
    trim_end = min(30_000, len(primary))
    trimmed = trim_audio(primary, 0, trim_end)
    export_audio(trimmed, OUTPUT_DIR / f"{stem}_trimmed.wav")

    # 2. Volume adjustment
    louder = adjust_volume(trimmed, +5)
    export_audio(louder, OUTPUT_DIR / f"{stem}_louder.wav")
    quieter = adjust_volume(trimmed, -5)
    export_audio(quieter, OUTPUT_DIR / f"{stem}_quieter.wav")

    # 3. Fade in/out
    faded = apply_fade(trimmed, fade_in_ms=2000, fade_out_ms=2000)
    export_audio(faded, OUTPUT_DIR / f"{stem}_faded.wav")

    # 4. Concatenate two clips (use second file if available, else duplicate)
    second = load_audio(files[1]) if len(files) > 1 else trimmed
    second_trim = trim_audio(second, 0, min(30_000, len(second)))
    concat = concatenate_audio([trimmed, second_trim])
    export_audio(concat, OUTPUT_DIR / f"{stem}_concat.wav")

    # 5. Convert to MP3 and FLAC
    convert_audio(files[0], "mp3")
    convert_audio(files[0], "flac")

    logging.info("[AudioProcessor] Demo complete. Check data/processed/audio/")


if __name__ == "__main__":
    run_demo()
