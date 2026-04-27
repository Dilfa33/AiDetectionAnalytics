"""
src/audio_processing/test_transcribe.py
Test transcription on short audio, video-extracted audio, and long chunked audio.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from src.audio_processing.transcriber import (
    transcribe,
    chunked_transcribe,
    save_all_formats,
)
from src.audio_processing.loader import SUPPORTED_FORMATS
from src.utils.logger import logging

AUDIO_DIR = Path("data/raw/audio")
VIDEO_AUDIO_DIR = Path("data/processed/audio")
TRANSCRIPTS_DIR = Path("data/processed/transcripts")


def test_short_audio(model_size: str = "base") -> None:
    """Transcribe each file in data/raw/audio and save all formats."""
    files = sorted([f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS])
    if not files:
        logging.warning("[TestTranscribe] No audio files found in data/raw/audio/")
        return

    for audio_file in files:
        logging.info(f"\n{'='*60}")
        logging.info(f"[TestTranscribe] Transcribing: {audio_file.name}")

        result = transcribe(audio_file, model_size=model_size)

        # Print summary
        print(f"\n  File          : {audio_file.name}")
        print(f"  Language      : {result['language']} ({result['language_probability']:.0%})")
        print(f"  Duration (s)  : {result['duration_sec']}")
        print(f"  Segments      : {len(result['segments'])}")
        print(f"  Text preview  : {result['full_text'][:200]}...")

        # Print word-level confidence for first 10 words
        all_words = [w for seg in result["segments"] for w in seg.get("words", [])]
        if all_words:
            print("\n  Word-level confidence (first 10):")
            for w in all_words[:10]:
                print(f"    {w['start']:.2f}s–{w['end']:.2f}s  {w['word']!r:<20}  p={w['probability']:.3f}")

        # Save outputs
        paths = save_all_formats(result, audio_file.stem)
        print(f"\n  Saved JSON : {paths['json']}")
        print(f"  Saved TXT  : {paths['txt']}")
        print(f"  Saved SRT  : {paths['srt']}")


def test_video_audio(model_size: str = "base") -> None:
    """
    Transcribe audio that was extracted from video files.
    Looks for *_audio.mp3 files in data/processed/audio/.
    """
    audio_files = sorted(VIDEO_AUDIO_DIR.glob("*_audio.mp3"))
    if not audio_files:
        logging.warning(
            "[TestTranscribe] No extracted video audio found in data/processed/audio/. "
            "Run video loader first."
        )
        return

    for audio_file in audio_files:
        logging.info(f"[TestTranscribe] Transcribing video audio: {audio_file.name}")
        result = transcribe(audio_file, model_size=model_size)

        print(f"\n  File          : {audio_file.name}")
        print(f"  Language      : {result['language']} ({result['language_probability']:.0%})")
        print(f"  Duration (s)  : {result['duration_sec']}")
        print(f"  Segments      : {len(result['segments'])}")
        print(f"  Text preview  : {result['full_text'][:200]}...")

        paths = save_all_formats(result, audio_file.stem)
        print(f"  Saved → {paths['json']}")


def test_chunked_audio(model_size: str = "base", chunk_min: float = 5.0) -> None:
    """
    Transcribe the first audio file in data/raw/audio/ using chunked strategy.
    """
    files = sorted([f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS])
    if not files:
        logging.warning("[TestTranscribe] No audio files found for chunked test.")
        return

    audio_file = files[0]
    logging.info(f"[TestTranscribe] Chunked transcription: {audio_file.name}")

    result = chunked_transcribe(
        audio_file,
        chunk_duration_min=chunk_min,
        model_size=model_size,
        cache_chunks=True,
    )

    print(f"\n  File          : {audio_file.name}")
    print(f"  Language      : {result['language']}")
    print(f"  Duration (s)  : {result['duration_sec']}")
    print(f"  Chunks        : {result['num_chunks']}")
    print(f"  Segments      : {len(result['segments'])}")
    print(f"  Text preview  : {result['full_text'][:300]}...")

    paths = save_all_formats(result, f"{audio_file.stem}_chunked")
    print(f"  Saved → {paths['json']}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST 1: Short audio transcription")
    print("="*60)
    test_short_audio()

    print("\n" + "="*60)
    print("TEST 2: Video-extracted audio transcription")
    print("="*60)
    test_video_audio()

    print("\n" + "="*60)
    print("TEST 3: Chunked long audio transcription")
    print("="*60)
    test_chunked_audio()
