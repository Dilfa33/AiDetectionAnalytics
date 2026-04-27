"""
src/audio_processing/transcriber.py
Speech-to-text transcription using faster-whisper.
Supports short files, video-extracted audio, and long chunked transcription.
"""

import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
from pydub import AudioSegment
from src.utils.logger import logging

TRANSCRIPTS_DIR = Path("data/processed/transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

_model_cache: dict[str, WhisperModel] = {}


# ── Model management ──────────────────────────────────────────────────────────
def get_model(model_size: str = "base", device: str = "cpu") -> WhisperModel:
    """Load and cache a WhisperModel (avoids reloading on repeated calls)."""
    key = f"{model_size}_{device}"
    if key not in _model_cache:
        logging.info(f"[Transcriber] Loading Whisper model: {model_size} on {device}")
        _model_cache[key] = WhisperModel(model_size, device=device, compute_type="int8")
    return _model_cache[key]


# ── Core transcription ────────────────────────────────────────────────────────
def transcribe(
    audio_path: str | Path,
    model_size: str = "base",
    device: str = "cpu",
    language: str | None = None,
) -> dict:
    """
    Transcribe an audio file and return structured results.

    Returns:
        {
          "source_path": str,
          "language": str,
          "language_probability": float,
          "duration_sec": float,
          "segments": [{"start", "end", "text", "words": [...]}],
          "full_text": str,
          "model": str,
          "transcribed_at": str,
        }
    """
    audio_path = Path(audio_path)
    model = get_model(model_size, device)

    logging.info(f"[Transcriber] Transcribing {audio_path.name} (model={model_size})")

    kwargs = {"word_timestamps": True}
    if language:
        kwargs["language"] = language

    segments_gen, info = model.transcribe(str(audio_path), **kwargs)

    segments = []
    full_text_parts = []

    for seg in segments_gen:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                })
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
        })
        full_text_parts.append(seg.text.strip())

    result = {
        "source_path": str(audio_path),
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_sec": round(info.duration, 2),
        "segments": segments,
        "full_text": " ".join(full_text_parts),
        "model": model_size,
        "transcribed_at": datetime.utcnow().isoformat(),
    }

    logging.info(
        f"[Transcriber] Done: lang={info.language} "
        f"({info.language_probability:.0%}), "
        f"{len(segments)} segment(s), {info.duration:.1f}s"
    )
    return result


# ── Save helpers ──────────────────────────────────────────────────────────────
def save_json(result: dict, output_path: str | Path) -> Path:
    """Save transcription result as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info(f"[Transcriber] Saved JSON → {output_path}")
    return output_path


def save_txt(result: dict, output_path: str | Path) -> Path:
    """Save plain-text transcript."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["full_text"])
    logging.info(f"[Transcriber] Saved TXT → {output_path}")
    return output_path


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(result: dict, output_path: str | Path) -> Path:
    """Save SRT subtitle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    logging.info(f"[Transcriber] Saved SRT → {output_path}")
    return output_path


def save_all_formats(result: dict, stem: str, out_dir: str | Path = TRANSCRIPTS_DIR) -> dict:
    """Save transcript in JSON, TXT, and SRT formats. Returns paths dict."""
    out_dir = Path(out_dir)
    return {
        "json": save_json(result, out_dir / f"{stem}.json"),
        "txt": save_txt(result, out_dir / f"{stem}.txt"),
        "srt": save_srt(result, out_dir / f"{stem}.srt"),
    }


# ── Chunked transcription ─────────────────────────────────────────────────────
def chunked_transcribe(
    audio_path: str | Path,
    chunk_duration_min: float = 5.0,
    model_size: str = "base",
    device: str = "cpu",
    language: str | None = None,
    cache_chunks: bool = True,
) -> dict:
    """
    Transcribe a long audio file by splitting it into chunks.

    Each chunk result is optionally cached as JSON so interrupted jobs can resume.
    Timestamps are adjusted to reflect position in the original timeline.
    Returns a combined result in the same format as transcribe().
    """
    audio_path = Path(audio_path)
    chunk_ms = int(chunk_duration_min * 60 * 1000)

    logging.info(
        f"[Transcriber] Chunked transcription: {audio_path.name} "
        f"(chunk={chunk_duration_min}min, model={model_size})"
    )

    audio = AudioSegment.from_file(str(audio_path))
    total_ms = len(audio)
    num_chunks = (total_ms + chunk_ms - 1) // chunk_ms

    chunk_dir = TRANSCRIPTS_DIR / "chunks" / audio_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)

    all_segments: list[dict] = []
    detected_language = None
    detected_lang_prob = 0.0
    total_duration = total_ms / 1000.0

    for i in range(num_chunks):
        start_ms = i * chunk_ms
        end_ms = min(start_ms + chunk_ms, total_ms)
        offset_sec = start_ms / 1000.0

        chunk_json = chunk_dir / f"chunk_{i:04d}.json"

        # Resume: load cached chunk if it exists
        if cache_chunks and chunk_json.exists():
            with open(chunk_json, encoding="utf-8") as f:
                chunk_result = json.load(f)
            logging.info(f"[Transcriber] Loaded cached chunk {i+1}/{num_chunks}")
        else:
            # Export chunk to a temporary WAV file
            chunk_audio = audio[start_ms:end_ms]
            chunk_wav = chunk_dir / f"chunk_{i:04d}.wav"
            chunk_audio.export(str(chunk_wav), format="wav")

            chunk_result = transcribe(chunk_wav, model_size=model_size, device=device, language=language)

            if cache_chunks:
                save_json(chunk_result, chunk_json)

        # Adjust timestamps by chunk offset
        for seg in chunk_result["segments"]:
            adjusted_seg = {
                **seg,
                "start": round(seg["start"] + offset_sec, 3),
                "end": round(seg["end"] + offset_sec, 3),
            }
            if seg.get("words"):
                adjusted_seg["words"] = [
                    {**w, "start": round(w["start"] + offset_sec, 3), "end": round(w["end"] + offset_sec, 3)}
                    for w in seg["words"]
                ]
            all_segments.append(adjusted_seg)

        if detected_language is None:
            detected_language = chunk_result["language"]
            detected_lang_prob = chunk_result["language_probability"]

        logging.info(f"[Transcriber] Chunk {i+1}/{num_chunks} processed")

    combined = {
        "source_path": str(audio_path),
        "language": detected_language,
        "language_probability": detected_lang_prob,
        "duration_sec": round(total_duration, 2),
        "segments": all_segments,
        "full_text": " ".join(s["text"] for s in all_segments),
        "model": model_size,
        "transcribed_at": datetime.utcnow().isoformat(),
        "chunked": True,
        "num_chunks": num_chunks,
    }

    # Save combined result
    combined_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_chunked.json"
    save_json(combined, combined_path)
    logging.info(f"[Transcriber] Chunked transcription complete → {combined_path}")
    return combined
