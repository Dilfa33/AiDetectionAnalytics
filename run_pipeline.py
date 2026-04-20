"""
run_pipeline.py
Audio/video pipeline:
  1. Load and process audio files (trim, convert, volume, fade)
  2. Load video files, extract audio tracks and keyframes
  3. Transcribe audio/video using faster-whisper -> MongoDB
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pathlib import Path
from src.utils.logger import logging
from src.storage.mongo import save_transcript_to_mongo
from src.audio_processing.loader import inspect_all_audio, SUPPORTED_FORMATS as AUDIO_FORMATS
from src.audio_processing.processor import run_demo as run_audio_demo
from src.audio_processing.transcriber import transcribe, chunked_transcribe, save_all_formats
from src.video_processing.loader import inspect_all_videos, extract_audio
from src.video_processing.frame_extractor import extract_keyframes


def run_audio_stage(audio_dir: Path = Path("data/raw/audio")) -> list[dict]:
    logging.info("=" * 60)
    logging.info("STAGE 1: Audio processing (inspect, trim, convert, fade)")
    logging.info("=" * 60)
    props = inspect_all_audio(audio_dir)
    if props:
        run_audio_demo(audio_dir)
    logging.info(f"Audio stage complete: {len(props)} file(s) inspected")
    return props


def run_video_stage(
    video_dir: Path = Path("data/raw/video"),
    frame_interval: float = 10.0,
) -> dict:
    logging.info("=" * 60)
    logging.info("STAGE 2: Video processing (inspect, extract audio, keyframes)")
    logging.info("=" * 60)
    video_props = inspect_all_videos(video_dir)
    extracted_audio: list[Path] = []
    total_frames = 0

    for props in video_props:
        video_path = video_dir / props["filename"]
        if props["has_audio"]:
            try:
                audio_path = extract_audio(video_path)
                extracted_audio.append(audio_path)
            except Exception as e:
                logging.error(f"[VideoStage] Audio extraction failed for {props['filename']}: {e}")

        try:
            records = extract_keyframes(video_path, interval_seconds=frame_interval)
            total_frames += len(records)
        except Exception as e:
            logging.error(f"[VideoStage] Keyframe extraction failed for {props['filename']}: {e}")

    logging.info(
        f"Video stage complete: {len(video_props)} video(s), "
        f"{len(extracted_audio)} audio track(s), {total_frames} frame(s)"
    )
    return {
        "video_props": video_props,
        "extracted_audio": extracted_audio,
        "total_frames": total_frames,
    }


def run_transcription_stage(
    audio_dir: Path = Path("data/raw/audio"),
    extracted_audio: list[Path] | None = None,
    model_size: str = "base",
    long_threshold_sec: float = 300.0,
) -> list[dict]:
    """
    Transcribe audio files from data/raw/audio/ and any video-extracted audio.
    Files longer than long_threshold_sec use chunked transcription.
    Results are stored in MongoDB.
    """
    logging.info("=" * 60)
    logging.info("STAGE 3: Speech-to-text transcription")
    logging.info("=" * 60)

    files: list[Path] = sorted(
        [f for f in audio_dir.iterdir() if f.suffix.lower() in AUDIO_FORMATS]
    )
    if extracted_audio:
        files += [p for p in extracted_audio if p.exists()]

    transcripts = []
    for audio_file in files:
        try:
            from pydub import AudioSegment
            duration_ms = len(AudioSegment.from_file(str(audio_file)))
            duration_sec = duration_ms / 1000.0

            if duration_sec > long_threshold_sec:
                logging.info(
                    f"[TranscriptionStage] Long file ({duration_sec:.0f}s), "
                    f"using chunked transcription: {audio_file.name}"
                )
                result = chunked_transcribe(
                    audio_file,
                    chunk_duration_min=5.0,
                    model_size=model_size,
                    cache_chunks=True,
                )
            else:
                result = transcribe(audio_file, model_size=model_size)

            save_all_formats(result, audio_file.stem)
            save_transcript_to_mongo(result)
            transcripts.append(result)
            logging.info(
                f"[TranscriptionStage] {audio_file.name}: "
                f"lang={result['language']}, {len(result['segments'])} segment(s)"
            )
        except Exception as e:
            logging.error(f"[TranscriptionStage] Failed for {audio_file.name}: {e}")

    logging.info(f"Transcription stage complete: {len(transcripts)} file(s) transcribed")
    return transcripts


if __name__ == "__main__":
    logging.info("Pipeline started")

    audio_results  = run_audio_stage()
    video_results  = run_video_stage()
    transcript_results = run_transcription_stage(
        extracted_audio=video_results["extracted_audio"]
    )

    logging.info("=" * 60)
    logging.info("Pipeline summary:")
    logging.info(f"  Audio files:        {len(audio_results)}")
    logging.info(f"  Videos processed:   {len(video_results['video_props'])}")
    logging.info(f"  Keyframes saved:    {video_results['total_frames']}")
    logging.info(f"  Transcripts:        {len(transcript_results)}")
    logging.info("Pipeline finished")
