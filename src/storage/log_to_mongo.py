"""
src/storage/log_to_mongo.py
Parses pipeline.log and stores audio/video log entries in MongoDB.
"""

import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from datetime import datetime
from src.storage.mongo import pipeline_logs_collection
from src.utils.logger import logging

LOG_FILE = Path("logs/pipeline.log")

# Tags that belong to audio or video processing
AUDIO_TAGS = {"[AudioLoader]", "[AudioProcessor]"}
VIDEO_TAGS = {"[VideoLoader]", "[FrameExtractor]", "[VideoStage]"}

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r" - (?P<level>\w+)"
    r" - (?P<message>.+)$"
)


def _parse_log(line: str) -> dict | None:
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None

    message = match.group("message")

    tag = next((t for t in AUDIO_TAGS | VIDEO_TAGS if message.startswith(t)), None)
    if not tag:
        return None

    stage = "audio" if tag in AUDIO_TAGS else "video"

    return {
        "timestamp": match.group("timestamp"),
        "level": match.group("level"),
        "stage": stage,
        "tag": tag,
        "message": message,
        "stored_at": datetime.utcnow().isoformat(),
    }


def store_logs(log_file: Path = LOG_FILE) -> int:
    if not log_file.exists():
        logging.error(f"[LogToMongo] Log file not found: {log_file}")
        return 0

    entries = []
    with open(log_file, "r") as f:
        for line in f:
            entry = _parse_log(line)
            if entry:
                entries.append(entry)

    if not entries:
        logging.info("[LogToMongo] No audio/video log entries found.")
        return 0

    try:
        result = pipeline_logs_collection.insert_many(entries)
        logging.info(f"[LogToMongo] Stored {len(result.inserted_ids)} log entries to MongoDB.")
        return len(result.inserted_ids)
    except Exception as e:
        logging.error(f"[LogToMongo] Insert failed: {e}")
        return 0


if __name__ == "__main__":
    count = store_logs()
    print(f"Stored {count} audio/video log entries to MongoDB.")
