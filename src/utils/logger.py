import logging
from pathlib import Path

# Absolute path so logs always land in the right place regardless of cwd
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
