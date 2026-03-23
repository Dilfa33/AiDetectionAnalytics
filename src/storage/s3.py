"""
src/storage/s3.py
Uploads raw JSON files to S3 (MinIO via Docker).
Run: python s3.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import boto3
from botocore.exceptions import NoCredentialsError, EndpointResolutionError
from pathlib import Path
from src.utils.logger import logging

# ── Config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT = "http://localhost:9000"       # MinIO via Docker
S3_BUCKET_NAME = "arxiv-pipeline-bucket"
AWS_ACCESS_KEY = "minioadmin"
AWS_SECRET_KEY = "minioadmin"


# ── Client ────────────────────────────────────────────────────────────────────
def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name="us-east-1"
    )


# ── Upload ────────────────────────────────────────────────────────────────────
def upload_file_to_s3(file_path, file_name):
    """Upload a single file to the S3 bucket."""
    s3 = create_s3_client()
    try:
        logging.info(f"Started uploading {file_name} to S3 bucket {S3_BUCKET_NAME}")
        s3.upload_file(file_path, S3_BUCKET_NAME, file_name)
        logging.info(f"Successfully uploaded {file_name} to {S3_BUCKET_NAME}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except NoCredentialsError:
        logging.error("Credentials not available")
    except Exception as e:
        logging.error(f"Upload failed for {file_name}: {e}")


def upload_all_raw_files():
    """Upload all JSON files from data/raw/api/ to S3."""
    api_dir = Path("data/raw/api")
    files = list(api_dir.glob("*.json")) if api_dir.exists() else []

    if not files:
        logging.warning("No files found in data/raw/api/ to upload")
        return

    for file in files:
        upload_file_to_s3(str(file), file.name)

    logging.info(f"Uploaded {len(files)} files to S3")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    file_path = "data/raw/api/papers_page_1.json"
    file_name = "papers_page_1.json"
    upload_file_to_s3(file_path, file_name)
