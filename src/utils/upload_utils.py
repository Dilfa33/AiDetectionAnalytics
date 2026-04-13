"""
src/utils/upload_utils.py
Google Drive upload utilities using google-api-python-client.

Setup:
  1. Create OAuth 2.0 credentials in Google Cloud Console
  2. Download the JSON and save to credentials/google_credentials.json
  3. Set GOOGLE_DRIVE_FOLDER_ID in your .env file
  4. First run will open a browser for OAuth consent
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import logging

load_dotenv()

CREDENTIALS_PATH = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials/google_credentials.json"))
FOLDER_ID        = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
SCOPES           = [os.getenv("GOOGLE_SCOPES", "https://www.googleapis.com/auth/drive.file")]
TOKEN_PATH       = Path("credentials/token.json")

MIME_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".pdf":  "application/pdf",
}


# ── Authentication ────────────────────────────────────────────────────────────

def get_drive_service():
    """
    Authenticate with Google Drive using OAuth 2.0.
    Caches the token in credentials/token.json after first login.
    Returns a Google Drive API service object.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        logging.error(
            "[Drive] Google API packages not installed. Run: "
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        return None

    creds = None

    # Load cached token
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    # Refresh or re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                logging.error(f"[Drive] Credentials file not found: {CREDENTIALS_PATH}")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json())
        logging.info("[Drive] Token saved to credentials/token.json")

    service = build("drive", "v3", credentials=creds)
    logging.info("[Drive] Authenticated with Google Drive")
    return service


# ── Upload single file ────────────────────────────────────────────────────────

def upload_image(service, file_path: Path, folder_id: str = None) -> str | None:
    """
    Upload a single image to Google Drive.
    Returns the file ID on success, None on failure.
    """
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        logging.error("[Drive] googleapiclient not installed")
        return None

    file_path = Path(file_path)
    folder_id = folder_id or FOLDER_ID

    if not file_path.exists():
        logging.warning(f"[Drive] File not found: {file_path}")
        return None

    mime = MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")

    metadata = {"name": file_path.name}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaFileUpload(str(file_path), mimetype=mime, resumable=True)

    try:
        f = service.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = f.get("id")
        logging.info(f"[Drive] Uploaded: {file_path.name} (id={file_id})")
        return file_id
    except Exception as e:
        logging.error(f"[Drive] Upload failed for {file_path.name}: {e}")
        return None


# ── Batch upload ──────────────────────────────────────────────────────────────

def upload_batch(service, file_paths: list, folder_id: str = None) -> list[dict]:
    """
    Upload a list of files to Google Drive.
    Returns a list of {file, drive_id} dicts.
    """
    if service is None:
        logging.error("[Drive] No Drive service — skipping upload")
        return []

    results = []
    for path in file_paths:
        path     = Path(path)
        drive_id = upload_image(service, path, folder_id)
        results.append({"file": path.name, "drive_id": drive_id})

    uploaded = sum(1 for r in results if r["drive_id"])
    logging.info(f"[Drive] Batch complete: {uploaded}/{len(file_paths)} files uploaded")
    return results
