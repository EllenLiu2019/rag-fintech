import os
import re
import json
import uuid
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from common import get_logger, file_utils
from common.config_utils import get_base_config

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Local storage dirs (used when provider == "local")
# ---------------------------------------------------------------------------
ORIGINAL_DIR = Path(file_utils.get_project_root_dir("repository", "s3", "original"))
PARSED_DIR = Path(file_utils.get_project_root_dir("repository", "s3", "parsed"))

# ---------------------------------------------------------------------------
# Storage config
# ---------------------------------------------------------------------------
_storage_config: dict = get_base_config("storage", {}) or {}
_PROVIDER: str = _storage_config.get("provider", "local")  # "local" | "oss"
_BUCKET: str = _storage_config.get("bucket", "")
_PREFIX: str = _storage_config.get("prefix", "").strip("/")

_s3 = None

if _PROVIDER in ("oss"):
    import boto3
    from botocore.config import Config as BotoConfig

    _endpoint = _storage_config.get("endpoint", "")
    _s3 = boto3.client(
        "s3",
        endpoint_url=_endpoint,
        aws_access_key_id=_storage_config.get("access_key_id", ""),
        aws_secret_access_key=_storage_config.get("access_key_secret", ""),
        config=BotoConfig(
            signature_version="s3",
            s3={"addressing_style": "virtual"},
        ),
    )
    logger.info(f"OSS/S3 client initialized: endpoint={_endpoint}, bucket={_BUCKET}, prefix={_PREFIX}")
else:
    logger.info("Using local file storage")


class StorageFile(BaseModel):
    doc_id: str = Field(default="", description="document id")
    filename: str = Field(default="", description="filename")
    doc_type: str = Field(default="", description="document type")
    file_path: str = Field(default="", description="file path or object key")


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: 255 - len(ext)] + ext
    return sanitized


def generate_doc_id(doc_type: str) -> str:
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:6]
    return f"{doc_type}_{timestamp}_{unique_id}"


# ---------------------------------------------------------------------------
# Key / path helpers
# ---------------------------------------------------------------------------


def _original_key(doc_type: str, doc_id: str, filename: str) -> str:
    safe = sanitize_filename(filename)
    return f"{_PREFIX}/original/{doc_type}/{doc_id}/{safe}" if _PREFIX else f"original/{doc_type}/{doc_id}/{safe}"


def _parsed_key(doc_type: str, doc_id: str, filename: str) -> str:
    safe = sanitize_filename(filename)
    name_without_ext = os.path.splitext(safe)[0]
    key = f"parsed/{doc_type}/{doc_id}/{name_without_ext}.json"
    return f"{_PREFIX}/{key}" if _PREFIX else key


def get_original_path(filename: str, doc_id: str, doc_type: str) -> Path:
    safe_filename = sanitize_filename(filename)
    return ORIGINAL_DIR / doc_type / doc_id / safe_filename


def get_parsed_path(doc_id: str, filename: str, doc_type: str) -> Path:
    safe_filename = sanitize_filename(filename)
    name_without_ext = os.path.splitext(safe_filename)[0]
    return PARSED_DIR / doc_type / doc_id / f"{name_without_ext}.json"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_original_file(filename: str, contents: bytes, doc_type: str) -> StorageFile:
    try:
        doc_id = generate_doc_id(doc_type)

        if _s3:
            key = _original_key(doc_type, doc_id, filename)
            _s3.put_object(Bucket=_BUCKET, Key=key, Body=contents)
            logger.info(f"Original file uploaded to OSS: {key}")
            return StorageFile(doc_id=doc_id, filename=filename, doc_type=doc_type, file_path=key)

        file_path = get_original_path(filename, doc_id, doc_type)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"Original file saved to: {file_path}")
        return StorageFile(doc_id=doc_id, filename=filename, doc_type=doc_type, file_path=str(file_path))
    except Exception as e:
        raise Exception(f"Failed to save original file for '{filename}': {e}")


def save_parsed_file(filename: str, file_info: dict[str, Any]) -> StorageFile:
    try:
        doc_id = file_info.get("document_id")
        doc_type = file_info.get("doc_type")

        if _s3:
            key = _parsed_key(doc_type, doc_id, filename)
            body = json.dumps(file_info, ensure_ascii=False, indent=2).encode("utf-8")
            _s3.put_object(Bucket=_BUCKET, Key=key, Body=body, ContentType="application/json")
            logger.info(f"Parsed file uploaded to OSS: {key}")
            return StorageFile(doc_id=doc_id, filename=filename, doc_type=doc_type, file_path=key)

        db_path = get_parsed_path(doc_id, filename, doc_type)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(file_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Parsed file saved to: {db_path}")
        return StorageFile(doc_id=doc_id, filename=filename, doc_type=doc_type, file_path=str(db_path))
    except Exception as e:
        raise Exception(f"Failed to save parsed file for '{filename}': {e}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_original_file(filename: str, doc_id: str, doc_type: str) -> Optional[bytes]:
    try:
        if _s3:
            key = _original_key(doc_type, doc_id, filename)
            resp = _s3.get_object(Bucket=_BUCKET, Key=key)
            return resp["Body"].read()

        file_path = get_original_path(filename, doc_id, doc_type)
        if not file_path.exists():
            logger.debug(f"Original file not found: {file_path}")
            return None
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        if _s3 and hasattr(e, "response") and e.response.get("Error", {}).get("Code") == "NoSuchKey":
            logger.debug(f"Original file not found in OSS: {_original_key(doc_type, doc_id, filename)}")
            return None
        logger.error(f"Failed to load original file '{filename}': {e}")
        return None


def load_parsed_file(filename: str, doc_id: str, doc_type: str) -> Optional[Dict]:
    try:
        if _s3:
            key = _parsed_key(doc_type, doc_id, filename)
            resp = _s3.get_object(Bucket=_BUCKET, Key=key)
            return json.loads(resp["Body"].read().decode("utf-8"))

        db_path = get_parsed_path(doc_id, filename, doc_type)
        if not db_path.exists():
            logger.debug(f"Parsed file not found: {db_path}")
            return None
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for '{filename}': {e}")
        if not _s3:
            db_path = get_parsed_path(doc_id, filename, doc_type)
            if db_path and db_path.exists():
                try:
                    db_path.unlink()
                    logger.warning(f"Deleted corrupted file: {db_path}")
                except Exception as del_error:
                    logger.error(f"Failed to delete corrupted file: {del_error}")
        return None
    except Exception as e:
        if _s3 and hasattr(e, "response") and e.response.get("Error", {}).get("Code") == "NoSuchKey":
            logger.debug(f"Parsed file not found in OSS: {_parsed_key(doc_type, doc_id, filename)}")
            return None
        logger.error(f"Failed to load parsed file '{filename}': {e}")
        return None
