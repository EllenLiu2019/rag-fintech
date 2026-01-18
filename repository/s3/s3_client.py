import os
import re
import json
import uuid
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from common import get_logger, file_utils

logger = get_logger(__name__)


ORIGINAL_DIR = Path(file_utils.get_project_root_dir("repository", "s3", "original"))
PARSED_DIR = Path(file_utils.get_project_root_dir("repository", "s3", "parsed"))


class StorageFile(BaseModel):
    doc_id: str = Field(default="", description="document id")
    filename: str = Field(default="", description="filename")
    doc_type: str = Field(default="", description="document type")
    file_path: str = Field(default="", description="file path")


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除或替换不安全字符
    保留基本字符，将特殊字符替换为下划线
    """
    # 移除路径分隔符和其他危险字符
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    # 限制长度
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: 255 - len(ext)] + ext
    return sanitized


def generate_doc_id(doc_type: str) -> str:
    """generate unique id"""
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:6]
    return f"{doc_type}_{timestamp}_{unique_id}"


def get_original_path(filename: str, doc_id: str, doc_type: str) -> Path:
    """get original file path"""
    safe_filename = sanitize_filename(filename)
    return ORIGINAL_DIR / doc_type / doc_id / safe_filename


def get_parsed_path(doc_id: str, filename: str, doc_type: str) -> Path:
    """get parsed file path"""
    safe_filename = sanitize_filename(filename)
    name_without_ext = os.path.splitext(safe_filename)[0]
    return PARSED_DIR / doc_type / doc_id / f"{name_without_ext}.json"


def save_original_file(filename: str, contents: bytes, doc_type: str) -> StorageFile:
    try:
        doc_id = generate_doc_id(doc_type)
        file_path = get_original_path(filename, doc_id, doc_type)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"original file saved to: {file_path}")
        return StorageFile(doc_id=doc_id, filename=filename, doc_type=doc_type, file_path=str(file_path))
    except Exception as e:
        raise Exception(f"failed to save original file for '{filename}': {str(e)}")


def save_parsed_file(filename: str, file_info: dict[str, Any]) -> StorageFile:
    try:
        doc_id = file_info.get("document_id")
        doc_type = file_info.get("doc_type")

        db_path = get_parsed_path(doc_id, filename, doc_type)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(file_info, f, ensure_ascii=False, indent=2)

        logger.info(f"file info saved to: {db_path}")
        return StorageFile(doc_id=doc_id, filename=filename, doc_type=doc_type, file_path=str(db_path))
    except Exception as e:
        raise Exception(f"failed to save file info to '{filename}': {str(e)}")


def load_original_file(filename: str, doc_id: str, doc_type: str) -> Optional[bytes]:
    """
    load original file from disk
    """
    try:
        file_path = get_original_path(filename, doc_id, doc_type)
        if not file_path.exists():
            logger.debug(f"original file not found: {file_path}")
            return None

        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"failed to load original file from {file_path}: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return None


def load_parsed_file(filename: str, doc_id: str, doc_type: str) -> Optional[Dict]:
    """
    load file info from JSON file
    """
    try:
        db_path = get_parsed_path(doc_id, filename, doc_type)
        if not db_path.exists():
            logger.debug(f"JSON file not found: {db_path}")
            return None

        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {db_path}: {str(e)}")
        logger.error("   File may be corrupted. Consider deleting and re-uploading the file.")
        if db_path and db_path.exists():
            try:
                db_path.unlink()
                logger.warning(f"Deleted corrupted file: {db_path}")
            except Exception as del_error:
                logger.error(f"Failed to delete corrupted file: {str(del_error)}")
        return None
    except Exception as e:
        logger.error(f"failed to load file info from {db_path if db_path else filename}: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return None
