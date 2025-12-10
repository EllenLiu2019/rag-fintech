"""
文件持久化模块
负责将文件信息保存到磁盘和从磁盘读取
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, List

from common import config, file_utils, get_logger
from repository.rdb.models.models import Document as RdbDocument, KnowledgeBase, LLM
from rag.ingestion.document import RagDocument

logger = get_logger(__name__)

# Convert to Path objects for proper path operations
ORIGINAL_FILES_DIR = Path(file_utils.get_project_root_dir("repository", "s3", "original_files"))
PARSED_FILES_DIR = Path(file_utils.get_project_root_dir("repository", "s3", "parsed_files"))


class DocumentService:
    def __init__(self):
        self.rdb_client = config.RDB_CLIENT

    def upload_file(self, filename: str, contents: bytes, content_type: str):
        logger.info(f"uploading file {filename}")

        file_path = save_original_file(filename, contents)
        if content_type == "application/pdf":
            kb_name = "default_kb"
        else:
            kb_name = "unknown"

        # kb_ids = self.rdb_client.execute_query(KnowledgeBase, kb_name)
        # if len(kb_ids) == 0:
        #     raise ValueError(f"Knowledge base {kb_name} not found")

        document = RdbDocument(
            file_name=filename,
            file_location=file_path,
            content_type=content_type,
            file_size=len(contents),
            doc_status="uploaded",
            kb_name=kb_name,
        )
        rdb_document = self.rdb_client.save(document)

        logger.info(f"file {filename} saved to rdb: id={rdb_document.id} with original file location: {file_path}")

        return rdb_document

    def update_file_info(self, filename: str, rag_document: RagDocument, rdb_document: RdbDocument):
        logger.info(f"updating file {filename} in rdb: id={rdb_document.id}")

        parsed_file = rag_document.to_parsed_file()
        parsed_file_path = save_file_info(filename, parsed_file)

        # Update document attributes
        rdb_document.document_id = rag_document.document_id
        rdb_document.file_name = rag_document.filename
        rdb_document.doc_status = "completed"
        rdb_document.doc_location = parsed_file_path
        rdb_document.content_type = rag_document.content_type
        rdb_document.page_count = len(rag_document.pages)
        rdb_document.upload_time = rag_document.upload_time
        rdb_document.business_data = rag_document.business_data
        rdb_document.confidence = rag_document.confidence
        rdb_document.token_num += rag_document.token_num
        rdb_document.chunk_num += rag_document.chunk_num
        # Save the updated document (merge will update existing record)
        updated_doc = self.rdb_client.save(rdb_document)

        logger.info(
            f"file {filename} updated in rdb: id={updated_doc.id} with parsed file location: {parsed_file_path}"
        )

    def get_embedding_model(self, kb_name: str) -> str:
        kb_ids = self.rdb_client.execute_query(KnowledgeBase, kb_name)
        kb_id = kb_ids[0]
        llm_model = self.rdb_client.select_by_id(LLM, kb_id.embed_llm_id)
        if llm_model is None:
            raise ValueError(f"LLM model not found for knowledge base {kb_id.kb_name}")
        return llm_model

    def download_file(self, filename: str) -> bytes:
        return load_original_file(filename)

    def list_files(self) -> List[str]:
        return list_stored_files()


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


def get_file_db_path(filename: str) -> Path:
    """获取文件的数据库存储路径"""
    safe_filename = sanitize_filename(filename)
    # 去掉文件扩展名，只保留文件名
    name_without_ext = os.path.splitext(safe_filename)[0]
    return PARSED_FILES_DIR / f"{name_without_ext}.json"


def save_file_info(filename: str, file_info: Dict) -> str:
    """
    将文件信息保存到 JSON 文件

    Args:
        filename: 原始文件名
        file_info: 文件信息字典

    Returns:
        str: 文件信息存储路径（字符串格式，用于数据库存储）
    """
    try:
        db_path = get_file_db_path(filename)
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(file_info, f, ensure_ascii=False, indent=2)
        logger.info(f"file info saved to: {db_path}")
        # Return as string for database storage
        return str(db_path)
    except Exception as e:
        raise Exception(f"failed to save file info to '{filename}': {str(e)}")


def load_file_info(filename: str) -> Optional[Dict]:
    """
    从 JSON 文件加载文件信息

    Args:
        filename: 原始文件名

    Returns:
        dict: 文件信息字典，如果文件不存在则返回 None
    """
    db_path = None  # Initialize to avoid UnboundLocalError
    try:
        db_path = get_file_db_path(filename)
        if not db_path.exists():
            logger.debug(f"JSON file not found: {db_path}")
            return None

        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {db_path}: {str(e)}")
        logger.error("   File may be corrupted. Consider deleting and re-uploading the file.")
        # 可选：自动删除损坏的文件
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


def get_original_file_path(filename: str) -> Path:
    """
    获取原始文件的存储路径

    Args:
        filename: 原始文件名

    Returns:
        Path: 文件存储路径
    """
    safe_filename = sanitize_filename(filename)
    return ORIGINAL_FILES_DIR / safe_filename


def save_original_file(filename: str, contents: bytes) -> str:
    """
    保存原始文件到磁盘

    Args:
        filename: 原始文件名
        contents: 文件内容（字节）

    Returns:
        str: 文件存储路径（字符串格式，用于数据库存储）
    """
    try:
        file_path = get_original_file_path(filename)
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"original file saved to: {file_path}")
        # Return as string for database storage
        return str(file_path)
    except Exception as e:
        raise Exception(f"failed to save original file for '{filename}': {str(e)}")


def load_original_file(filename: str) -> Optional[bytes]:
    """
    从磁盘加载原始文件

    Args:
        filename: 原始文件名

    Returns:
        bytes: 文件内容，如果文件不存在则返回 None
    """
    try:
        file_path = get_original_file_path(filename)
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


def list_stored_files() -> List[str]:
    """
    列出所有已存储的文件名（从 JSON 文件名提取）

    Returns:
        list: 文件名列表
    """
    try:
        files = []
        for json_file in PARSED_FILES_DIR.glob("*.json"):
            # 从 JSON 文件中读取原始文件名
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "filename" in data:
                        files.append(data["filename"])
            except Exception:
                # 如果读取失败，使用文件名（去掉 .json 扩展名）
                files.append(json_file.stem)
        return files
    except Exception as e:
        logger.error(f"failed to list stored files: {str(e)}")
        return []
