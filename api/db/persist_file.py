"""
文件持久化模块
负责将文件信息保存到磁盘和从磁盘读取
"""
import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, List

# 添加项目根目录到 Python 路径
import sys
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.log_utils import get_logger

logger = get_logger(__name__)

# 数据库文件夹路径
DB_DIR = Path(__file__).parent
DB_DIR.mkdir(exist_ok=True)  # 确保目录存在

# 原始文件存储文件夹路径
FILES_DIR = DB_DIR / "original_files"
FILES_DIR.mkdir(exist_ok=True)  # 确保目录存在


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除或替换不安全字符
    保留基本字符，将特殊字符替换为下划线
    """
    # 移除路径分隔符和其他危险字符
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # 限制长度
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    return sanitized


def get_file_db_path(filename: str) -> Path:
    """获取文件的数据库存储路径"""
    safe_filename = sanitize_filename(filename)
    # 去掉文件扩展名，只保留文件名
    name_without_ext = os.path.splitext(safe_filename)[0]
    return DB_DIR / f"{name_without_ext}.json"


def save_file_info(filename: str, file_info: Dict) -> bool:
    """
    将文件信息保存到 JSON 文件
    
    Args:
        filename: 原始文件名
        file_info: 文件信息字典
        
    Returns:
        bool: 保存是否成功
    """
    try:
        db_path = get_file_db_path(filename)
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(file_info, f, ensure_ascii=False, indent=2)
        logger.info(f"file info saved to: {db_path}")
        return True
    except Exception as e:
        logger.error(f"failed to save file info: {str(e)}")
        return False


def load_file_info(filename: str) -> Optional[Dict]:
    """
    从 JSON 文件加载文件信息
    
    Args:
        filename: 原始文件名
        
    Returns:
        dict: 文件信息字典，如果文件不存在则返回 None
    """
    try:
        db_path = get_file_db_path(filename)
        if not db_path.exists():
            logger.debug(f"JSON file not found: {db_path}")
            return None
        
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {db_path}: {str(e)}")
        logger.error(f"   File may be corrupted. Consider deleting and re-uploading the file.")
        # 可选：自动删除损坏的文件
        try:
            db_path.unlink()
            logger.warning(f"Deleted corrupted file: {db_path}")
        except Exception as del_error:
            logger.error(f"Failed to delete corrupted file: {str(del_error)}")
        return None
    except Exception as e:
        logger.error(f"failed to load file info from {db_path}: {str(e)}")
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
    return FILES_DIR / safe_filename


def save_original_file(filename: str, contents: bytes) -> bool:
    """
    保存原始文件到磁盘
    
    Args:
        filename: 原始文件名
        contents: 文件内容（字节）
        
    Returns:
        bool: 保存是否成功
    """
    try:
        file_path = get_original_file_path(filename)
        with open(file_path, 'wb') as f:
            f.write(contents)
        logger.info(f"original file saved to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"failed to save original file: {str(e)}")
        return False


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
        
        with open(file_path, 'rb') as f:
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
        for json_file in DB_DIR.glob("*.json"):
            # 从 JSON 文件中读取原始文件名
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'filename' in data:
                        files.append(data['filename'])
            except:
                # 如果读取失败，使用文件名（去掉 .json 扩展名）
                files.append(json_file.stem)
        return files
    except Exception as e:
        logger.error(f"failed to list stored files: {str(e)}")
        return []

