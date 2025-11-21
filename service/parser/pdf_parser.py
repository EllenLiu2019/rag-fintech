from .base import BaseParser
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PDFParser(BaseParser):
    """PDF 文件解析器"""
    
    supported_extensions = ['.pdf']
    supported_mime_types = ['application/pdf']
    
    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        return filename.lower().endswith('.pdf') or \
               (content_type and 'pdf' in content_type.lower())
    
    def extract_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> str:
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            import sys
            error_msg = (
                f"PDF parsing failed: PyPDF2 not installed in current Python environment ({sys.executable})\n"
                f"Please activate virtual environment and install: pip install PyPDF2>=3.0.1"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            logger.error(f"PDF parsing failed: {str(e)}")
            raise ValueError(f"PDF parsing failed: {str(e)}")