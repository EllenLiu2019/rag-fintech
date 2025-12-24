from typing import Optional

from rag.ingestion.parser.base import BaseParser
from rag.ingestion.parser.parser import ParseResult
from common import get_logger
from common.prompt_manager import get_prompt_manager

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """PDF 文件解析器"""

    supported_extensions = [".pdf"]
    supported_mime_types = ["application/pdf"]
    prompt_manager = get_prompt_manager()

    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        return filename.lower().endswith(".pdf") or (content_type and "pdf" in content_type.lower())

    def parse_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> ParseResult:
        extra_info = {"file_name": filename, "content_type": content_type, "size": len(contents)}
        logger.info(f"extract document info: {extra_info}")

        try:
            import os
            from llama_parse import LlamaParse

            llm_parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                parse_mode="parse_document_with_agent",
                model="anthropic-sonnet-4.5",
                high_res_ocr=True,
                adaptive_long_table=True,
                outlined_table_extraction=True,
                output_tables_as_HTML=True,
                preset="forms",
                language="ch_sim",
                preserve_layout_alignment_across_pages=True,
                merge_tables_across_pages_in_markdown=True,
                ignore_document_elements_for_layout_detection=False,
                do_not_cache=True,
                strict_mode_reconstruction=True,
                strict_mode_buggy_font=True,
                user_prompt=self.prompt_manager.get("parse_prompt"),
                system_prompt_append=self.prompt_manager.get("parse_prompt"),
                hide_footers=True,
                preserve_very_small_text=True,
                invalidate_cache=True,
            )

            logger.info("LlamaParse parse job starting...")
            result = llm_parser.parse(contents, extra_info=extra_info)

            markdown_documents = result.get_markdown_documents(split_by_page=True)

            if markdown_documents:
                first_doc_text = markdown_documents[0].text if markdown_documents[0].text else ""
                result_preview = first_doc_text[:100] if first_doc_text else ""
                logger.info(
                    f"return doc count: {len(markdown_documents)} documents; "
                    f"first doc preview (first 100 chars): {result_preview}"
                )
            else:
                logger.warning("no documents returned from LlamaParse")

            return ParseResult(documents=markdown_documents, job_id=result.job_id)
        except Exception as e:
            logger.error(f"PDF parsing failed: {str(e)}")
            raise ValueError(f"PDF parsing failed: {str(e)}")
