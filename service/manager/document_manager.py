import uuid

from llama_index.core.schema import Document
from service.parser import parse_content
import logging
from service.extractor.extraction_pipeline import ExtractionPipeline
from service.manager.document import RagDocument
from service.parser.serializer_deserializer import serialize_documents

logger = logging.getLogger(__name__)


class DocumentManager:
    def __init__(self):
        self.extraction_pipeline = ExtractionPipeline()

    def handle_document(self, filename: str, contents: bytes, content_type: str) -> RagDocument:
        try:
            documents = parse_content(contents, filename, content_type)
        except ValueError as e:
            raise ValueError(f"file parsing failed: {str(e)}")

        rag_document = self.build_from_parsed_documents(filename, contents, content_type, documents)

        # TODO: Asynchronously save rag_document to Milvus (next step)
        # self.milvus_service.save(rag_document)

        return rag_document

    def build_from_parsed_documents(
        self, filename: str, contents: bytes, content_type: str, documents: list[Document]
    ) -> RagDocument:

        pages = serialize_documents(documents)
        extracted_data, confidence, metadata = self.extraction_pipeline.run(pages)

        rag_document = RagDocument.from_extraction_result(
            parsed_documents=pages,
            extracted_data=extracted_data,
            confidence=confidence,
            metadata=metadata,
            filename=filename,
            file_size=len(contents),
            content_type=content_type,
            document_id=str(uuid.uuid4()),
        )

        logger.info(
            f"Built RagDocument for '{filename}': "
            f"{len(pages)} pages, "
            f"{len(extracted_data)} extracted fields, "
            f"{len(metadata)} metadata fields"
        )

        return rag_document
