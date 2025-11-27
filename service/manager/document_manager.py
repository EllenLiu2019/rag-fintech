from service.parser import parse_content
import logging
from service.extractor.extraction_pipeline import ExtractionPipeline
from service.extractor.schema.metadata_creator import MetadataCreator
from service.manager.document import RagDocument
from service.parser.serializer_deserializer import serialize_documents
from typing import Any

logger = logging.getLogger(__name__)


class DocumentManager:
    def __init__(self):
        self.extraction_pipeline = ExtractionPipeline()
        self.metadata_creator = MetadataCreator()

    def handle_document(self, filename: str, contents: bytes, content_type: str) -> RagDocument:
        try:
            documents = parse_content(contents, filename, content_type)
        except ValueError as e:
            raise ValueError(f"file parsing failed: {str(e)}")

        pages = serialize_documents(documents)

        rag_document = self.build_from_parsed_documents(filename, contents, content_type, pages)

        # TODO: Asynchronously save rag_document to Milvus (next step)
        # self.milvus_service.save(rag_document)

        return rag_document

    def build_from_parsed_documents(
        self, filename: str, contents: bytes, content_type: str, pages: list[dict[str, Any]]
    ) -> RagDocument:

        extracted_data, confidence = self.extraction_pipeline.run(pages)
        metadata = self.metadata_creator.create(extracted_data)

        document_id = metadata.get("document_id")
        if not document_id:
            document_id = filename

        rag_document = RagDocument.from_extraction_result(
            parsed_documents=pages,
            extracted_data=extracted_data,
            confidence=confidence,
            metadata=metadata,
            filename=filename,
            file_size=len(contents),
            content_type=content_type,
            document_id=document_id,
        )

        logger.info(
            f"Built Document for '{filename}': "
            f"{len(pages)} pages, "
            f"{len(extracted_data)} extracted fields, "
            f"{len(metadata)} metadata fields"
        )

        return rag_document
