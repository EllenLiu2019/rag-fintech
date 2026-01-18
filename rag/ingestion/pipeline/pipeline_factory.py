from typing import Dict

from .base_pipeline import BasePipeline
from rag.entity import DocumentType
from .policy_pipeline import PolicyPipeline
from .claim_pipeline import ClaimPipeline
from common import get_logger

logger = get_logger(__name__)


class PipelineFactory:

    _pipelines: Dict[DocumentType, type] = {
        DocumentType.POLICY: PolicyPipeline,
        DocumentType.CLAIM: ClaimPipeline,
    }

    @classmethod
    def create(cls, doc_type: DocumentType) -> BasePipeline:
        pipeline_class = cls._pipelines.get(doc_type)

        if pipeline_class is None:
            raise ValueError(
                f"Unsupported document type: {doc_type}. " f"Supported types: {list(cls._pipelines.keys())}"
            )

        logger.info(f"Creating pipeline for document type: {doc_type.value}")
        return pipeline_class(doc_type)

    @classmethod
    def register_pipeline(cls, doc_type: DocumentType, pipeline_class: type):
        if not issubclass(pipeline_class, BasePipeline):
            raise TypeError("Pipeline class must inherit from BasePipeline")

        cls._pipelines[doc_type] = pipeline_class
        logger.info(f"Registered pipeline for type: {doc_type.value}")
