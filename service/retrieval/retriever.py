from service.embedding.embedder import EmbeddingService
from repository.vector.milvus_client import VectorStoreClient
from repository.vector.doc_store_client import MatchDenseExpr
import logging

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        self.embedder = EmbeddingService()
        self.vector_store = VectorStoreClient()

    def search(self, query: str, kb_id: str = "default_kb", top_k: int = 5, filters: dict = None):

        logger.info(f"Searching for: '{query}' in kb: {kb_id}")

        # 1. Embed Query
        query_vector = self.embedder.embed_query(query)

        if not query_vector:
            logger.warning("Empty query vector generated.")
            return []

        # 2. Construct Search Expression
        match_expr = MatchDenseExpr(
            vector_column_name="dense_vector",
            embedding_data=query_vector,
            topn=top_k,
            embedding_data_type="float",
            distance_type="COSINE",
            extra_options={"filters": filters or {}},
        )

        # 3. Execute Search
        try:
            results = self.vector_store.search(
                selectFields=["id", "text", "policy_number", "holder_name", "insured_name", "metadata", "doc_id"],
                matchExprs=[match_expr],
                limit=top_k,
                indexNames="rag_fintech",
                knowledgebaseIds=[kb_id],
            )

            logger.info(f"Found {len(results) if results else 0} results.")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
