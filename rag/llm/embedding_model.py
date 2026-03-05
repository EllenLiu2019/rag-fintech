from abc import ABC
from typing import List, Dict
import numpy as np
import time

from common import get_logger

logger = get_logger(__name__)


class Base(ABC):
    def __init__(self, key, model_name, **kwargs):
        """
        Parameters are accepted for interface consistency but are not stored.
        Subclasses should implement their own initialization as needed.
        """
        pass

    def encode(self, texts: list):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")

    def total_token_count(self, resp):
        try:
            return resp.usage.total_tokens
        except Exception:
            pass
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass
        return 0


class VoyageEmbed(Base):
    _FACTORY_NAME = "Voyage AI"

    def __init__(self, key, model_name, base_url=None):
        import voyageai

        self.client = voyageai.Client(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        batch_size = 16
        ress = []
        token_count = 0
        for i in range(0, len(texts), batch_size):
            res = self.client.embed(texts=texts[i : i + batch_size], model=self.model_name, input_type="document")
            try:
                ress.extend(res.embeddings)
                token_count += res.total_tokens
            except Exception as _e:
                logger.error(f"Failed to encode text: {_e}")
        return np.array(ress), token_count

    def encode_queries(self, text):
        res = self.client.embed(texts=[text], model=self.model_name, input_type="query")
        try:
            return np.array(res.embeddings)[0], res.total_tokens
        except Exception as _e:
            logger.error(f"Failed to encode query: {_e}")


class BAAIBgeM3Embed(Base):
    _FACTORY_NAME = "BGE-M3"

    def __init__(self, key, model_name, base_url=None):
        from FlagEmbedding import BGEM3FlagModel

        self.model_name = model_name
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
            batch_size=256,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def encode(self, texts: list):
        try:
            start = time.time()
            res: List[Dict[str, float]] = self.model.encode(sentences=texts)["lexical_weights"]
            elapsed = time.time() - start
            logger.info(f"BAAI BGE-M3 encode {len(texts)} texts completed in {elapsed:.3f}s.")

            return res
        except Exception as _e:
            logger.error(f"Failed to encode {len(texts)} texts: {_e}")
            raise


class MilvusBgeM3Embed(Base):
    _FACTORY_NAME = "Milvus-BGE-M3"

    def __init__(self, key, model_name, base_url=None):
        from pymilvus import model
        from huggingface_hub import snapshot_download

        self.model_name = model_name
        try:
            local_path = snapshot_download(model_name, local_files_only=True)
        except Exception:
            local_path = snapshot_download(model_name, local_files_only=False)
        self.model = model.hybrid.BGEM3EmbeddingFunction(
            model_name=local_path,
            device="cpu",
            normalize_embeddings=False,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def encode(self, texts: list):
        from scipy.sparse import csr_array

        try:
            start = time.time()
            res: List[csr_array] = self.model(texts)["sparse"]
            elapsed = time.time() - start
            logger.info(f"Milvus BGE-M3 encode {len(texts)} texts completed in {elapsed:.3f}s.")

            sparse_vectors = []
            for sparse_vector in res:
                sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_vector.indices, sparse_vector.data)}
                sparse_vectors.append(sparse_dict)
            return sparse_vectors
        except Exception as _e:
            logger.error(f"Failed to encode text: {_e}")
            raise


# Provider -> Class mapping
embedding_model = {
    "Voyage": VoyageEmbed,
    "BAAI": BAAIBgeM3Embed,
    "Milvus": MilvusBgeM3Embed,
}
