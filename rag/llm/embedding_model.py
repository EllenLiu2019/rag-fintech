from abc import ABC

import numpy as np

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


# Provider -> Class mapping
embedding_model = {
    "Voyage": VoyageEmbed,
}
