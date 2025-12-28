import json
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from common import file_utils, get_logger
from common.constants import LLM_FACTORIES_CONF
from common.decorator import singleton

logger = get_logger(__name__)

# Type definitions for model purposes
ChatModelPurpose = Literal["qa_reasoner", "qa_lite", "query_lite", "query_reasoner"]
EmbeddingModelPurpose = Literal["dense", "sparse"]
RerankerModelPurpose = Literal["jina", "cohere", "bge"]


@dataclass
class ModelConfig:

    provider: str
    model_name: str
    max_tokens: int = 0
    description: str = ""
    dimensions: Optional[int] = None  # For embedding models
    base_url: Optional[str] = None  # For service-based models (TEI, etc.)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "provider": self.provider,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "description": self.description,
            "dimensions": self.dimensions,
        }
        if self.base_url:
            result["base_url"] = self.base_url
        return result


@singleton
class ModelRegistry:
    def __init__(self):
        self._chat_models: Dict[str, ModelConfig] = {}
        self._embedding_models: Dict[str, ModelConfig] = {}
        self._reranker_models: Dict[str, ModelConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        try:
            config_path = file_utils.get_project_root_dir("conf", LLM_FACTORIES_CONF)
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Load chat models
            for purpose, model_conf in config.get("chat_models", {}).items():
                self._chat_models[purpose] = ModelConfig(
                    provider=model_conf["provider"],
                    model_name=model_conf["model_name"],
                    max_tokens=model_conf.get("max_tokens", 4096),
                    base_url=model_conf.get("base_url"),
                    description=model_conf.get("description", ""),
                )

            # Load embedding models
            for purpose, model_conf in config.get("embedding_models", {}).items():
                self._embedding_models[purpose] = ModelConfig(
                    provider=model_conf["provider"],
                    model_name=model_conf["model_name"],
                    max_tokens=model_conf.get("max_tokens", 8192),
                    description=model_conf.get("description", ""),
                    dimensions=model_conf.get("dimensions"),
                )

            # Load reranker models
            for purpose, model_conf in config.get("reranker_models", {}).items():
                self._reranker_models[purpose] = ModelConfig(
                    provider=model_conf["provider"],
                    model_name=model_conf["model_name"],
                    description=model_conf.get("description", ""),
                    base_url=model_conf.get("base_url"),
                )

            logger.info(
                f"Loaded {len(self._chat_models)} chat models, "
                f"{len(self._embedding_models)} embedding models, "
                f"{len(self._reranker_models)} reranker models"
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"Model config not found: {LLM_FACTORIES_CONF}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {LLM_FACTORIES_CONF}: {e}")

    def get_chat_model(self, purpose: ChatModelPurpose) -> ModelConfig:
        if purpose not in self._chat_models:
            available = list(self._chat_models.keys())
            raise KeyError(f"Unknown chat model purpose: {purpose}. Available: {available}")
        return self._chat_models[purpose]

    def get_embedding_model(self, purpose: EmbeddingModelPurpose) -> ModelConfig:
        if purpose not in self._embedding_models:
            available = list(self._embedding_models.keys())
            raise KeyError(f"Unknown embedding model purpose: {purpose}. Available: {available}")
        return self._embedding_models[purpose]

    def get_reranker_model(self, purpose: RerankerModelPurpose = "jina") -> ModelConfig:
        if purpose not in self._reranker_models:
            available = list(self._reranker_models.keys())
            raise KeyError(f"Unknown reranker model purpose: {purpose}. Available: {available}")
        return self._reranker_models[purpose]


def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
