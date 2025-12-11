import json
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from common import file_utils
from common.constants import LLM_FACTORIES_CONF
from common.log_utils import get_logger
from common.decorator import singleton

logger = get_logger(__name__)

# Type definitions for model purposes
ChatModelPurpose = Literal["qa_reasoner", "qa_lite", "query_lite", "query_reasoner"]
EmbeddingModelPurpose = Literal["dense", "sparse"]


@dataclass
class ModelConfig:

    provider: str
    model_name: str
    max_tokens: int
    description: str = ""
    dimensions: Optional[int] = None  # For embedding models

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "description": self.description,
            "dimensions": self.dimensions,
        }


@singleton
class ModelRegistry:
    def __init__(self):
        self._chat_models: Dict[str, ModelConfig] = {}
        self._embedding_models: Dict[str, ModelConfig] = {}
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

            logger.info(f"Loaded {len(self._chat_models)} chat models, {len(self._embedding_models)} embedding models")

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


def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
