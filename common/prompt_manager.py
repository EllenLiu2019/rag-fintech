import yaml
from typing import Dict, Any
from pathlib import Path
from jinja2 import Template

from common import file_utils, get_logger

logger = get_logger(__name__)


class PromptManager:
    """
    Centralized prompt management with:
    - YAML-based prompt storage
    - Jinja2 template rendering
    - Prompt versioning support
    """

    def __init__(self):
        self._prompts: Dict[str, Dict[str, Any]] = {}
        self._load_all_prompts()

    def _load_all_prompts(self) -> None:
        """Load all prompt YAML files from conf/prompts directory."""
        prompts_dir = Path(file_utils.get_project_root_dir("conf", "prompts"))

        if not prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {prompts_dir}")
            return

        for yaml_file in prompts_dir.rglob("*.yaml"):
            self._load_prompt_file(yaml_file)

        logger.info(f"Loaded {len(self._prompts)} prompt templates")

    def _load_prompt_file(self, file_path: Path) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                return

            for prompt_key, prompt_config in data.items():
                if isinstance(prompt_config, dict) and "template" in prompt_config:
                    self._prompts[prompt_key] = prompt_config
                    logger.debug(f"Loaded prompt: {prompt_key} (v{prompt_config.get('version', 'unknown')})")

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

    def get(self, prompt_key: str, **variables) -> str:
        if prompt_key not in self._prompts:
            available = list(self._prompts.keys())
            raise KeyError(f"Prompt not found: {prompt_key}. Available: {available}")

        template_str = self._prompts[prompt_key]["template"]

        if variables:
            template = Template(template_str)
            return template.render(**variables)

        return template_str

    def get_raw(self, prompt_key: str) -> str:
        if prompt_key not in self._prompts:
            raise KeyError(f"Prompt not found: {prompt_key}")
        return self._prompts[prompt_key]["template"]

    def get_metadata(self, prompt_key: str) -> Dict[str, Any]:
        if prompt_key not in self._prompts:
            raise KeyError(f"Prompt not found: {prompt_key}")

        config = self._prompts[prompt_key]
        return {
            "name": config.get("name", ""),
            "version": config.get("version", "unknown"),
            "description": config.get("description", ""),
        }

    def list_prompts(self) -> Dict[str, Dict[str, Any]]:
        return {key: self.get_metadata(key) for key in self._prompts}

    def reload(self) -> None:
        self._prompts.clear()
        self._load_all_prompts()


def get_prompt_manager() -> PromptManager:
    return PromptManager()


prompt_manager = get_prompt_manager()
