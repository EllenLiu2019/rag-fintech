import re
import json
from typing import Dict

from langchain_openai.chat_models.base import _convert_message_to_dict as _original_convert_message_to_dict
from langchain_core.messages import BaseMessage, AIMessage
import langchain_openai.chat_models.base

from common import get_logger

logger = get_logger(__name__)


def extract_content(text: str) -> dict:
    """
    Extract JSON from text that may contain descriptive text, markdown code blocks, etc.
    """
    if not text or not text.strip():
        return {}

    # Remove markdown code block markers
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    # Try to find JSON object by matching balanced braces
    start_idx = text.find("{")
    if start_idx == -1:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    # Find the matching closing brace
    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if brace_count != 0:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    json_str = text[start_idx:end_idx]

    # Remove comments
    json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse extracted JSON: {e}, json_str: {json_str[:200]}")
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}


def extract_ai_message(message: AIMessage) -> dict | None:

    medical_response_data = None
    if isinstance(message, AIMessage) and message.content:
        try:
            medical_response_data = extract_content(message.content) if isinstance(message.content, str) else {}
        except Exception as e:
            logger.warning(f"Failed to extract JSON from message: {e}")

    if not medical_response_data:
        logger.warning("No AI message found in agent result")
        return None

    return medical_response_data


def _patched_convert_message_to_dict(message: BaseMessage) -> Dict:
    result = _original_convert_message_to_dict(message)

    # move it to the top level of the dict
    if isinstance(message, AIMessage) and message.tool_calls:
        reasoning_content = message.additional_kwargs.get("reasoning_content")
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content

    return result


langchain_openai.chat_models.base._convert_message_to_dict = _patched_convert_message_to_dict
