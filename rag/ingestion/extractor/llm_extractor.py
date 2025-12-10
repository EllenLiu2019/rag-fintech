import json
from typing import Any
from rag.llm.chat_model import DeepSeek

function_mapping = {
    "DeepSeek": DeepSeek,
}


class LLMExtractor:

    def __init__(self, chat_model: dict[str, Any]):
        model_name = chat_model["models"][1]["model_name"]
        self.llm = function_mapping[chat_model["provider"]](model_name=model_name)

    def extract(self, content: str, hints: dict = None) -> dict:

        prompt = f"""
        你是保险合同信息提取专家。请从以下文本中提取信息：

        已知信息（规则提取结果）：
        {json.dumps(hints, ensure_ascii=False)}

        文本内容：
        {content} 

        请提取以下字段并以JSON格式返回，不要携带 "```json" 和 "```" 标签：
        {{
            "total_premium": {{'总保费': ''}},
            "coverage_amount": {{'保险金额': ''}},
            "confidence": 0.95
        }}

        要求：
        1. 如果字段不确定，设为 null
        2. 金额必须转为数字
        3. 日期统一为 YYYY-MM-DD 格式
        4. 为每个字段评估置信度
        """

        reasoning, content, tokens = self.llm.generate(prompt=prompt, temperature=0)
        return {"content": content, "reasoning": reasoning, "tokens": tokens}
