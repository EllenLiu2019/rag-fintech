import json

class LLMExtractor:
    """
    基于大模型的智能提取
    适用于：关系推理、条款理解、复杂约定
    """
    
    # def __init__(self, client, prompt_manager):
    #     self.client = client
    #     self.prompts = prompt_manager
    
    def extract(self, content: str, hints: dict = None) -> dict:
        """
        使用 Function Calling / Structured Output
        """

        prompt = f"""
        你是保险合同信息提取专家。请从以下文本中提取信息：

        已知信息（规则提取结果）：
        {json.dumps(hints, ensure_ascii=False)}

        文本内容：
        {content[:2000]}  # 截断长文本

        请提取以下字段并以JSON格式返回：
        {{
        "policy_holder": {{"name": "", "id_number": "", ...}},
        "insured": {{"name": "", "relationship_to_holder": "", ...}},
        "coverages": [...],
        "confidence": 0.95
        }}

        要求：
        1. 如果字段不确定，设为 null
        2. 金额必须转为数字
        3. 日期统一为 YYYY-MM-DD 格式
        4. 为每个字段评估置信度
        """
        
        response = self.client.predict(prompt)
        result = {
            "value": response.parsed_value,
            "confidence": 0.85, # LLM 默认置信度通常设为中高
            "source": "llm"
        }
        return result
    