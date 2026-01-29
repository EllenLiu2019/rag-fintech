import os
import json
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable
from langchain.agents import create_agent

from agent.tools import (
    calculate_thyroid_tnm_stage,
    extract_content,
)
from common import get_logger, get_model_registry
from agent.entity import MedicalEntity


logger = get_logger(__name__)


SYSTEM_PROMPT = """
你是一名专业的医疗理赔编码专家。你的目标是**调用工具**计算输入的医疗实体的 TNM 分期。


## 工作流程
1. 根据“患者年龄”、“肿瘤大小”、“是否淋巴结转移”等关键信息，提取 T、N、M 分期、病理类型；
    - 若是甲状腺癌，则**务必**调用 `calculate_thyroid_tnm_stage` 工具，计算 TNM 分期。
    - 若是其他癌症，请**务必**回忆该类型癌症在 AJCC 8th edition 中的 TNM 分期规则，并计算 TNM 分期。
2. 将结果填充到返回结果的 `tnm_stage` 字段。
   
## 输出: 
请直接输出合法 JSON 字符串，不要携带 "```json" 和 "```" 标签，不要输出任何其他内容。示例如下：
{
    "tnm_code": "TNM Code, e.g. T1N0M0",
    "tnm_stage": "TNM Stage, e.g. I期",
    "reasoning": "Reasoning for the output",
}   
"""


class MedicalStageAgent:
    def __init__(self):
        model_config = get_model_registry().get_chat_model("qwen_32B").to_dict()
        configurable_model = init_chat_model(
            model=model_config["model_name"],
            model_provider=model_config["provider"],
            base_url=model_config["base_url"],
            api_key=os.getenv(f"{model_config['provider'].upper()}_API_KEY", "EMPTY"),
            temperature=0.0,
        )
        self.tools = {
            "calculate_thyroid_tnm_stage": calculate_thyroid_tnm_stage,
        }
        model = configurable_model.bind_tools(self.tools.values(), strict=True)

        self.agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=self.tools.values(),
            context_schema=MedicalEntity,
            debug=True,
        )

    @traceable(run_type="llm", name="MedicalStageAgent.run")
    async def run(self, medical_entity: MedicalEntity):
        user_message = HumanMessage(
            content=f"请处理医疗实体: {json.dumps(medical_entity.to_dict(), ensure_ascii=False, indent=2)}"
        )
        inputs = {"messages": [user_message]}
        results = await self.agent.ainvoke(inputs, context=medical_entity)
        self._enrich_medical_entity(results, medical_entity)
        return results

    def _enrich_medical_entity(self, agent_result: dict[str, Any], medical_entity: MedicalEntity):
        """
        Extract tnm_stage from agent result and enrich medical_entity
        """
        medical_response_data = None
        for message in reversed(agent_result.get("messages", [])):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                # This is the final response without tool calls
                try:
                    medical_response_data = extract_content(message.content) if isinstance(message.content, str) else {}
                    if medical_response_data:
                        break
                except Exception as e:
                    logger.warning(f"Failed to extract JSON from message: {e}")

        if not medical_response_data:
            logger.warning("No medical response found in agent result")
            return

        try:
            if medical_entity.agent_reasoning is None:
                medical_entity.agent_reasoning = {}

            medical_entity.agent_reasoning["tnm_stage"] = {
                "tnm_code": medical_response_data.get("tnm_code", ""),
                "tnm_stage": medical_response_data.get("tnm_stage", ""),
                "reasoning": medical_response_data.get("reasoning", ""),
            }

            logger.info(
                f"Filled medical entity from agent response: tnm_code={medical_response_data.get('tnm_code')}, tnm_stage={medical_response_data.get('tnm_stage')}"
            )
        except Exception as e:
            logger.warning(f"Failed to fill medical entity: {e}, medical_response_data: {medical_response_data}")


if __name__ == "__main__":
    import json
    import asyncio
    import time
    from agent.medical_stage_agent import MedicalStageAgent
    from agent.entity import MedicalEntity

    medical_entity = MedicalEntity(
        patient_age=56,
        term_cn="甲状腺乳头状癌",
        term_en="papillary thyroid carcinoma",
        entity_type="diagnosis",
        attributes={
            "tumor_max_diameter_cm": 1.2,
            "is_lymph_metastasis": False,
        },
        description="甲状腺乳头状癌，肿瘤位置: 右叶下极，肿瘤大小: 1.2 cm × 1.0 cm，被膜侵犯: (-)，脉管侵犯: (-)，神经侵犯: (-)，中央区淋巴结未见癌转移 (0/6).",
    )
    agent = MedicalStageAgent()

    start = time.time()
    results = asyncio.run(agent.run(medical_entity))
    print(f"result gotten in seconds: {time.time() - start}")
    print(f"medical_entity: {json.dumps(medical_entity.to_dict(), ensure_ascii=False, indent=2)}")
