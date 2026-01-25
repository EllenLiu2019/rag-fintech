import os
import json
from typing import Any
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable
from langchain.agents import create_agent

from agent.tools import (
    search_medical_kb,
    align_medical_concepts,
    calculate_thyroid_tnm_stage,
    extract_content,
)
from common import get_logger, get_model_registry
from agent.entity import MedicalEntity


logger = get_logger(__name__)


SYSTEM_PROMPT = """
你是一名专业的医疗理赔编码专家。你的目标是将输入的医疗实体标准化，并计算分期。

**工作流程：**
1. 分析用户输入的实体类型，如果实体类型为 diagnosis，分别搜索 ICD10CN 和 SNOMED 库。
2. 请分别从返回的 ICD10CN 和 SNOMED 结果中找到最高置信度的1个或多个结果，调用 `align_medical_concepts` 工具，对高置信度且合理的 ICD & SNOMED concepts进行匹配。
   - 如果匹配：将合理的且较短路径的匹配结果作为最佳匹配结果，填充到 `best_matched_concept` 字段。
   - 如果不匹配：设置 `human_in_the_loop` 字段为 True，并在 `reasoning` 中说明原因并标记歧义。
3. 对于 TNM 分期，请根据“患者年龄”、“肿瘤大小”、“是否淋巴结转移”等关键信息，提取 T、N、M 分期、病理类型；
   - 若是甲状腺癌，则调用 `calculate_thyroid_tnm_stage` 工具，计算 TNM 分期。
   - 若是其他癌症，请回忆该类型癌症在 AJCC 8th edition 中的 TNM 分期规则，并计算 TNM 分期。
   - 将结果填充到返回结果的 `tnm_stage` 字段。
4. 如果搜索结果或匹配结果与实体描述明显不符，或者无法确定 TNM 分期，请标记为需要人工审核，设置 `human_in_the_loop` 字段为 True。
   
**输出:** 请直接输出合法 JSON 字符串，不要携带 "```json" 和 "```" 标签，不要输出任何其他内容。示例如下：
{
    "best_matched_concept": {
        "icd_id": "ICD10CN ID",
        "icd_concept_code": "ICD10CN Concept Code of ICD10CN",
        "icd_name": "ICD10CN Name of ICD10CN",
        "mapped_snomed_id": "SNOMED ID of mapped SNOMED",
        "mapped_snomed_name": "SNOMED Name of mapped SNOMED",
        "target_snomed_id": "SNOMED ID of target SNOMED",
        "target_snomed_concept_code": "SNOMED Concept Code of target SNOMED",
        "target_snomed_name": "SNOMED Name of target SNOMED",
        "path_length": 3,
        "rel_types": ["MAPS_TO", "ISA", ...] # relationship types between ICD10CN and Target SNOMED
    },
    "tnm_code": "TNM Code, e.g. T1N0M0",
    "tnm_stage": "TNM Stage, e.g. I期",
    "reasoning": "Reasoning for the output",
    "human_in_the_loop": True # if the output is not confident, set to True, otherwise set to False
}   
"""


@dataclass
class MedicalResponse:
    best_matched_concept: dict
    tnm_stage: str
    reasoning: str
    human_in_the_loop: bool


class MedicalAgent:
    def __init__(self):
        model_config = get_model_registry().get_chat_model("qa_reasoner").to_dict()
        model = init_chat_model(
            model_config["model_name"],
            model_provider=model_config["provider"],
            api_key=os.getenv(f"{model_config['provider'].upper()}_API_KEY"),
        )
        tools = [search_medical_kb, align_medical_concepts, calculate_thyroid_tnm_stage]
        self.agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            context_schema=MedicalEntity,
            debug=True,
        )

    @traceable(run_type="chain", name="MedicalAgent.run")
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
        Extract MedicalResponse from agent result and enrich medical_entity
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
            aligned_concept = medical_response_data.get("best_matched_concept", {})
            if aligned_concept:
                medical_entity.agent_reasoning["aligned_concept"] = {
                    "icd_id": aligned_concept.get("icd_id"),
                    "icd_concept_code": aligned_concept.get("icd_concept_code"),
                    "icd_name": aligned_concept.get("icd_name"),
                    "mapped_snomed_id": aligned_concept.get("mapped_snomed_id"),
                    "mapped_snomed_name": aligned_concept.get("mapped_snomed_name"),
                    "target_snomed_id": aligned_concept.get("target_snomed_id"),
                    "target_snomed_concept_code": aligned_concept.get("target_snomed_concept_code"),
                    "target_snomed_name": aligned_concept.get("target_snomed_name"),
                    "path_length": aligned_concept.get("path_length"),
                    "rel_types": aligned_concept.get("rel_types", []),
                }

            medical_entity.agent_reasoning["tnm_code"] = medical_response_data.get("tnm_code", "")
            medical_entity.agent_reasoning["tnm_stage"] = medical_response_data.get("tnm_stage", "")
            medical_entity.agent_reasoning["reasoning"] = medical_response_data.get("reasoning", "")
            medical_entity.agent_reasoning["human_in_the_loop"] = medical_response_data.get("human_in_the_loop", False)

            logger.info(
                f"Filled medical entity from agent response: aligned_concept={bool(aligned_concept)}, tnm_stage={medical_response_data.get('tnm_stage')}"
            )
        except Exception as e:
            logger.warning(f"Failed to fill medical entity: {e}, medical_response_data: {medical_response_data}")


if __name__ == "__main__":
    import json
    import asyncio
    import time
    from agent.medical_agent import MedicalAgent
    from agent.entity import MedicalEntity
    from langchain_core.messages.ai import AIMessage
    from langchain_core.messages.tool import ToolMessage

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
    agent = MedicalAgent()

    start = time.time()
    results = asyncio.run(agent.run(medical_entity))
    print(f"result gotten in seconds: {time.time() - start}")
    print("-" * 40)
    for message in results.get("messages"):
        if isinstance(message, AIMessage):
            print("AI", "-" * 40)
            print(f"response_metadata:\n {json.dumps(message.response_metadata, ensure_ascii=False, indent=2)}")
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"tool_call input:\n {json.dumps(tool_call, ensure_ascii=False, indent=2)}")
        elif isinstance(message, ToolMessage):
            print("Tool", "-" * 40)
            print(f"tool_call output:\n{message.content}")
    print("-" * 40)
    print(f"medical_entity: {json.dumps(medical_entity.to_dict(), ensure_ascii=False, indent=2)}")
