import os
import json
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable
from langchain.agents import create_agent

from agent.tools import (
    align_medical_concepts,
    extract_content,
)
from common import get_logger, get_model_registry
from agent.entity import MedicalEntity


logger = get_logger(__name__)


SYSTEM_PROMPT = """
你是一名专业的医疗理赔编码专家。你的目标是**调用工具**将输入的医疗实体标准化，并计算分期。

## 要求
1. 如果满足工具调用条件，请**确保**调用以下工具，**不要自行计算或推断**：
    - `align_medical_concepts`

## 工作流程
请务必确保调用 `align_medical_concepts` 工具对上一步的结果进行匹配。
- 如果匹配：将合理的且最短路径的匹配结果作为最佳匹配结果，填充到 `best_matched_concept` 字段中。
- 如果不匹配或匹配结果与实体描述明显不符，设置 `human_in_the_loop` 字段为 True，并在 `reasoning` 中说明原因。

## 输出: 
请直接输出合法 JSON 字符串，不要携带 "```json" 和 "```" 标签，不要输出任何其他内容。示例如下：
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
    "reasoning": "Reasoning for the output",
    "human_in_the_loop": True # if the output is not confident, set to True, otherwise set to False
}   
"""


class MedicalAlignmentAgent:
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
            "align_medical_concepts": align_medical_concepts,
        }
        model = configurable_model.bind_tools(self.tools.values(), strict=True)

        self.agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=self.tools.values(),
            context_schema=MedicalEntity,
            debug=True,
        )

    @traceable(run_type="llm", name="MedicalAlignmentAgent.run")
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
        Extract best_matched_concept from agent result and enrich medical_entity
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
            alignment: dict[str, Any] | None = None
            if aligned_concept:
                alignment = {
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
            # Initialize agent_reasoning if None
            if medical_entity.agent_reasoning is None:
                medical_entity.agent_reasoning = {}

            medical_entity.agent_reasoning["alignment"] = {
                "alignment": alignment or {},
                "reasoning": medical_response_data.get("reasoning", ""),
                "human_in_the_loop": medical_response_data.get("human_in_the_loop", False),
            }
            logger.info(f"Filled medical entity from agent response: alignment={bool(alignment)}")
        except Exception as e:
            logger.warning(
                f"Failed to fill medical entity: {e}, medical_response_data: {medical_response_data}, aligned_concept: {aligned_concept}"
            )


if __name__ == "__main__":
    import json
    import asyncio
    import time
    from agent.medical_alignment_agent import MedicalAlignmentAgent
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
        agent_reasoning={
            "icd10_concepts": [
                {
                    "concept_id": 1406476,
                    "concept_code": "C73.00",
                    "concept_name": "甲状腺乳头状癌",
                    "score": 0.9795809984207153,
                },
            ],
            "snomed_concepts": [
                {
                    "concept_id": 4116228,
                    "concept_name": "Papillary thyroid carcinoma",
                    "concept_code": "255029007",
                    "score": 0.9977038502693176,
                },
                {
                    "concept_id": 37165586,
                    "concept_name": "Primary papillary thyroid carcinoma",
                    "concept_code": "1255191007",
                    "score": 0.9842970371246338,
                },
            ],
            "encode_reasoning": "用户输入的实体是'甲状腺乳头状癌'（papillary thyroid carcinoma），属于诊断类型。SNOMED编码库中有直接匹配的'Papillary thyroid carcinoma'（255029007）和'Primary papillary thyroid carcinoma'（1255191007），置信度均高于0.98，因此选用。ICD-10-CN编码库中没有专门的'甲状腺乳头状癌'编码，但'甲状腺恶性肿瘤'（C73.x00）是最接近的编码，置信度0.9796，符合甲状腺乳头状癌属于恶性肿瘤的分类。其他ICD-10-CN概念如'甲状腺原位癌'（D09.301）与恶性肿瘤性质不符，故不选用。",
        },
        description="甲状腺乳头状癌，肿瘤位置: 右叶下极，肿瘤大小: 1.2 cm × 1.0 cm，被膜侵犯: (-)，脉管侵犯: (-)，神经侵犯: (-)，中央区淋巴结未见癌转移 (0/6).",
    )
    agent = MedicalAlignmentAgent()

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
