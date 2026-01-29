import os
import json
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable
from langchain.agents import create_agent

from agent.tools import (
    search_medical_kb,
    extract_content,
)
from common import get_logger, get_model_registry
from agent.entity import MedicalEntity


logger = get_logger(__name__)


SYSTEM_PROMPT = """
你是一名专业的医疗理赔编码专家。你的目标是**调用工具**将输入的医疗实体编码为 ICD10CN 和 SNOMED 编码，并将选择结果的序号填充到 `icd10_concepts` 和 `snomed_concepts` 字段中。

## 工作流程
1. 分析用户输入的实体类型，如果实体类型为 diagnosis，分别搜索 ICD10CN 和 SNOMED 库。
2. 从 ICD10CN 的搜索结果中找到置信度较高的1个或多个 ICD10CN 编码描述，将你选择的序号填充到 `icd10_concepts` 字段中。
3. 从 SNOMED 的搜索结果中找到置信度较高的1个或多个 SNOMED 编码描述，将你选择的序号填充到 `snomed_concepts` 字段中。
4. 填充 `reasoning` 字段说明原因。
   
## 示例：
    搜索结果如下：
    ICD10CN 搜索结果：
    {
        1234567: {
            "concept_id": 1234567,
            "concept_name": "ICD10CN Concept Name 1",
            "concept_code": "ICD10CN Concept Code 1",
            "score": 0.9,
        },
        6789087: {
            "concept_id": 6789087,
            "concept_name": "ICD10CN Concept Name 2",
            "concept_code": "ICD10CN Concept Code 2",
            "score": 0.9,
        },
        ...
    }
    - 对于 ICD10CN 搜索结果，你选择的为 1234567，则 `icd10_concepts` 字段为 1234567。

    SNOMED 搜索结果：
    {
        422666: {
            "concept_id": 422666,
            "concept_name": "SNOMED Concept Name 1",
            "concept_code": "SNOMED Concept Code 1",
            "score": 0.9,
        },
        678845: {
            "concept_id": 678845,
            "concept_name": "SNOMED Concept Name 2",
            "concept_code": "SNOMED Concept Code 2",
            "score": 0.9,
        },
        ...
    }
    
    - 对于 SNOMED 搜索结果，你选择的为 422666，则 `snomed_concepts` 字段为 422666。
    最后输出结果示例如下：
    {
        "icd10_concepts": [1234567],
        "snomed_concepts": [422666],
        "reasoning": "Reasoning for the output",
    }

## 输出: 
请直接输出合法 JSON 字符串，不要携带 "```json" 和 "```" 标签，不要输出任何其他内容。示例如下：
{
    "icd10_concepts": [1234567], # 置信度较高且合理的 ICD10CN 结果序列号
    "snomed_concepts": [422666], # 置信度较高且合理的 SNOMED 结果序列号
    "reasoning": "Reasoning for the output",
}   
"""


class MedicalEncodeAgent:
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
            "search_medical_kb": search_medical_kb,
        }
        model = configurable_model.bind_tools(self.tools.values(), strict=True)

        self.agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=self.tools.values(),
            debug=True,
        )

    @traceable(run_type="llm", name="MedicalEncodeAgent.run")
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
        Extract icd10_concepts and snomed_concepts from agent result and enrich medical_entity
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
            icd10_concepts_ids = medical_response_data.get("icd10_concepts", [])
            snomed_concepts_ids = medical_response_data.get("snomed_concepts", [])

            icd10_concepts = []
            snomed_concepts = []

            if icd10_concepts_ids and medical_entity.icd10_concepts:
                for concept_id in icd10_concepts_ids:
                    if concept_id in medical_entity.icd10_concepts:
                        icd10_concepts.append(medical_entity.icd10_concepts[concept_id])
                    else:
                        logger.warning(
                            f"Invalid icd10_concepts: {concept_id}, available indices: {list(medical_entity.icd10_concepts.keys()) if medical_entity.icd10_concepts else 'None'}"
                        )

            if snomed_concepts_ids and medical_entity.snomed_concepts:
                for concept_id in snomed_concepts_ids:
                    if concept_id in medical_entity.snomed_concepts:
                        snomed_concepts.append(medical_entity.snomed_concepts[concept_id])
                    else:
                        logger.warning(
                            f"Invalid snomed_index: {concept_id}, available indices: {list(medical_entity.snomed_concepts.keys()) if medical_entity.snomed_concepts else 'None'}"
                        )

            if medical_entity.agent_reasoning is None:
                medical_entity.agent_reasoning = {}

            medical_entity.agent_reasoning["encode"] = {
                "icd10_concepts": icd10_concepts,
                "snomed_concepts": snomed_concepts,
                "reasoning": medical_response_data.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning(f"Failed to fill medical entity: {e}, medical_response_data: {medical_response_data}")

        logger.info(
            f"Filled medical entity from agent response: icd10_concepts={bool(icd10_concepts)}, snomed_concepts={bool(snomed_concepts)}"
        )


if __name__ == "__main__":
    import json
    import asyncio
    import time
    from agent.medical_encode_agent import MedicalEncodeAgent
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
    agent = MedicalEncodeAgent()

    start = time.time()
    results = asyncio.run(agent.run(medical_entity))
    print(f"result gotten in seconds: {time.time() - start}")
    print("-" * 40)
    print(f"medical_entity: {json.dumps(medical_entity.to_dict(), ensure_ascii=False, indent=2)}")
