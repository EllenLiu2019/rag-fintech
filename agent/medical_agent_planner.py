import os
import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable
from langchain.agents import create_agent

from common import get_logger, get_model_registry
from agent.entity import MedicalEntity


logger = get_logger(__name__)


PLANNER_SYSTEM_PROMPT = """
你是一名专业的医疗理赔团队Leader，你有一些医疗理赔编码专家作为你的下属，你的目标是**决定以怎样的顺序分配任务给这些下属**。

## 任务描述
你将得到一个**医疗实体**，你的团队需要对该医疗实体进行标准化（即找到正确的 ICD10CN 和 SNOMED 编码），并计算分期。

## 医疗理赔编码专家
1. search_medical_kb：擅长根据实体描述，搜索 ICD10CN 和 SNOMED 库，找到置信度较高的多个 ICD10CN 和 SNOMED 编码描述，以便后续进行匹配。
2. align_medical_concepts：擅长根据实体描述，从多个 ICD10CN 和 SNOMED 描述中，找到最高置信度且合理的1个或多个结果，验证 ICD10CN 和 SNOMED 描述是否匹配，如果匹配，返回最佳匹配结果。
3. calculate_thyroid_tnm_stage：擅长根据患者年龄、肿瘤大小、是否淋巴结转移等关键信息，计算恶性肿瘤的 TNM 分期。

## 输出
请直接输出合法 JSON 字符串，不要携带 "```json" 和 "```" 标签，不要输出任何其他内容。示例如下：
{{
    "steps": ["expert_1", "expert_2", "expert_3"]
}}
"""

PLANNER_USER_PROMPT = """
请规划使用医疗理赔专家下属处理医疗实体的步骤，并返回规划的医疗理赔专家下属列表。

## 医疗实体
{medical_entity}
"""


class MedicalAgentPlanner:
    def __init__(self):
        model_config = get_model_registry().get_chat_model("qwen_32B").to_dict()
        model = init_chat_model(
            model=model_config["model_name"],
            model_provider=model_config["provider"],
            base_url=model_config["base_url"],
            api_key=os.getenv(f"{model_config['provider'].upper()}_API_KEY", "EMPTY"),
            temperature=0.0,
        )
        self.agent = create_agent(
            model=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            debug=True,
        )

    @traceable(run_type="llm", name="MedicalAgentPlanner.run")
    async def run(self, medical_entity: MedicalEntity) -> list[str]:
        inputs = {
            "messages": [
                HumanMessage(
                    content=PLANNER_USER_PROMPT.format(
                        medical_entity=json.dumps(medical_entity.to_dict(), ensure_ascii=False, indent=2)
                    )
                )
            ]
        }
        results = await self.agent.ainvoke(inputs)
        return self._extract_steps(results)

    def _extract_steps(self, results: dict) -> list[str]:
        try:
            for message in reversed(results.get("messages", [])):
                if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                    content = json.loads(message.content.replace("```json", "").replace("```", ""))
                    if content.get("steps"):
                        return content["steps"]
        except Exception as e:
            logger.warning(f"Failed to extract steps: {e}, results: {results}")
            return []


if __name__ == "__main__":
    import asyncio
    import time
    from agent.medical_agent_planner import MedicalAgentPlanner
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
    agent_planner = MedicalAgentPlanner()

    start = time.time()
    results = asyncio.run(agent_planner.run(medical_entity))
    print(f"result gotten in seconds: {time.time() - start}")
    print(f"steps: {results}")
