import json

from langsmith import traceable

from common import get_logger
from agent.entity import MedicalEntity
from agent.medical_agent_planner import MedicalAgentPlanner
from agent.medical_encode_agent import MedicalEncodeAgent
from agent.medical_alignment_agent import MedicalAlignmentAgent
from agent.medical_stage_agent import MedicalStageAgent

logger = get_logger(__name__)


AGENTS_REGISTRY = {
    "search_medical_kb": MedicalEncodeAgent(),
    "align_medical_concepts": MedicalAlignmentAgent(),
    "calculate_thyroid_tnm_stage": MedicalStageAgent(),
}


class MedicalAgent:

    @traceable(run_type="chain", name="MedicalAgent.run")
    async def run(self, medical_entity: MedicalEntity):
        planner = MedicalAgentPlanner()
        steps = await planner.run(medical_entity)
        for step in steps:
            await AGENTS_REGISTRY[step].run(medical_entity)


if __name__ == "__main__":
    import asyncio
    import time
    from agent.medical_agent import MedicalAgent
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
    agent = MedicalAgent()

    start = time.time()
    asyncio.run(agent.run(medical_entity))
    print(f"result gotten in seconds: {time.time() - start}")
    print("-" * 40)
    print(f"medical_entity: {json.dumps(medical_entity.to_dict(), ensure_ascii=False, indent=2)}")
