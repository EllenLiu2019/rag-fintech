from langchain.chat_models.base import _ConfigurableModel

from agent.medical_base_agent import BaseMedicalAgent
from common import get_logger, get_prompt_manager
from agent.graph_state import AgentOutput, MedicalState, Step
from agent.entity import MedicalEntity


logger = get_logger(__name__)


class MedicalEncodeAgent(BaseMedicalAgent):
    def __init__(self, configurable_model: _ConfigurableModel):

        prompt = get_prompt_manager().get("medical_encode")
        super().__init__(
            configurable_model=configurable_model,
            prompt=prompt,
            tools={},
            logger=logger,
            tag=Step.ENCODE_AGENT.value,
        )

    @property
    def step(self):
        from agent.graph_state import Step

        return Step.ENCODE_AGENT

    def _post_process(self, response: dict, state: MedicalState, context: MedicalEntity):
        medical_entity = context
        agent_output_dict = state.agent_output_dict

        icd10_concepts_ids = response.get("icd10_concepts", [])
        snomed_concepts_ids = response.get("snomed_concepts", [])

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

        agent_output_dict[self.step.value] = AgentOutput(
            name=self.step.value,
            tool_calls=[],
            agent_response=response,
            step_output={
                "icd10_concepts": icd10_concepts,
                "snomed_concepts": snomed_concepts,
            },
        )

        logger.info(
            "Filled medical entity from agent response: "
            f"icd10_concepts={bool(icd10_concepts)}, snomed_concepts={bool(snomed_concepts)}"
        )
