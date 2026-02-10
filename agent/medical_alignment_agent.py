from langchain.chat_models.base import _ConfigurableModel
from langgraph.types import interrupt

from agent.medical_base_agent import BaseMedicalAgent
from agent.tools import align_medical_concepts
from common import get_logger, get_prompt_manager
from agent.graph_state import MedicalState, AgentOutput, Step
from agent.entity import MedicalEntity

logger = get_logger(__name__)


class MedicalAlignmentAgent(BaseMedicalAgent):
    def __init__(
        self,
        configurable_model: _ConfigurableModel,
    ):
        prompt = get_prompt_manager().get("medical_alignment")
        super().__init__(
            configurable_model=configurable_model,
            prompt=prompt,
            tools={"align_medical_concepts": align_medical_concepts},
            logger=logger,
            tag=Step.ALIGN_AGENT.value,
        )
        self._approval_node = self._approval_node

    @property
    def step(self):
        from agent.graph_state import Step

        return Step.ALIGN_AGENT

    @property
    def approval_node(self):
        return self._approval_node

    def _post_process(self, response: dict, state: MedicalState, context: MedicalEntity):
        agent_output_dict = state.agent_output_dict

        align_agent_output: AgentOutput = agent_output_dict[self.step.value]
        aligned_concepts = align_agent_output.tool_calls[0].get("output", {})
        concept_key = response.get("best_matched_concept", "")
        best_matched_concept = aligned_concepts.get(concept_key) if concept_key else None

        align_agent_output.agent_response = response
        align_agent_output.step_output = best_matched_concept

        logger.info(f"Filled medical entity from agent response: best_matched_concept={bool(best_matched_concept)}")

    def _approval_node(self, state: MedicalState) -> MedicalState:
        info = {
            "graph_name": self.step.value,
            "question": "please confirm the icd_concept_code",
            "agent_output": state.agent_output_dict,
        }
        human_decision = interrupt(info)

        return MedicalState(
            messages=[],
            agent_output_dict=state.agent_output_dict,
            human_decision=human_decision,
        )
