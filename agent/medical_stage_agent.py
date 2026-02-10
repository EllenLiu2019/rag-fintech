from langchain.chat_models.base import _ConfigurableModel
from langgraph.types import interrupt

from agent.medical_base_agent import BaseMedicalAgent
from agent.tools import calculate_thyroid_tnm_stage
from common import get_logger, get_prompt_manager
from agent.graph_state import MedicalState, AgentOutput, Step
from agent.entity import MedicalEntity


logger = get_logger(__name__)


class MedicalStageAgent(BaseMedicalAgent):
    def __init__(self, configurable_model: _ConfigurableModel):

        prompt = get_prompt_manager().get("medical_stage")
        super().__init__(
            configurable_model=configurable_model,
            prompt=prompt,
            tools={"calculate_thyroid_tnm_stage": calculate_thyroid_tnm_stage},
            logger=logger,
            tag=Step.STAGE_AGENT.value,
        )

    @property
    def step(self):
        from agent.graph_state import Step

        return Step.STAGE_AGENT

    def _post_process(self, response: dict, state: MedicalState, context: MedicalEntity):
        stage_agent_output: AgentOutput = state.agent_output_dict[self.step.value]
        tnm_stage = stage_agent_output.tool_calls[0].get("output", "")

        stage_agent_output.agent_response = response
        stage_agent_output.step_output = tnm_stage

        logger.info(f"Filled medical entity from agent response: tnm_stage={tnm_stage}")

    def approval_node(self, state: MedicalState) -> MedicalState:
        info = {
            "graph_name": self.step.value,
            "question": "please confirm the tnm_stage",
            "agent_output": state.agent_output_dict,
        }
        human_decision = interrupt(info)

        return MedicalState(
            messages=[],
            agent_output_dict=state.agent_output_dict,
            human_decision=human_decision,
        )
