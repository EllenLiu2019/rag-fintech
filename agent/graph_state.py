from dataclasses import dataclass, field
from enum import Enum
from typing_extensions import Annotated
from typing import Any
from pydantic import BaseModel, Field
from langchain.messages import AnyMessage
from langgraph.graph import END, START
from langgraph.graph.message import add_messages


class Step(Enum):
    START = START
    ENCODE_AGENT = "encode_agent"
    ALIGN_AGENT = "align_agent"
    STAGE_AGENT = "stage_agent"
    TOOL_NODE = "tool_node"
    APPROVAL = "approval"
    END = END


class AgentOutput(BaseModel):
    name: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    agent_response: dict[Any, Any] = Field(default_factory=dict)
    step_output: Any = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def update(o1: dict[str, AgentOutput], o2: dict[str, AgentOutput]) -> dict[str, AgentOutput]:
    output = o1.copy()
    output.update(o2)
    return output


class HumanDecision(BaseModel):
    icd_concept_code: str = ""
    icd_concept_name: str = ""
    tnm_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def merge_decision(d1: HumanDecision, d2: HumanDecision) -> HumanDecision:
    """Merge two HumanDecision objects. Newer value (d2) takes priority if non-empty.

    This supports both:
    - Normal flow: both subgraphs get the same decision, merge(d, d) = d
    - Time-travel: propagated decision (d2) overrides the old one (d1)
    """
    return HumanDecision(
        icd_concept_code=d2.icd_concept_code if d2.icd_concept_code else d1.icd_concept_code,
        icd_concept_name=d2.icd_concept_name if d2.icd_concept_name else d1.icd_concept_name,
        tnm_stage=d2.tnm_stage if d2.tnm_stage else d1.tnm_stage,
    )


@dataclass
class MedicalState:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    agent_output_dict: Annotated[dict[str, AgentOutput], update] = field(default_factory=dict)
    human_decision: Annotated[HumanDecision, merge_decision] = field(default_factory=lambda: HumanDecision())

    def last_message(self) -> AnyMessage | None:
        return self.messages[-1] if self.messages else None
