from langchain.tools import tool
from pydantic import BaseModel, Field
from enum import Enum

from langchain.tools import ToolRuntime


class HistologyType(Enum):
    PAPILLARY = "乳头状癌"
    FOLLICULAR = "滤泡状癌"
    MEDULLARY = "髓样癌"
    ANAPLASTIC = "未分化癌"


class TNMInfo(BaseModel):
    age: int = Field(..., description="Age of the patient")
    t_stage_raw: str = Field(..., description="T stage raw text, e.g. T0, Tx, T1a, T1b, T2, T3 etc.")
    n_stage_raw: str = Field(..., description="N stage raw text, e.g. N0, Nx, N1, N1a, N3 etc.")
    m_stage_raw: str = Field(..., description="M stage raw text, e.g. M0, M1.")
    histology_type: HistologyType = Field(..., description="Histology type")

    # thyroid_rules: dict[str, any] = {
    #     "乳头状或滤泡状癌": [
    #         {
    #             "age < 55": {
    #                 "I": {"M": "0"},
    #                 "II": {"M": "1"},
    #             },
    #             "age >= 55": {
    #                 "I": {"T": ["1", "2"], "N": ["0", "x"], "M": "0"},
    #                 "II": [
    #                     {"T": ["1", "2"], "N": ["1"], "M": "0"},
    #                     {"T": ["3", "3a", "3b"], "M": "0"},
    #                 ],
    #                 "III": {"T": ["4a"], "M": "0"},
    #                 "IVA": {"T": ["4b"], "M": "0"},
    #                 "IVB": {"M": "1"},
    #             },
    #         }
    #     ],
    #     "髓样癌": {
    #         "I": {"T": ["1"], "N": ["0"], "M": "0"},
    #         "II": {"T": ["2", "3"], "N": ["0"], "M": "0"},
    #         "III": {"T": ["1", "2", "3"], "N": ["1a"], "M": "0"},
    #         "IVA": [{"T": ["4a"], "M": "0"}, {"T": ["1", "2", "3"], "N": ["1b"], "M": "0"}],
    #         "IVB": {"T": ["4b"], "M": "0"},
    #         "IVC": {"M": "1"},
    #     },
    #     "未分化癌": {
    #         "IVA": {"T": ["1~3a"], "N": ["0", "x"], "M": "0"},
    #         "IVB": [{"T": ["1~3a"], "N": ["1"], "M": "0"}, {"T": ["3b~4"], "M": "0"}],
    #         "IVC": {"M": "1"},
    #     },
    # }


@tool(args_schema=TNMInfo)
def calculate_thyroid_tnm_stage(
    age: int,
    t_stage_raw: str,
    n_stage_raw: str,
    m_stage_raw: str,
    histology_type: HistologyType,
    runtime: ToolRuntime,
) -> str:
    """Determine the TNM staging classification for thyroid cancer in accordance with the American Joint Committee on Cancer (AJCC) Eighth Edition guidelines.

    This tool accepts TNM information and calculates the TNM stage according to the AJCC 8th edition.
    Returns the TNM stage description.
    """
    from agent.graph_state import AgentOutput, Step

    agent_output_dict: dict[str, AgentOutput] = runtime.state.agent_output_dict

    t = t_stage_raw.replace("p", "").replace("c", "").lower()
    n = n_stage_raw.replace("p", "").replace("c", "").lower()
    m = m_stage_raw.replace("p", "").replace("c", "").lower()

    # AJCC 8th edition: check from highest severity to lowest
    match histology_type:
        case HistologyType.PAPILLARY | HistologyType.FOLLICULAR:
            tnm_stage = _stage_papillary_follicular(age, t, n, m)
        case HistologyType.MEDULLARY:
            tnm_stage = _stage_medullary(t, n, m)
        case HistologyType.ANAPLASTIC:
            tnm_stage = _stage_anaplastic(t, n, m)
        case _:
            tnm_stage = "无法判定"

    tool_call = {
        "name": "calculate_thyroid_tnm_stage",
        "output": tnm_stage,
    }
    agent_output_dict[Step.STAGE_AGENT.value] = AgentOutput(
        name=Step.STAGE_AGENT.value,
        tool_calls=[tool_call],
        agent_response={},
        step_output=None,
    )

    return tnm_stage


def _stage_papillary_follicular(age: int, t: str, n: str, m: str) -> str:
    """Papillary / Follicular thyroid carcinoma staging (AJCC 8th edition)."""
    if age < 55:
        # Age < 55: only M status matters
        return "II期" if "m1" in m else "I期"
    else:
        # Age >= 55: check from highest severity to lowest
        if "m1" in m:
            return "IVB期"
        elif "t4b" in t:
            return "IVA期"
        elif "t4a" in t:
            return "III期"
        elif "t3" in t or "n1" in n:
            return "II期"
        else:
            return "I期"


def _stage_medullary(t: str, n: str, m: str) -> str:
    """Medullary thyroid carcinoma staging (AJCC 8th edition)."""
    if "m1" in m:
        return "IVC期"
    elif "t4b" in t:
        return "IVB期"
    elif "t4a" in t or "n1b" in n:
        return "IVA期"
    elif "n1a" in n:
        return "III期"
    elif "t2" in t or "t3" in t:
        return "II期"
    else:
        return "I期"


def _stage_anaplastic(t: str, n: str, m: str) -> str:
    """Anaplastic thyroid carcinoma staging (AJCC 8th edition)."""
    if "m1" in m:
        return "IVC期"
    elif "t4" in t or "n1" in n:
        return "IVB期"
    else:
        return "IVA期"
