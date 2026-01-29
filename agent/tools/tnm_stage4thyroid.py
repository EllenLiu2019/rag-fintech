from langchain.tools import tool
from pydantic import BaseModel, Field
from enum import Enum


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
async def calculate_thyroid_tnm_stage(
    age: int,
    t_stage_raw: str,
    n_stage_raw: str,
    m_stage_raw: str,
    histology_type: HistologyType,
) -> str:
    """Determine the TNM staging classification for thyroid cancer in accordance with the American Joint Committee on Cancer (AJCC) Eighth Edition guidelines.

    This tool accepts TNM information and calculates the TNM stage according to the AJCC 8th edition.
    Returns the TNM stage description.
    """

    t = t_stage_raw.replace("p", "").replace("c", "").lower()
    n = n_stage_raw.replace("p", "").replace("c", "").lower()
    m = m_stage_raw.replace("p", "").replace("c", "").lower()

    if histology_type in [HistologyType.PAPILLARY, HistologyType.FOLLICULAR]:
        if age < 55:
            if "m1" in m:
                return "II期"
            return "I期"
        else:
            if "m1" in m:
                return "IVB期"
            if "t4b" in t:
                return "IVA期"
            if "t4a" in t:
                return "III期"
            if "t3" in t or "n1" in n:
                return "II期"
            return "I期"

    elif histology_type == HistologyType.MEDULLARY:
        if "m1" in m:
            return "IVC期"
        if "t4b" in t:
            return "IVB期"
        if "t4a" in t or "n1b" in n:
            return "IVA期"
        if "n1a" in n:
            return "III期"
        if "t2" in t or "t3" in t:
            return "II期"
        return "I期"

    elif histology_type == HistologyType.ANAPLASTIC:
        if "m1" in m:
            return "IVC期"
        if "t4" in t or "n1" in n:
            return "IVB期"
        return "IVA期"

    return "无法判定"
