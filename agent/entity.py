from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field


class ClaimStatus(Enum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    PARTIAL = "partial"
    NEED_MORE_INFO = "need_info"
    UNDER_REVIEW = "under_review"


class MedicalEntity(BaseModel):
    """Custom runtime context schema."""

    entity_type: str  # diagnosis, procedure, symptom, medication
    patient_age: int
    term_cn: str
    term_en: str
    icd10_concepts: Dict[int, dict[str, Any]] | None = Field(default=None, description="ICD-10 concepts")  # ICD-10
    snomed_concepts: Dict[int, dict[str, Any]] | None = Field(default=None, description="SNOMED concepts")  # SNOMED
    attributes: Dict[str, Any] = Field(
        ..., description="TNM & max_diameter_cm & is_lumph"
    )  # TNM & max_diameter_cm & is_lumph
    agent_reasoning: Dict[str, Dict[str, Any]] | None = Field(default=None, description="Agent reasoning")
    description: str = Field(default="", description="Description")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "patient_age": self.patient_age,
            "term_cn": self.term_cn,
            "term_en": self.term_en,
            "snomed_concepts": self.snomed_concepts,
            "icd10_concepts": self.icd10_concepts,
            "attributes": self.attributes,
            "agent_reasoning": self.agent_reasoning,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MedicalEntity":
        return cls(
            entity_type=data.get("entity_type", ""),
            patient_age=data.get("patient_age", 0),
            term_cn=data.get("term_cn", ""),
            term_en=data.get("term_en", ""),
            snomed_concepts=data.get("snomed_concepts", []),
            icd10_concepts=data.get("icd10_concepts", []),
            attributes=data.get("attributes", {}),
            agent_reasoning=data.get("agent_reasoning", {}),
            description=data.get("description", ""),
        )


@dataclass
class ClaimRequest:
    patient_id: str
    patient_age: int
    policy_doc_id: str  # 保单文档ID
    medical_entities: List[MedicalEntity]  # 患者医疗实体
    claim_type: str  # medical
    claim_date: str
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "patient_age": self.patient_age,
            "policy_doc_id": self.policy_doc_id,
            "medical_entities": [entity.to_dict() for entity in self.medical_entities],
            "claim_type": self.claim_type,
            "claim_date": self.claim_date,
            "additional_info": self.additional_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimRequest":
        return cls(
            patient_id=data.get("patient_id", ""),
            patient_age=data.get("patient_age", 0),
            policy_doc_id=data.get("policy_doc_id", ""),
            medical_entities=[MedicalEntity.from_dict(entity) for entity in data.get("medical_entities", [])],
            claim_type=data.get("claim_type", ""),
            claim_date=data.get("claim_date", ""),
            additional_info=data.get("additional_info", {}),
        )


@dataclass
class ClaimDecision:
    status: ClaimStatus
    eligible_items: List[Dict[str, Any]]
    excluded_items: List[Dict[str, Any]]
    matched_clauses: List[Dict[str, Any]]
    explanation: str
    recommendations: List[str]
    reasoning: str
    tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "eligible_items": self.eligible_items,
            "excluded_items": self.excluded_items,
            "matched_clauses": self.matched_clauses,
            "explanation": self.explanation,
            "recommendations": self.recommendations,
            "reasoning": self.reasoning,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimDecision":
        return cls(
            status=ClaimStatus(data.get("status", "under_review")),
            eligible_items=data.get("eligible_items", []),
            excluded_items=data.get("excluded_items", []),
            matched_clauses=data.get("matched_clauses", []),
            explanation=data.get("explanation", ""),
            recommendations=data.get("recommendations", []),
            reasoning=data.get("reasoning", ""),
            tokens=data.get("tokens", 0),
        )
