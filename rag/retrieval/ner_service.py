from typing import List, Dict, Any
from transformers import pipeline

from common import get_logger

logger = get_logger(__name__)


class NERService:

    GROUP2SNOMED = {
        "SIGN_SYMPTOM": ("Condition; Observation", "Clinical Finding; Morphologic Abnormality"),
        "DISEASE_DISORDER": ("Condition", "Disorder; Clinical Finding"),
        "MEDICATION": ("Drug", "Pharmaceutical; biologic product; Substance; Clinical Drug"),
        "DIAGNOSTIC_PROCEDURE": ("Procedure", "Procedure"),
        "THERAPEUTIC_PROCEDURE": ("Procedure", "Procedure"),
        "BIOLOGICAL_STRUCTURE": ("Specimen; Spec Anatomic Site", "Body structure"),
        "DETAILED_DESCRIPTION": ("disease", "Disorder"),
    }
    COMBINED_GROUPS = ["SIGN_SYMPTOM", "DISEASE_DISORDER", "BIOLOGICAL_STRUCTURE"]

    def __init__(self):
        self.pipe = pipeline(
            task="token-classification",
            model="Clinical-AI-Apollo/Medical-NER",
            aggregation_strategy="simple",
        )
        self.entities: Dict[str, List[Dict[str, Any]]] = {}

    def get_entities(self, text: str) -> List[Dict[str, Any]]:
        if self.entities.get(text):
            return self.entities.get(text)
        self.extract_entities(text)
        return self.entities.get(text, [])

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        try:
            results = self.pipe(text)
            for ent in results:
                snomed_concept = self.GROUP2SNOMED.get(ent["entity_group"])
                if snomed_concept:
                    ent["domain_id"] = snomed_concept[0]
                    ent["concept_class_id"] = snomed_concept[1]
                    self.entities.setdefault(text, []).append(ent)
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")

    def combine_entities(self, entities: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        combined_results = []
        skip_next = False
        for i, entity in enumerate(entities):
            if skip_next:
                skip_next = False
                continue

            if i < len(entities) - 1 and self._should_combine(entity, entities[i + 1]):
                entities[i + 1]["score"] = float(entities[i + 1]["score"])
                combined_results.append(self._create_combined_entity(entity, entities[i + 1], query))
                skip_next = True
                continue
            combined_results.append(entity)

        return self._remove_overlapping_entities(combined_results)

    def _should_combine(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        return (
            entity1["entity_group"] in ["BIOLOGICAL_STRUCTURE", "DETAILED_DESCRIPTION"]
            and entity2["entity_group"] in ["SIGN_SYMPTOM", "DISEASE_DISORDER"]
            and entity1["end"] == entity2["start"]
        )

    def _create_combined_entity(self, entity1: Dict[str, Any], entity2: Dict[str, Any], text: str) -> Dict[str, Any]:
        start = min(entity1["start"], entity2["start"])
        end = max(entity1["end"], entity2["end"])
        word = text[start:end]
        return {
            "entity_group": "COMBINED_BIO_SYMPTOM",
            "word": word,
            "start": start,
            "end": end,
            "score": (entity1["score"] + entity2["score"]) / 2,
            "original_entities": [entity1, entity2],
        }

    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        remove overlapping entities, keep the highest score entity
        """
        if not entities:
            return []

        # sort by score descending, keep the highest score entity
        sorted_by_score = sorted(entities, key=lambda x: -x["score"])

        non_overlapping = []
        for entity in sorted_by_score:
            # check if the entity overlaps with the already selected entities
            overlaps = any(
                not (entity["end"] <= kept["start"] or entity["start"] >= kept["end"]) for kept in non_overlapping
            )
            if not overlaps:
                non_overlapping.append(entity)

        # sort by position and return
        return sorted(non_overlapping, key=lambda x: x["start"])


def _create_ner_service() -> NERService:
    ner_service = NERService()
    logger.info("Initialized NERService singleton")
    return ner_service


# ner_service = _create_ner_service()
ner_service = None
