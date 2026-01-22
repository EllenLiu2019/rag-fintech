import asyncio
from typing import Dict, Any
import re

from common import get_logger
from common.constants import VECTOR_SNOMED_KB
from rag.embedding import dense_embedder
from repository.vector import vector_store
from agent.entity import MedicalEntity
from repository.graph import neo4j_client

logger = get_logger(__name__)

TNM_PATTERN_REGEX = re.compile(r"[pc]?T([0-4][a-c]?|is|a|x)N([0-3][a-c]?|x)M([01][a-c]?|x)", re.IGNORECASE)


class MedicalNormalizer:

    SELECT_FIELDS = ["concept_id", "concept_name", "concept_code", "domain_id", "concept_class_id"]

    async def normalize(self, entity: MedicalEntity) -> None:
        """normalize a medical entity"""

        await self._search_icd10_snomed(entity)
        await self._align_icd10_snomed(entity)

        if entity.entity_type == "diagnosis":
            await self._extract_tnm_stage(entity)

    async def _search_icd10_snomed(self, entity: MedicalEntity) -> None:
        """search ICD-10 and SNOMED standard concepts"""

        # Domain mapping: entity_type -> SNOMED domain
        domain_mapping = {
            "diagnosis": "Condition",
            "procedure": "Procedure",
            "medication": "Drug",
            "symptom": "Observation",
        }

        # Embed query
        terms = [entity.term_cn, entity.term_en]
        embeddings = await asyncio.to_thread(dense_embedder.embed_queries_batch, terms)

        # Build filter for domain
        domain = domain_mapping.get(entity.entity_type, "")

        icd_results = await asyncio.to_thread(
            vector_store.search,
            selectFields=self.SELECT_FIELDS,
            dense_vectors=[embeddings[0]],
            limit=2,
            knowledgebaseIds=[VECTOR_SNOMED_KB],
            filters=f'vocabulary_id == "ICD10CN" and domain_id == "{domain}" and concept_class_id == "ICD10 code"',
        )

        if icd_results and icd_results[0]:
            for icd in icd_results[0]:
                if icd["score"] > 0.9:
                    entity.icd10_concepts.append(
                        {
                            "concept_id": int(icd["concept_id"]),
                            "concept_name": icd["concept_name"],
                            "concept_code": icd["concept_code"],
                        }
                    )

        snomed_results = await asyncio.to_thread(
            vector_store.search,
            selectFields=self.SELECT_FIELDS,
            dense_vectors=[embeddings[1]],
            limit=2,
            knowledgebaseIds=[VECTOR_SNOMED_KB],
            filters=f'vocabulary_id == "SNOMED" and domain_id == "{domain}"',
        )

        if snomed_results and snomed_results[0]:
            for snomed in snomed_results[0]:
                if snomed["score"] > 0.9:
                    entity.snomed_concepts.append(
                        {
                            "concept_id": int(snomed["concept_id"]),
                            "concept_name": snomed["concept_name"],
                            "concept_code": snomed["concept_code"],
                        }
                    )

    async def _align_icd10_snomed(self, entity: MedicalEntity) -> None:
        """align ICD-10 and SNOMED standard concepts"""
        icd10cn_ids: list[int] = [icd["concept_id"] for icd in entity.icd10_concepts]
        snomed_ids: list[int] = [snomed["concept_id"] for snomed in entity.snomed_concepts]
        aligned_concepts = []

        direct_query = """
            MATCH (icd:Concept)-[:MAPS_TO]->(snomed:Concept)
            WHERE icd.id = $icd10cn_id AND snomed.id = $snomed_id
            RETURN icd.id AS icd_id,
                icd.FSN AS icd_name,
                snomed.id AS snomed_id,
                snomed.FSN AS snomed_name
            ORDER BY icd.id
            """
        for icd10cn_id in icd10cn_ids:
            for snomed_id in snomed_ids:
                results = await asyncio.to_thread(
                    neo4j_client.execute_query,
                    direct_query,
                    {"icd10cn_id": icd10cn_id, "snomed_id": snomed_id},
                )
                for result in results:
                    aligned_concepts.append(
                        {
                            "path_length": 1,
                            "icd_id": result["icd_id"],
                            "icd_name": result["icd_name"],
                            "mapped_snomed_id": result["snomed_id"],
                            "mapped_snomed_name": result["snomed_name"],
                            "target_snomed_id": result["snomed_id"],
                            "target_snomed_name": result["snomed_name"],
                            "rel_types": ["MAPS_TO"],
                        }
                    )

        if not aligned_concepts:
            query = """
                MATCH path = (icd:Concept)-[:MAPS_TO]->(snomed1:Concept)-[:ISA*1..5]-(snomed2:Concept)
                WHERE icd.id = $icd10cn_id
                AND snomed2.id = $snomed_id
                RETURN icd.id AS icd_id,
                    icd.FSN AS icd_name,
                    snomed1.id AS mapped_snomed_id,
                    snomed1.FSN AS mapped_snomed_name,
                    snomed2.id AS target_snomed_id,
                    snomed2.FSN AS target_snomed_name,
                    length(path) AS path_length,
                    [rel in relationships(path) | type(rel)] AS rel_types
                ORDER BY path_length
                LIMIT 1
                """

            for icd10cn_id in icd10cn_ids:
                for snomed_id in snomed_ids:
                    results = await asyncio.to_thread(
                        neo4j_client.execute_query,
                        query,
                        {"icd10cn_id": icd10cn_id, "snomed_id": snomed_id},
                    )
                    for result in results:
                        aligned_concepts.append(
                            {
                                "path_length": int(result["path_length"]),
                                "icd_id": result["icd_id"],
                                "icd_name": result["icd_name"],
                                "mapped_snomed_id": result["mapped_snomed_id"],
                                "mapped_snomed_name": result["mapped_snomed_name"],
                                "target_snomed_id": result["target_snomed_id"],
                                "target_snomed_name": result["target_snomed_name"],
                                "rel_types": result["rel_types"],
                            }
                        )

        logger.info(f"aligned {len(aligned_concepts)} concepts")
        if aligned_concepts:
            entity.aligned_concept = sorted(aligned_concepts, key=lambda x: x["path_length"])[0]

    async def _extract_tnm_stage(self, entity: MedicalEntity) -> Dict[str, Any]:
        """
        input: "pT1bN0Mx" or is_lymph_metastasis or tumor_max_diameter_cm
        output: {"tnm_stage": "I期"}
        """
        tnm_stage = entity.attributes["tnm_stage_code"]
        if tnm_stage:
            match = TNM_PATTERN_REGEX.match(tnm_stage)
            if match:
                entity.attributes["tnm_stage"] = self._calculate_stage(match.group(1), match.group(2), match.group(3))

        is_lymph_metastasis: bool = entity.attributes.get("is_lymph_metastasis", False)
        tumor_max_diameter_cm: float = entity.attributes.get("tumor_max_diameter_cm")
        if is_lymph_metastasis:
            entity.attributes["tnm_stage"] = "II期"
        elif tumor_max_diameter_cm:
            if tumor_max_diameter_cm <= 4.0:
                entity.attributes["tnm_stage"] = "I期"

    def _calculate_stage(self, t: str, n: str, m: str) -> str:
        t_num = t[0] if t else "0"
        n_num = n[0] if n else "0"
        m_num = m[0] if m else "0"

        if m_num == "1":
            return "IV期"
        if n_num in ["2", "3"]:
            return "III期"
        if n_num == "1":
            return "II期"
        if t_num in ["1", "2", "3", "4"]:
            return "I期"
        return "0期"
