import json
import re
from typing import Dict, Any, List, Literal, Optional

from rag.llm.chat_model import chat_model
from common import get_logger, get_model_registry
from common.prompt_manager import get_prompt_manager
from rag.core import embedder
from repository.vector.milvus_client import VectorStoreClient
from rag.retrieval.ner_service import ner_service
from repository.cache import cached

logger = get_logger(__name__)


class BaseRewriter:
    def __init__(self, model: dict[str, Any], temperature: float = 0.0):
        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )
        self.temperature = temperature
        self.prompt_manager = get_prompt_manager()

    def _build_prompt(self) -> str:
        pass

    def _clean_response(self, content: str) -> str:
        if not content:
            return ""

        result = content.strip()

        # Remove quotes if present
        if (result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]

        # Remove markdown code blocks if present
        if result.startswith("```") and result.endswith("```"):
            result = re.sub(r"```(?:\w+)?\n?", "", result).strip()

        # Remove "输出:" prefix if present
        if result.startswith("输出:") or result.startswith("输出："):
            result = result[3:].strip()

        return result

    def _clean_multi_response(self, content: str) -> List[str]:
        if not content:
            return []

        result = content.strip()
        results = result.split("\n")
        return [self._clean_response(result) for result in results]


class UnifiedRewriter(BaseRewriter):
    def __init__(
        self,
        model: dict[str, Any],
        temperature: float = 0.0,
        history_max_length: int = 10,
    ):
        super().__init__(model=model, temperature=temperature)
        self.history_max_length = history_max_length
        self.histories: List[str] = []

    def _build_prompt(self, required_entities: Optional[List[str]] = None) -> str:
        history_str = json.dumps(self.histories[-self.history_max_length :], ensure_ascii=False)

        # Build required entities section if entities exist
        if required_entities:
            entities_str = "、".join(required_entities)
            required_section = f"\n    ### 4. 必须包含的医疗词汇\n    以下词汇已通过 NER 识别，必须在改写结果中保留：【{entities_str}】\n"
        else:
            required_section = ""

        return self.prompt_manager.get("unified_rewrite", histories=history_str, required_entities=required_section)

    def rewrite(self, query: str, medical_entities: Optional[List[str]] = None) -> Dict[str, Any]:
        try:

            reasoning, content, tokens = self.llm.generate(
                messages=[
                    {"role": "system", "content": self._build_prompt(medical_entities)},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
            )

            rewritten = self._clean_response(content)

            # if medical_entities:
            #     missing = [ent for ent in medical_entities if ent not in rewritten]
            #     if missing:
            #         logger.warning(f"Missing entities in rewritten query: {missing}")
            #         # Append missing entities to the rewritten query
            #         rewritten = rewritten + " " + " ".join(missing)
            #         logger.info(f"Appended missing entities: '{rewritten}'")

            self.histories.append(query)

            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return {
                "rewritten_query": rewritten,
                "medical_entities": medical_entities,
                "tokens": tokens,
            }

        except Exception as e:
            logger.error(f"Query rewrite failed: {e}", exc_info=True)
            return {
                "rewritten_query": query,
                "medical_entities": [],
                "tokens": 0,
            }

    def clear_history(self) -> None:
        self.histories.clear()

    def add_to_history(self, query: str) -> None:
        self.histories.append(query)


class HyDERewriter(BaseRewriter):
    def __init__(self, model: dict[str, Any], temperature: float = 0.3):
        super().__init__(model=model, temperature=temperature)

    def _build_prompt(self) -> str:
        return self.prompt_manager.get("hyde_rewrite")

    def rewrite(self, query: str) -> Dict[str, Any]:
        try:
            reasoning, content, tokens = self.llm.generate(
                messages=[
                    {"role": "system", "content": self._build_prompt()},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
            )

            hypothetical_doc = content.strip() if content else query
            logger.info(f"HyDE generated: '{query}' -> '{hypothetical_doc}...'")

            return {
                "rewritten_query": hypothetical_doc,
                "tokens": tokens,
            }

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}", exc_info=True)
            return {
                "rewritten_query": query,
                "tokens": 0,
            }


class MiltiQueryOptimizer(BaseRewriter):
    def __init__(self, model: dict[str, Any]):
        super().__init__(model=model)

    def _build_prompt(self) -> str:
        return self.prompt_manager.get("multi_query_rewrite")

    def rewrite(self, query: str) -> Dict[str, Any]:
        try:
            reasoning, content, tokens = self.llm.generate(
                messages=[
                    {"role": "system", "content": self._build_prompt()},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
            )
            rewritten = self._clean_multi_response(content)

            logger.info(f"Multi query rewritten: '{query}' -> '{rewritten}'")
            return {
                "rewritten_query": rewritten,
                "tokens": tokens,
            }
        except Exception as e:
            logger.error(f"Multi query rewrite failed: {e}", exc_info=True)
            return {
                "rewritten_query": query,
                "tokens": 0,
            }


class GlossaryInjector:

    select_fields = ["concept_name", "domain_id", "concept_class_id"]
    threshold = 0.4

    def __init__(self):
        self.vector_store = VectorStoreClient()

    def ner(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities from query and standardize them using SNOMED.

        Returns:
            Dict mapping original words to standardized SNOMED concepts:
            {
                "原始词": {
                    "concept_names": "标准化概念名1; 标准化概念名2",
                    "start": int,
                    "end": int
                }
            }
        """
        snomed_entities = {}
        entities = ner_service.get_entities(query)
        if not entities:
            return snomed_entities
        combined_results = ner_service.combine_entities(entities, query)
        words_to_search = []
        entities_to_search = []
        for ent in combined_results:
            if float(ent["score"]) >= self.threshold or ent["entity_group"] == "COMBINED_BIO_SYMPTOM":
                words_to_search.append(ent["word"])
                entities_to_search.append(ent)

        if not words_to_search:
            return snomed_entities

        word_embeddings = embedder.embed_queries_batch(words_to_search)

        for entity, word_embedding in zip(entities_to_search, word_embeddings):
            filter_expr = ""
            if entity["entity_group"] not in ["COMBINED_BIO_SYMPTOM"]:
                domain_ids = entity["domain_id"].split(";")
                concept_class_ids = entity["concept_class_id"].split(";")
                filter_expr = f"domain_id in {domain_ids} or concept_class_id in {concept_class_ids}"
            search_result = self.vector_store.search(
                self.select_fields,
                [word_embedding],
                limit=2,
                indexNames="rag_fintech",
                knowledgebaseIds=["snomed_kb"],
                filters=filter_expr,
            )
            if search_result and search_result[0]:
                concept_names = []
                for res in search_result[0]:
                    concept_name = res["concept_name"]
                    if concept_name not in concept_names:
                        concept_names.append(concept_name)

                snomed_entities[entity["word"]] = {
                    "concept_names": "; ".join(concept_names),
                    "start": entity["start"],
                    "end": entity["end"],
                }

        return snomed_entities

    def inject(self, query: str, snomed_entities: Dict[str, Dict[str, Any]]) -> str:
        """
        Replace original words in query with standardized SNOMED concepts.

        Args:
            query: Original query string
            snomed_entities: Result from inject() method

        Returns:
            Query with original words replaced by standardized concepts
        """
        if not snomed_entities:
            return query

        # Sort by start position (descending) to replace from end to start
        sorted_entities = sorted(snomed_entities.items(), key=lambda x: x[1]["start"], reverse=True)

        result = query
        for original_word, entity_info in sorted_entities:
            end = entity_info["end"]
            first_concept = f"(SNOMED:{entity_info['concept_names']})"
            result = result[:end] + first_concept + result[end:]

        logger.info("Glossary Injected, result: %s", result)
        return result

    def get_concept_terms(self, snomed_entities: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Extract all standardized concept names as a list of terms.

        Args:
            snomed_entities: Result from inject() method

        Returns:
            List of standardized concept names
        """
        concept_terms = []
        for entity_info in snomed_entities.values():
            concept_names = entity_info["concept_names"]
            # Split by semicolon and add each concept
            for concept in concept_names.split(";"):
                concept = concept.strip()
                if concept and concept not in concept_terms:
                    concept_terms.append(concept)
        return concept_terms

    def enhance_query(self, query: str, snomed_entities: Dict[str, Dict[str, Any]]) -> str:
        """
        Enhance query by appending standardized concepts.

        Args:
            query: Original query string
            snomed_entities: Result from inject() method

        Returns:
            Enhanced query with standardized concepts appended
        """
        if not snomed_entities:
            return query

        concept_terms = self.get_concept_terms(snomed_entities)
        if concept_terms:
            enhanced = f"{query} {' '.join(concept_terms)}"
            logger.info(f"Enhanced query with concepts: '{query}' -> '{enhanced}'")
            return enhanced
        return query


class QueryOptimizer:

    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.glossary_injector = GlossaryInjector()
        self.unified_rewriter = UnifiedRewriter(model=model)
        self.hyde_rewriter = HyDERewriter(model=model)
        self.multi_query_rewriter = MiltiQueryOptimizer(model=model)

    @cached(prefix="optimize", ttl=1800)
    def optimize(
        self, query: str, mode: Literal["unified", "hyde", "multi"] = "unified", use_snomed_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize query with optional SNOMED entity enhancement.

        Args:
            query: Original query string
            mode: Optimization mode ("unified", "hyde", or "multi")
            use_snomed_enhancement: If True, enhance query with standardized SNOMED concepts

        Returns:
            Dict containing:
            - snomed_entities: Dict of recognized and standardized entities
            - optimized_queries: List of optimized query strings
            - tokens: Total tokens used
        """
        query_to_use = query
        optimized_queries = []
        tokens = 0
        snomed_entities = {}

        if mode == "unified":
            # Extract and standardize entities using SNOMED
            # snomed_entities = self.glossary_injector.ner(query)
            # if use_snomed_enhancement and snomed_entities:
            #     query_to_use = self.glossary_injector.inject(query, snomed_entities)

            # # Rewrite query with entity constraints
            # if snomed_entities:
            #     entities = snomed_entities.keys()
            # else:
            #     entities = []
            entities = []
            rewritten_query = self.unified_rewriter.rewrite(query, entities)
            optimized_queries.append(rewritten_query["rewritten_query"])
            tokens += rewritten_query["tokens"]

            # Log recognized entities
            if snomed_entities:
                logger.info(f"Recognized {len(snomed_entities)} entities: {list(snomed_entities.keys())}")

        elif mode == "hyde":
            rewritten_query = self.hyde_rewriter.rewrite(query)
            optimized_queries.append(rewritten_query["rewritten_query"])
            tokens += rewritten_query["tokens"]
        elif mode == "multi":
            optimized_queries.append(query)
            rewritten_query = self.multi_query_rewriter.rewrite(query)
            optimized_queries.append(rewritten_query["rewritten_query"])
            tokens += rewritten_query["tokens"]
        else:
            raise ValueError(f"Invalid optimization mode: {mode}")

        return {
            "query_to_use": query_to_use,
            "snomed_entities": snomed_entities,
            "optimized_queries": optimized_queries,
            "tokens": tokens,
        }

    def clear_history(self) -> None:
        self.unified_rewriter.clear_history()

    def add_to_history(self, query: str) -> None:
        self.unified_rewriter.add_to_history(query)


def _create_query_optimizer() -> QueryOptimizer:
    registry = get_model_registry()
    model_config = registry.get_chat_model("query_lite")
    query_optimizer = QueryOptimizer(model=model_config.to_dict())

    logger.info("Initialized QueryOptimizer singleton")
    return query_optimizer


query_optimizer = _create_query_optimizer()
