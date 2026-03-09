import os
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Callable
import time
from common import get_logger, model_registry, prompt_manager
from common.constants import GRAPH_FIELD_SEP, DEFAULT_ENTITY_TYPES
from graphrag.utils import (
    chat_limiter,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    split_string_by_multi_markers,
    generate_entity_id,
    generate_relationship_id,
)
from rag.llm.chat_model import chat_model


logger = get_logger(__name__)


class Extractor:

    def __init__(self):
        model_config = model_registry.get_chat_model("qa_reasoner")
        model = model_config.to_dict()

        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )

    async def chat(self, system, history):
        hist = deepcopy(history)

        # Build messages from system and history
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(hist)

        for attempt in range(3):
            try:
                # Run synchronous llm.generate in thread pool to avoid blocking
                start = time.time()
                reasoning, content, tokens = await asyncio.to_thread(self.llm.generate, messages=messages)
                logger.info(f"Time taken: {time.time() - start} seconds, Tokens used: {tokens}")
                return content or "", tokens
            except Exception as e:
                logger.error(e)
                if attempt == 2:
                    raise

        return "", 0

    async def __call__(self, doc_id: str, clauses: dict[int, str], callback: Callable | None = None):
        self.callback = callback

        async def extract_all(clauses):
            out_results = defaultdict(tuple)
            error_count = 0
            max_errors = int(os.environ.get("GRAPHRAG_MAX_ERRORS", 3))

            async def worker(root_id: int, clause_text: str):
                nonlocal error_count
                async with chat_limiter:
                    logger.debug(f"Acquired chat limiter, {chat_limiter._value} slots remaining")
                    try:
                        await self.process_single_content(root_id, clause_text, out_results)
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Error processing root {root_id}: {str(e)}"
                        logger.warning(error_msg)
                        if self.callback:
                            self.callback(msg=error_msg)

                        if error_count > max_errors:
                            raise Exception(f"Maximum error count ({max_errors}) reached. Last errors: {str(e)}")

            tasks = [worker(root_id, clause_text) for root_id, clause_text in clauses.items()]
            await asyncio.gather(*tasks)

            if error_count > 0:
                warning_msg = f"Completed with {error_count} errors (out of {len(clauses)} roots processed)"
                logger.warning(warning_msg)
                if self.callback:
                    self.callback(msg=warning_msg)

            return out_results

        out_results = await extract_all(clauses)

        maybe_nodes = defaultdict(lambda: defaultdict(list))
        maybe_edges = defaultdict(lambda: defaultdict(list))
        sum_token_count = 0
        for root_id, (m_nodes, m_edges, token_count) in out_results.items():
            for entity_name, entities in m_nodes.items():
                maybe_nodes[root_id][entity_name].extend(entities)
            for (src_id, tgt_id), relations in m_edges.items():
                maybe_edges[root_id][(src_id, tgt_id)].extend(relations)
            sum_token_count += token_count
        if self.callback:
            self.callback(
                msg=f"Entities and relationships extraction done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges."
            )
        logger.info(f"extracted {len(maybe_nodes)} roots.")

        entity_tasks = []
        for root_id, entities in maybe_nodes.items():
            for en_nm, ents in entities.items():
                entity_tasks.append(self._merge_nodes(doc_id, root_id, en_nm, ents))
        entity_results = await asyncio.gather(*entity_tasks)
        entities_data = [r for r in entity_results if r is not None]

        if self.callback:
            self.callback(msg="Entities merging done.")

        relationship_tasks = []
        for root_id, edges in maybe_edges.items():
            for (src, tgt), rels in edges.items():
                relationship_tasks.append(self._merge_edges(doc_id, root_id, src, tgt, rels))
        relationship_results = await asyncio.gather(*relationship_tasks)
        relationships_data = [r for r in relationship_results if r is not None]

        if self.callback:
            self.callback(msg="Relationships merging done.")

        logger.info(f"merged {len(entities_data)} nodes, {len(relationships_data)} edges.")

        if not len(entities_data) and not len(relationships_data):
            logger.warning("Didn't extract any entities and relationships, maybe your LLM is not working")

        if not len(entities_data):
            logger.warning("Didn't extract any entities")
        if not len(relationships_data):
            logger.warning("Didn't extract any relationships")

        return entities_data, relationships_data

    def _entities_and_relations(self, records: list, tuple_delimiter: str):
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        # Track seen entities and relations to avoid duplicates
        entities = set()
        relations = set()
        ent_types = [t.lower() for t in DEFAULT_ENTITY_TYPES]

        for record in records:
            record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

            if_entities = handle_single_entity_extraction(record_attributes)
            if if_entities is not None and if_entities.get("entity_type", "unknown").lower() in ent_types:
                entity = (
                    if_entities["entity_name"],
                    if_entities["entity_type"],
                    if_entities["description"],
                )
                if entity in entities:
                    continue
                entities.add(entity)
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = handle_single_relationship_extraction(record_attributes)
            if if_relation is not None:
                relation = (
                    if_relation["src_id"],
                    if_relation["tgt_id"],
                    if_relation["rel_type"],
                    if_relation["description"],
                )
                if relation in relations:
                    continue
                relations.add(relation)
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        return dict(maybe_nodes), dict(maybe_edges)

    async def _merge_nodes(
        self,
        doc_id: str,
        root_id: int,
        entity_name: str,
        entities: list[dict],
    ) -> dict[str, Any] | None:
        if not entities:
            return

        # Generate unique numeric ID
        entity_id = generate_entity_id(entity_name, root_id, doc_id)

        entity_type = sorted(
            Counter([dp["entity_type"] for dp in entities]).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        description = GRAPH_FIELD_SEP.join(sorted(set([entity["description"] for entity in entities])))
        description = await self._handle_entity_relation_summary(entity_name, description)

        clause_ids = sorted(set([cid for entity in entities for cid in entity.get("clause_ids", [])]))

        return dict(
            id=entity_id,  # Numeric primary key
            entity_name=entity_name,
            entity_type=entity_type,
            description=description,
            doc_id=doc_id,
            root_id=root_id,
            clause_ids=clause_ids,
        )

    async def _merge_edges(
        self,
        doc_id: str,
        root_id: int,
        src: str,
        tgt: str,
        edges_data: list[dict],
    ) -> dict[str, Any] | None:
        if not edges_data:
            return

        source_id = generate_entity_id(src, root_id, doc_id)
        target_id = generate_entity_id(tgt, root_id, doc_id)

        rel_type_cnt = sorted(
            Counter([dp["rel_type"] for dp in edges_data]).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        rel_type = rel_type_cnt[0][0]

        relationship_id = generate_relationship_id(src, tgt, rel_type, root_id, doc_id)

        description = GRAPH_FIELD_SEP.join(sorted(set([edge["description"] for edge in edges_data])))
        description = await self._handle_entity_relation_summary(f"{src} -> {tgt}", description)
        return dict(
            id=relationship_id,  # Numeric ID
            source_id=source_id,  # Numeric ID
            target_id=target_id,  # Numeric ID
            source_entity=src,
            target_entity=tgt,
            rel_type=rel_type,
            description=description,
            doc_id=doc_id,
            root_id=root_id,
        )

    async def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str) -> str:
        use_description = description[:512]
        description_list = use_description.split(GRAPH_FIELD_SEP)
        if len(description_list) <= 5:
            return use_description
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=description_list,
        )
        use_prompt = prompt_manager.get("summarize_descriptions", **context_base)
        logger.info(f"Trigger summary: {entity_or_relation_name}")

        async with chat_limiter:
            logger.debug(f"Acquired chat limiter, {chat_limiter._value} slots remaining")
            summary, _ = await self.chat("", [{"role": "user", "content": use_prompt}])
        return summary
