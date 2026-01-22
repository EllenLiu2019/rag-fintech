import networkx as nx
import json
from collections import defaultdict
from typing import Any
import asyncio
import re

from graphrag.extractor import Extractor
from common import get_logger
from common.constants import VECTOR_GRAPH_KB, VECTOR_GRAPH_SIMILAR_FIELDS
from repository.vector import vector_store
from rag.embedding import dense_embedder, sparse_embedder
from graphrag.utils import (
    chat_limiter,
    generate_entity_id,
    generate_relationship_id,
)

logger = get_logger(__name__)


class EntityAlignment(Extractor):
    def __init__(self, merged_graph: nx.DiGraph, graph_lock: asyncio.Lock):
        super().__init__()
        self.merged_graph = merged_graph
        self.graph_lock = graph_lock

    async def __call__(self, root_id: int):
        candidate_attrs: dict[int, dict[str, Any]] = defaultdict()
        sim_entities: dict[int, list[dict]] = defaultdict(list)
        sim_entities_prompt: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict())
        aligned_entities = defaultdict()

        for node_id, node_attrs in self.merged_graph.nodes(data=True):
            if node_attrs["root_id"] != root_id:
                continue
            candidate_attrs[node_id] = {
                "entity_name": node_attrs["entity_name"],
                "entity_type": node_attrs["entity_type"],
                "doc_id": node_attrs["doc_id"],
                "root_id": node_attrs["root_id"],
                "clause_ids": node_attrs["clause_ids"],
            }

        await self._search_similar_entities(
            candidate_attrs,
            sim_entities,
            sim_entities_prompt,
        )

        if not sim_entities_prompt:
            logger.info("No candidate entities found for alignment")
            return

        variables = {"candidates": sim_entities_prompt}
        prompt = self.prompt_manager.get("entity_alignment", **variables)

        async with chat_limiter:
            logger.debug(f"Acquired chat limiter, {chat_limiter._value} slots remaining")
            try:
                response, _ = await self.chat(prompt, [{"role": "user", "content": "Output:"}])
            except Exception as e:
                logger.error(f"Error calling LLM for entity alignment: {e}")
                return

            try:
                # Clean up markdown code blocks from response
                cleaned_response = response.replace("```json", "").replace("```", "").strip()

                aligned_entities = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                # Pattern: match numeric keys like "123:" or "  123:" at the start of a line
                fixed_response = re.sub(r"(\s+)(\d+)(\s*):", r'\1"\2"\3:', cleaned_response)

                try:
                    aligned_entities = json.loads(fixed_response)
                    logger.warning("Fixed LLM JSON response with unquoted numeric keys")
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse LLM response as JSON: {e}. Response: {response[:200]}")
                    logger.debug(f"Attempted fix also failed: {e2}")
                    return
            except Exception as e:
                logger.error(f"Unexpected error parsing LLM response: {e}")
                return

        async with self.graph_lock:
            self._align_graph(
                candidate_attrs,
                sim_entities,
                aligned_entities,
            )

    async def _search_similar_entities(
        self,
        candidate_attrs: dict[int, dict[str, Any]],
        sim_entities: dict[int, list[dict]],
        sim_entities_prompt: dict[int, dict[str, Any]],
    ):
        # Track which entities have been chosen as candidates
        # Prevents the same entity from appearing in multiple candidate groups
        chosen_node_ids = set()

        for node_id, attrs in candidate_attrs.items():

            if node_id in chosen_node_ids:
                continue

            # Search for similar entities in vector store using HNSW
            # Purpose: Narrow down candidate pool for LLM alignment and deduplication
            dense_task = asyncio.to_thread(dense_embedder.embed_query, attrs["entity_name"])
            sparse_task = asyncio.to_thread(sparse_embedder.embed_queries, [attrs["entity_name"]])
            dense_vector, sparse_vectors = await asyncio.gather(dense_task, sparse_task)

            # Perform hybrid search (dense + sparse) in thread pool to avoid blocking
            search_results = await asyncio.to_thread(
                vector_store.hybrid_search,
                selectFields=VECTOR_GRAPH_SIMILAR_FIELDS,
                dense_vectors=[dense_vector],
                sparse_vectors=sparse_vectors,
                limit=5,
                knowledgebaseIds=[VECTOR_GRAPH_KB],
                filters={"doc_id": attrs["doc_id"], "graph_type": "entity", "root_id": attrs["root_id"]},
            )
            results = search_results[0]

            for result in results:
                if result["score"] < 0.03:
                    continue

                found_id = result["id"]
                if found_id in chosen_node_ids:
                    continue

                chosen_node_ids.add(found_id)

                nodes = {
                    "entity_name": result["entity_name"],
                    "description": result["description"],
                    "clause_ids": result["clause_ids"].split(",") if result["clause_ids"] else [],
                }
                sim_entities[node_id].append(dict(id=found_id, **nodes))

        sim_entities = {node_id: entities for node_id, entities in sim_entities.items() if len(entities) > 1}

        for node_id, entities in sim_entities.items():
            identified_entities: list[dict] = []
            edges: list[dict] = []
            for entity in entities:
                entity_info = entity.copy()
                id = entity_info.pop("id")
                identified_entities.append(entity_info)

                for successor_id in list(self.merged_graph.successors(id)):
                    edge_data = self.merged_graph.get_edge_data(id, successor_id)
                    edge_info = dict(edge_data)
                    edge = {
                        "source_name": edge_info["source_entity"],
                        "target_name": edge_info["target_entity"],
                        "rel_type": edge_info["rel_type"],
                        "description": edge_info["description"],
                    }
                    edges.append(edge)

                for predecessor_id in list(self.merged_graph.predecessors(id)):
                    edge_data = self.merged_graph.get_edge_data(predecessor_id, id)
                    edge_info = dict(edge_data)
                    edge = {
                        "source_name": edge_info["source_entity"],
                        "target_name": edge_info["target_entity"],
                        "rel_type": edge_info["rel_type"],
                        "description": edge_info["description"],
                    }
                    edges.append(edge)

            sim_entities_prompt[node_id]["identified_entities"] = identified_entities
            sim_entities_prompt[node_id]["reference_edges"] = edges

    def _align_graph(
        self,
        candidate_attrs: dict[int, dict[str, Any]],
        sim_entities: dict[int, list[dict]],
        aligned_results: dict[int, list[dict]],
    ):
        alignments_by_id: dict[int, dict[str, Any]] = defaultdict()

        for node_id_str, aligned_entities in aligned_results.items():
            try:
                node_id = int(node_id_str)
            except ValueError:
                logger.warning(f"Invalid node ID: {node_id_str}")
                continue
            for aligned_entity in aligned_entities:
                aligned_entity["id"] = generate_entity_id(
                    aligned_entity["entity_name"],
                    candidate_attrs[node_id]["root_id"],
                    candidate_attrs[node_id]["doc_id"],
                )

            alignment = {
                "sim_entities": sim_entities[node_id],
                "aligned_entities": aligned_entities,
            }
            alignments_by_id[node_id] = alignment

        for node_id, alignment in alignments_by_id.items():
            sim_entities: list[dict] = alignment["sim_entities"]
            aligned_entities: list[dict] = alignment["aligned_entities"]
            sim_entity_ids = [sim_entity["id"] for sim_entity in sim_entities]
            aligned_entity_ids = [aligned_entity["id"] for aligned_entity in aligned_entities]

            # Collect all edges from candidates that will be removed
            # Map: candidate_id -> (successors, predecessors)
            edges_to_transfer: dict[int, tuple[list[tuple], list[tuple]]] = {}

            for sim_entity in sim_entities:
                # replaced entity - need to transfer its edges
                if sim_entity["id"] not in aligned_entity_ids:
                    candidate_id = sim_entity["id"]

                    # Collect successor edges BEFORE removing node
                    successor_edges = []
                    for successor_id in list(self.merged_graph.successors(candidate_id)):
                        edge_data = self.merged_graph.get_edge_data(candidate_id, successor_id)
                        successor_edges.append((successor_id, edge_data))

                    # Collect predecessor edges BEFORE removing node
                    predecessor_edges = []
                    for predecessor_id in list(self.merged_graph.predecessors(candidate_id)):
                        edge_data = self.merged_graph.get_edge_data(predecessor_id, candidate_id)
                        predecessor_edges.append((predecessor_id, edge_data))

                    edges_to_transfer[candidate_id] = (successor_edges, predecessor_edges)

                    # Now remove the node (and its edges)
                    self.merged_graph.remove_node(candidate_id)

            # Add or update aligned entities
            for aligned_entity in aligned_entities:
                # New entity added by alignment
                if aligned_entity["id"] not in sim_entity_ids:
                    aligned_entity["entity_type"] = candidate_attrs[node_id]["entity_type"]
                    aligned_entity["doc_id"] = candidate_attrs[node_id]["doc_id"]
                    aligned_entity["root_id"] = candidate_attrs[node_id]["root_id"]
                    aligned_entity["clause_ids"] = sorted(set(aligned_entity.get("clause_ids", "")))
                    self.merged_graph.add_node(aligned_entity["id"], **aligned_entity)

                    # Transfer edges from all removed candidates to this new entity
                    # This assumes all removed candidates should be merged into the aligned entities
                    for candidate_id, (successor_edges, predecessor_edges) in edges_to_transfer.items():
                        # Add outgoing edges (this entity -> successors)
                        for successor_id, edge_data in successor_edges:
                            if self.merged_graph.has_node(successor_id):  # Check target still exists
                                edge_info = dict(edge_data)
                                edge_info["source_id"] = aligned_entity["id"]
                                edge_info["source_entity"] = aligned_entity["entity_name"]
                                edge_info["id"] = generate_relationship_id(
                                    edge_info["source_entity"],
                                    edge_info["target_entity"],
                                    edge_info["rel_type"],
                                    candidate_attrs[node_id]["root_id"],
                                    candidate_attrs[node_id]["doc_id"],
                                )
                                self.merged_graph.add_edge(aligned_entity["id"], successor_id, **edge_info)

                        # Add incoming edges (predecessors -> this entity)
                        for predecessor_id, edge_data in predecessor_edges:
                            if self.merged_graph.has_node(predecessor_id):  # Check source still exists
                                edge_info = dict(edge_data)
                                edge_info["target_id"] = aligned_entity["id"]
                                edge_info["target_entity"] = aligned_entity["entity_name"]
                                edge_info["id"] = generate_relationship_id(
                                    edge_info["source_entity"],
                                    edge_info["target_entity"],
                                    edge_info["rel_type"],
                                    candidate_attrs[node_id]["root_id"],
                                    candidate_attrs[node_id]["doc_id"],
                                )
                                self.merged_graph.add_edge(predecessor_id, aligned_entity["id"], **edge_info)

                else:  # Existing entity - just update attributes
                    existing_node = self.merged_graph.nodes[aligned_entity["id"]]
                    existing_node["entity_name"] = aligned_entity["entity_name"]
                    existing_node["description"] = aligned_entity["description"]
                    existing_node["clause_ids"] = sorted(set(aligned_entity.get("clause_ids", [])))
                    self.merged_graph.nodes[aligned_entity["id"]].update(existing_node)
