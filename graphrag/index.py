from typing import Callable
import asyncio
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from common import get_logger
from graphrag.graph_extractor import GraphExtractor
from graphrag.entity_alignment import EntityAlignment
from graphrag.utils import (
    tidy_graph,
    build_foc,
    do_merge_graph,
    update_pagerank,
    to_graph,
)
from rag.entity.clause_tree import ClauseForest
from repository.graph.neo4j_client import neo4j_client
from rag.embedding import dense_embedder, sparse_embedder
from repository.vector import vector_store
from common.constants import VECTOR_GRAPH_KB, VECTOR_GRAPH_FIELDS

logger = get_logger(__name__)


async def index(
    doc_id: str,
    clause_forest: ClauseForest,
    callback: Callable | None = None,
):
    foc: dict[int, str] = build_foc(clause_forest)
    subgraph = await generate_subgraph(doc_id, foc)
    merged_graph = await merge_graph(subgraph)
    await align_graph(merged_graph, foc.keys())


async def generate_subgraph(doc_id: str, foc: dict[int, str], callback: Callable | None = None) -> nx.DiGraph:
    extractor = GraphExtractor()
    entities, relationships = await extractor(doc_id, foc)

    # Use directed graph to preserve relationship direction
    subgraph = nx.DiGraph()

    for ent in entities:
        node_id = ent["id"]
        subgraph.add_node(node_id, **ent)

    ignored_rels = 0
    for rel in relationships:
        source_id = rel["source_id"]
        target_id = rel["target_id"]

        if not subgraph.has_node(source_id) or not subgraph.has_node(target_id):
            ignored_rels += 1
            logger.warning(
                f"Ignored relationship: {rel['source_entity']} -> {rel['target_entity']} "
                f"(IDs: {source_id} -> {target_id}) because one of the entities is not in the subgraph"
            )
            continue

        # Add directed edge from source to target
        subgraph.add_edge(
            source_id,
            target_id,
            **rel,
        )

    tidy_graph(subgraph, callback)
    subgraph.graph["doc_id"] = doc_id

    return subgraph


async def merge_graph(subgraph: nx.DiGraph) -> nx.DiGraph:
    # merge graph with existing graph in vector store
    doc_id = subgraph.graph["doc_id"]
    existing_graph = None
    vector_chunks = vector_store.query(
        VECTOR_GRAPH_KB,
        filters={"doc_id": doc_id},
        selectFields=VECTOR_GRAPH_FIELDS,
    )
    if vector_chunks:
        existing_graph = to_graph(doc_id, vector_chunks)

    if existing_graph:
        merged_graph = do_merge_graph(subgraph, existing_graph)
        logger.info(f"Graph already exists for doc_id={doc_id}, merging with new subgraph")
    else:
        merged_graph = subgraph
        logger.info(f"Graph does not exist for doc_id={doc_id}, creating new graph")

    update_pagerank(merged_graph)

    await persist_vector(merged_graph)

    return merged_graph


async def persist_vector(merged_graph: nx.DiGraph):
    doc_id = merged_graph.graph["doc_id"]
    entities = []
    for _, attrs in merged_graph.nodes(data=True):
        entity_data = dict(attrs)
        entities.append(entity_data)

    relationships = [dict(attrs) for _, _, attrs in merged_graph.edges(data=True)]

    chunks = []

    for entity in entities:
        entity_name = entity["entity_name"][:200]
        chunk = {
            "id": entity["id"],
            "graph_type": "entity",
            "entity_name": entity_name,
            "entity_type": entity["entity_type"],
            "description": entity["description"],
            "doc_id": entity["doc_id"],
            "root_id": entity["root_id"],
            "clause_ids": ",".join(sorted(set(entity["clause_ids"]))) if entity["clause_ids"] else "",
            "pagerank": entity["pagerank"],
            "text": entity["entity_name"],
        }
        chunks.append(chunk)

    for relationship in relationships:
        source_entity = relationship["source_entity"][:200]
        target_entity = relationship["target_entity"][:200]
        chunk = {
            "id": relationship["id"],
            "graph_type": "relationship",
            "source_id": relationship["source_id"],
            "target_id": relationship["target_id"],
            "source_entity": source_entity,
            "target_entity": target_entity,
            "rel_type": relationship["rel_type"],
            "description": relationship["description"],
            "doc_id": relationship["doc_id"],
            "root_id": relationship["root_id"],
            "text": relationship["description"],
        }
        chunks.append(chunk)

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(dense_embedder.embed_chunks, chunks)
        sparse_future = executor.submit(sparse_embedder.embed_chunks, chunks)

        dense_future.result()
        sparse_future.result()

    vector_store.delete(condition={"doc_id": doc_id}, knowledgebaseId=VECTOR_GRAPH_KB)

    vector_store.insert(chunks, VECTOR_GRAPH_KB)
    logger.info(
        f"Saved {len(chunks)} chunks ({len(entities)} entities + {len(relationships)} relationships) to Milvus collection: rag_fintech_{VECTOR_GRAPH_KB}"
    )


async def align_graph(merged_graph: nx.DiGraph, root_ids: list[int]):
    # align graph with existing graph in neo4j and vector store
    graph_lock = asyncio.Lock()
    entity_alignment = EntityAlignment(merged_graph, graph_lock)
    alignment_tasks = [entity_alignment(root_id) for root_id in root_ids]
    await asyncio.gather(*alignment_tasks)
    update_pagerank(merged_graph)
    await persist_graph(merged_graph)


async def persist_graph(merged_graph: nx.DiGraph):
    entities = []
    doc_id = merged_graph.graph["doc_id"]
    for _, attrs in merged_graph.nodes(data=True):
        entity_data = dict(attrs)
        entities.append(entity_data)

    relationships = [dict(attrs) for _, _, attrs in merged_graph.edges(data=True)]

    neo4j_client.delete_graph_by_doc_id(doc_id)

    neo4j_client.import_entities(entities)
    neo4j_client.import_relationships(relationships)

    await persist_vector(merged_graph)


if __name__ == "__main__":
    import asyncio
    from rag.persistence.persistent_service import PersistentService

    async def main():
        doc_id = "policy_0119223547_a02169"
        clause_forest = PersistentService.get_clause_forest(doc_id)
        await index(doc_id, clause_forest)

    asyncio.run(main())
