import html
import re
import asyncio
from collections import defaultdict
from hashlib import sha256, md5
from typing import Any
import networkx as nx
import xxhash

from common import get_logger
from rag.entity.clause_tree import ClauseForest, ClauseNode
from common.constants import GRAPH_FIELD_SEP


logger = get_logger(__name__)

chat_limiter = asyncio.Semaphore(5)


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\"\x00-\x1f\x7f-\x9f]", "", result)


def tidy_graph(graph: nx.DiGraph, callback, check_attribute: bool = True):
    """
    Ensure all nodes and edges in the directed graph have some essential attribute.
    """

    def is_valid_item(node_attrs: dict) -> bool:
        valid_node = True
        for attr in ["description"]:
            if attr not in node_attrs:
                valid_node = False
                break
        return valid_node

    if check_attribute:
        purged_nodes = []
        for node, node_attrs in graph.nodes(data=True):
            if not is_valid_item(node_attrs):
                purged_nodes.append(node)
        for node in purged_nodes:
            graph.remove_node(node)
            logger.warning(f"Purged node {node} from graph due to missing essential attributes.")
        if purged_nodes and callback:
            callback(msg=f"Purged {len(purged_nodes)} nodes from graph due to missing essential attributes.")

    purged_edges = []
    for source, target, attr in graph.edges(data=True):
        if check_attribute:
            if not is_valid_item(attr):
                purged_edges.append((source, target))
                logger.warning(f"Purged edge {source} -> {target} from graph due to missing essential attributes.")
    for source, target in purged_edges:
        graph.remove_edge(source, target)
    if purged_edges and callback:
        callback(msg=f"Purged {len(purged_edges)} edges from graph due to missing essential attributes.")


def do_merge_graph(new_graph: nx.DiGraph, existing_graph: nx.DiGraph):
    """Merge new_graph into existing_graph in place.

    Works with directed graphs. Node IDs are numeric identifiers based on (entity_name, doc_id, root_id).
    """

    for node_id, attr in new_graph.nodes(data=True):
        if not existing_graph.has_node(node_id):
            existing_graph.add_node(node_id, **attr)
            continue
        node = existing_graph.nodes[node_id]
        node["description"] += GRAPH_FIELD_SEP + attr["description"]

        # Merge clause_ids (ensure both are lists)
        existing_clause_ids = node.get("clause_ids", [])
        new_clause_ids = attr.get("clause_ids", [])
        if isinstance(existing_clause_ids, str):
            existing_clause_ids = [cid.strip() for cid in existing_clause_ids.split(",") if cid.strip()]
        if isinstance(new_clause_ids, str):
            new_clause_ids = [cid.strip() for cid in new_clause_ids.split(",") if cid.strip()]
        node["clause_ids"] = sorted(set(existing_clause_ids + new_clause_ids))

    for source, target, attr in new_graph.edges(data=True):
        edge = existing_graph.get_edge_data(source, target)
        if edge is None:
            existing_graph.add_edge(source, target, **attr)
            continue
        edge["description"] += GRAPH_FIELD_SEP + attr["description"]

    # Update rank based on degree (for directed graphs, this includes in-degree + out-degree)
    for node_id, degree in existing_graph.degree:
        existing_graph.nodes[node_id]["rank"] = int(degree)

    return existing_graph


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def handle_single_entity_extraction(record_attributes: list[str]):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1])
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    clause_ids = clean_str(record_attributes[4]).split(",")
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        clause_ids=clause_ids,
    )


def handle_single_relationship_extraction(record_attributes: list[str]):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    return dict(
        src_id=clean_str(record_attributes[1]),
        tgt_id=clean_str(record_attributes[2]),
        rel_type=clean_str(record_attributes[3].upper()),
        description=clean_str(record_attributes[4]),
    )


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(args)]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def chunk_id(chunk):
    return xxhash.xxh64((chunk["content_with_weight"] + chunk["kb_id"]).encode("utf-8")).hexdigest()


def to_graph(doc_id: str, chunks: list[dict]) -> nx.DiGraph:
    entities = []
    relationships = []

    for chunk in chunks:
        if chunk["graph_type"] == "entity":
            clause_ids_raw = chunk.get("clause_ids", "")
            if isinstance(clause_ids_raw, str):
                clause_ids = [cid.strip() for cid in clause_ids_raw.split(",") if cid.strip()]
            else:
                clause_ids = clause_ids_raw
            entity = {
                "id": chunk["id"],
                "entity_name": chunk["entity_name"],
                "entity_type": chunk["entity_type"],
                "description": chunk["description"],
                "doc_id": chunk["doc_id"],
                "root_id": chunk["root_id"],
                "clause_ids": clause_ids,
            }
            entities.append(entity)
        elif chunk["graph_type"] == "relationship":
            relationship = {
                "id": chunk["id"],
                "source_id": chunk["source_id"],
                "target_id": chunk["target_id"],
                "source_entity": chunk["source_entity"],
                "target_entity": chunk["target_entity"],
                "rel_type": chunk["rel_type"],
                "description": chunk["description"],
                "doc_id": chunk["doc_id"],
                "root_id": chunk["root_id"],
            }
            relationships.append(relationship)
    return get_graph(doc_id, entities, relationships)


def get_graph(doc_id: str, entities: list[dict], relationships: list[dict]) -> nx.DiGraph:
    """Build NetworkX directed graph from entities and relationships

    Node uniqueness is determined by entity ID (based on entity_name + doc_id + clause_id)
    """
    graph = nx.DiGraph()

    # Add nodes (entities) using numeric ID as node identifier
    for entity in entities:
        entity_data = entity.copy()
        node_id = entity_data["id"]
        graph.add_node(node_id, **entity_data)

    # Add directed edges (relationships)
    for relationship in relationships:
        rel_data = relationship.copy()
        source_id = rel_data["source_id"]
        target_id = rel_data["target_id"]
        graph.add_edge(source_id, target_id, **rel_data)

    graph.graph["doc_id"] = doc_id

    return graph


def is_continuous_subsequence(subseq, seq):
    def find_all_indexes(tup, value):
        indexes = []
        start = 0
        while True:
            try:
                index = tup.index(value, start)
                indexes.append(index)
                start = index + 1
            except ValueError:
                break
        return indexes

    index_list = find_all_indexes(seq, subseq[0])
    for idx in index_list:
        if idx != len(seq) - 1:
            if seq[idx + 1] == subseq[-1]:
                return True
    return False


def merge_tuples(list1, list2):
    result = []
    for tup in list1:
        last_element = tup[-1]
        if last_element in tup[:-1]:
            result.append(tup)
        else:
            matching_tuples = [t for t in list2 if t[0] == last_element]
            already_match_flag = 0
            for match in matching_tuples:
                matchh = (match[1], match[0])
                if is_continuous_subsequence(match, tup) or is_continuous_subsequence(matchh, tup):
                    continue
                already_match_flag = 1
                merged_tuple = tup + match[1:]
                result.append(merged_tuple)
            if not already_match_flag:
                result.append(tup)
    return result


def flat_uniq_list(arr, key):
    res = []
    for a in arr:
        a = a[key]
        if isinstance(a, list):
            res.extend(a)
        else:
            res.append(a)
    return list(set(res))


def build_foc(clause_forest: ClauseForest) -> list[dict[str, Any]]:

    result: defaultdict[int, str] = defaultdict(str)

    def build_tree(node: ClauseNode) -> str:
        header = "#" * (node.level + 2)
        markdown = f"{header} {node.title} [ID:{node.id}] \n {node.content}\n"

        for child in node.children:
            markdown += build_tree(child)
        return markdown

    for root in clause_forest.trees.keys():
        result[root.id] = build_tree(root)

    return result


def generate_entity_id(entity_name: str, root_id: int, doc_id: str) -> int:
    """
    Generate a unique numeric ID for an entity based on its composite key.
    Uses SHA256 hash of (entity_name, root_id, doc_id) to ensure consistency.

    ID is a 64-bit integer (first 16 hex digits of SHA256) to be compatible with Neo4j.
    Neo4j's PackStream protocol supports signed 64-bit integers:
    Range: -2^63 to 2^63-1 (-9223372036854775808 to 9223372036854775807)

    We use the first 15 hex digits (60 bits) and ensure it's positive to stay within safe range.
    Collision probability with 60 bits:
    - < 1M entities: negligible (< 10^-12)
    - < 1B entities: very low (< 10^-3)
    - For most use cases, this is acceptable

    Returns a positive 64-bit integer.
    """
    # Create composite key
    composite_key = f"{doc_id}:{root_id}:{entity_name}"
    # Generate SHA256 hash
    hash_value = sha256(composite_key.encode("utf-8")).hexdigest()
    # Take first 15 hex digits (60 bits) to ensure we stay within 64-bit signed integer range
    # This gives us 2^60 ≈ 1.15 × 10^18 possible values
    entity_id = int(hash_value[:15], 16)
    return entity_id


def generate_relationship_id(src: str, tgt: str, rel_type: str, root_id: int, doc_id: str) -> int:
    """
    Generate a unique numeric ID for a relationship based on its composite key.
    Uses SHA256 hash of (src, tgt, root_id, doc_id) to ensure consistency.
    """
    composite_key = f"{doc_id}:{root_id}:{src}:{tgt}:{rel_type}"
    hash_value = sha256(composite_key.encode("utf-8")).hexdigest()
    relationship_id = int(hash_value[:15], 16)
    return relationship_id


def update_pagerank(graph: nx.DiGraph):
    pr: dict[int, float] = nx.pagerank(graph)
    for node_id, pagerank in pr.items():
        graph.nodes[node_id]["pagerank"] = pagerank
    return graph
