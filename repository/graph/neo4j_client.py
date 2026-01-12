import neo4j

from common.config_utils import get_base_config
from common import get_logger
from common.exceptions import ConnectionError

logger = get_logger(__name__)

ALLOWED_RELATIONSHIP_TYPES = ["INCLUDE", "NOT_INCLUDE", "RELATED_TO"]


class Neo4jClient:
    def __init__(self):
        try:
            self.driver = self._init_driver()
            self.database = self._get_database()
            logger.info("Neo4j client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j client: {e}")
            raise

    def _init_driver(self):
        neo4j_config = get_base_config("neo4j", {})
        uri = neo4j_config.get("uri")
        username = neo4j_config.get("username")
        password = neo4j_config.get("password")

        try:
            driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
            driver.verify_connectivity()
        except Exception as e:
            if driver:
                driver.close()
            raise ConnectionError(f"Failed to verify Neo4j connectivity: {e}") from e

        return driver

    def _get_database(self):
        """Get database name from config"""
        neo4j_config = get_base_config("neo4j", {})
        return neo4j_config.get("database", "neo4j")

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed.")

    def create_node(self, node_type, properties):
        """Create a node with the given type and properties"""
        with self.driver.session(database=self.database) as session:
            result = session.run(f"CREATE (n:{node_type}) SET n = $properties RETURN n", properties=properties)
            return result.single()

    def execute_query(self, query, parameters=None):
        """Execute a Cypher query and return results"""
        with self.driver.session(database=self.database, default_access_mode=neo4j.READ_ACCESS) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def import_entities(self, entities):
        """Import entities into Neo4j"""
        with self.driver.session(database=self.database) as session:
            for row in entities:
                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n.entity_name = $entity_name,
                        n.entity_type = $entity_type,
                        n.description = $description,
                        n.doc_id = $doc_id,
                        n.clause_id = $clause_id,
                        n.chunk_id = $chunk_id
                    """,
                    id=row["id"],
                    entity_name=row["entity_name"],
                    entity_type=row["entity_type"],
                    description=row["description"],
                    doc_id=row["doc_id"],
                    clause_id=row["clause_id"],
                    chunk_id=row.get("chunk_id", ""),
                )

    def import_relationships(self, relationships):
        """Import relationships into Neo4j with specific relationship types"""
        with self.driver.session(database=self.database) as session:
            for row in relationships:
                rel_type = row["rel_type"]

                # Security: Validate relationship type against whitelist
                # Since we use f-string to build query (not Cypher parameters),
                # we must validate to prevent Cypher injection attacks
                if rel_type in ALLOWED_RELATIONSHIP_TYPES:
                    query = f"""
                    MATCH (e1:Entity {{id: $source_id}})
                    MATCH (e2:Entity {{id: $target_id}})
                    MERGE (e1)-[r:{rel_type}]->(e2)
                    SET r.id = $id,
                        r.description = $description,
                        r.clause_id = $clause_id,
                        r.doc_id = $doc_id
                    """
                    session.run(
                        query,
                        id=row["id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        description=row["description"],
                        clause_id=row["clause_id"],
                        doc_id=row["doc_id"],
                    )
                else:
                    query = """
                    MATCH (e1:Entity {id: $source_id})
                    MATCH (e2:Entity {id: $target_id})
                    MERGE (e1)-[r:RELATED_TO]->(e2)
                    SET r.rel_type = $rel_type,
                        r.id = $id,
                        r.description = $description,
                        r.clause_id = $clause_id,
                        r.doc_id = $doc_id
                    """
                    session.run(
                        query,
                        id=row["id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        rel_type=rel_type,
                        description=row["description"],
                        clause_id=row["clause_id"],
                        doc_id=row["doc_id"],
                    )

    def load_subgraph(self, doc_id: str) -> tuple[list[dict], list[dict]]:
        """Load subgraph from Neo4j for a specific document and clause and return lists of entities and relationships"""
        with self.driver.session(database=self.database, default_access_mode=neo4j.READ_ACCESS) as session:
            # Load entities for the document
            entities_result = session.run(
                """
                MATCH (n:Entity {doc_id: $doc_id})
                RETURN n.id AS id,
                       n.entity_name AS entity_name,
                       n.entity_type AS entity_type,
                       n.description AS description,
                       n.doc_id AS doc_id,
                       n.clause_id AS clause_id,
                       n.chunk_id AS chunk_id
                """,
                doc_id=doc_id,
            )

            # Convert Result to list of dicts
            entities = [dict(record) for record in entities_result]

            # Load relationships for the document (support multiple relationship types)
            relationships_result = session.run(
                """
                MATCH (e1:Entity {doc_id: $doc_id})-[r]->(e2:Entity {doc_id: $doc_id})
                WHERE type(r) IN ['INCLUDE', 'NOT_INCLUDE', 'RELATED_TO']
                RETURN r.id AS id,
                       e1.id AS source_id,
                       e2.id AS target_id,
                       e1.entity_name AS source_entity,
                       e2.entity_name AS target_entity,
                       CASE 
                         WHEN type(r) IN ['INCLUDE', 'NOT_INCLUDE'] THEN type(r)
                         ELSE coalesce(r.rel_type, 'RELATED_TO')
                       END AS rel_type,
                       r.description AS description,
                       r.clause_id AS clause_id,
                       r.doc_id AS doc_id
                """,
                doc_id=doc_id,
            )

            # Convert Result to list of dicts
            relationships = [dict(record) for record in relationships_result]

        return entities, relationships

    def load_full_graph(self):
        """Load entire graph from Neo4j and return NetworkX graph"""
        import networkx as nx

        graph = nx.Graph()

        with self.driver.session(database=self.database, default_access_mode=neo4j.READ_ACCESS) as session:
            # Load all entities
            entities_result = session.run(
                """
                MATCH (n:Entity)
                RETURN n.id AS id,
                       n.entity_name AS entity_name,
                       n.entity_type AS entity_type,
                       n.description AS description,
                       n.doc_id AS doc_id,
                       n.clause_id AS clause_id,
                       n.chunk_id AS chunk_id
                """
            )

            # Add nodes to graph
            for record in entities_result:
                node_data = dict(record)
                entity_name = node_data.pop("entity_name")
                graph.add_node(entity_name, **node_data)

            # Load all relationships (support multiple relationship types)
            relationships_result = session.run(
                """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE type(r) IN ['INCLUDE', 'NOT_INCLUDE', 'RELATED_TO']
                RETURN r.id AS id,
                       e1.entity_name AS source_entity,
                       e2.entity_name AS target_entity,
                       CASE 
                         WHEN type(r) IN ['INCLUDE', 'NOT_INCLUDE'] THEN type(r)
                         ELSE coalesce(r.rel_type, 'RELATED_TO')
                       END AS rel_type,
                       r.description AS description,
                       r.clause_id AS clause_id,
                       r.doc_id AS doc_id
                """
            )

            # Add edges to graph
            for record in relationships_result:
                edge_data = dict(record)
                source = edge_data.pop("source_entity")
                target = edge_data.pop("target_entity")
                graph.add_edge(source, target, **edge_data)

        logger.info(f"Loaded full graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph

    def get_entity_subgraph(self, entity_name: str, max_depth: int = 2):
        """Get subgraph around a specific entity within max_depth hops"""
        import networkx as nx

        graph = nx.Graph()

        with self.driver.session(database=self.database, default_access_mode=neo4j.READ_ACCESS) as session:
            # Use Cypher path query to get nodes within max_depth
            result = session.run(
                """
                MATCH path = (start:Entity {entity_name: $entity_name})-[*1..{max_depth}]-(connected:Entity)
                WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
                UNWIND path_nodes AS node
                WITH DISTINCT node, path_rels
                RETURN node.id AS id,
                       node.entity_name AS entity_name,
                       node.entity_type AS entity_type,
                       node.description AS description,
                       node.doc_id AS doc_id,
                       node.clause_id AS clause_id,
                       node.chunk_id AS chunk_id
                """.replace(
                    "{max_depth}", str(max_depth)
                ),
                entity_name=entity_name,
            )

            # Add nodes
            for record in result:
                node_data = dict(record)
                entity_name_node = node_data.pop("entity_name")
                graph.add_node(entity_name_node, **node_data)

            # Get relationships between these nodes (support multiple relationship types)
            node_names = list(graph.nodes())
            if node_names:
                rel_result = session.run(
                    """
                    MATCH (e1:Entity)-[r]->(e2:Entity)
                    WHERE e1.entity_name IN $node_names 
                      AND e2.entity_name IN $node_names
                      AND type(r) IN ['INCLUDE', 'NOT_INCLUDE', 'RELATED_TO']
                    RETURN r.id AS id,
                           e1.entity_name AS source_entity,
                           e2.entity_name AS target_entity,
                           CASE 
                             WHEN type(r) IN ['INCLUDE', 'NOT_INCLUDE'] THEN type(r)
                             ELSE coalesce(r.rel_type, 'RELATED_TO')
                           END AS rel_type,
                           r.description AS description,
                           r.clause_id AS clause_id,
                           r.doc_id AS doc_id
                    """,
                    node_names=node_names,
                )

                # Add edges
                for record in rel_result:
                    edge_data = dict(record)
                    source = edge_data.pop("source_entity")
                    target = edge_data.pop("target_entity")
                    graph.add_edge(source, target, **edge_data)

        logger.info(
            f"Loaded subgraph for entity={entity_name}, depth={max_depth}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        return graph

    def visualize_subgraph_cypher(
        self, entity_name: str, max_depth: int = 2, limit: int = 25, relationship_types: list[str] | None = None
    ) -> str:
        """
        Generate a Cypher query string for visualization in Neo4j browser
        """
        if relationship_types:
            rel_types_str = "|".join(relationship_types)
            return f"""
                    MATCH path = (start:Entity {{entity_name: "{entity_name}"}})-[:{rel_types_str}*1..{max_depth}]-(connected:Entity)
                    RETURN path
                    LIMIT {limit};
                                """.strip()
        else:
            return f"""
                    MATCH path = (start:Entity {{entity_name: "{entity_name}"}})-[*1..{max_depth}]-(connected:Entity)
                    RETURN path
                    LIMIT {limit};
                                """.strip()

    def delete_graph_by_doc_id(self, doc_id: str):
        """Delete graph by document ID"""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n:Entity {doc_id: $doc_id})
                DETACH DELETE n
                """,
                doc_id=doc_id,
            )
            summary = result.consume()
            deleted = summary.counters.nodes_deleted
            rel_deleted = summary.counters.relationships_deleted
            logger.info(
                f"Deleted {deleted} nodes with doc_id '{doc_id}' " f"and {rel_deleted} associated relationships"
            )


neo4j_client = Neo4jClient()
