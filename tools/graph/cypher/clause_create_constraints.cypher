// ==================================================================================
// Neo4j Constraints and Indexes for GraphRAG Entity Storage
// ==================================================================================
// 
// This script creates the necessary constraints and indexes for the Entity graph.
// 
// Entity Model:
//   - id: Numeric primary key (deterministic hash of entity_name + clause_id + doc_id)
//   - entity_name: Name of the entity
//   - clause_id: ID of the clause where entity appears
//   - doc_id: ID of the source document
//   - entity_type: Type of entity (CLAUSE, COVERAGE_SCOPE, etc.)
//   - description: Entity description
//   - chunk_id: Associated chunk IDs
//
// Uniqueness: (entity_name, clause_id, doc_id) must be unique
// Primary Key: id (numeric)
//
// Usage:
//   cypher-shell -u neo4j -p password < create_constraints.cypher
//   Or copy-paste into Neo4j Browser
// ==================================================================================

// Drop existing constraints if they exist (for clean re-creation)
// Note: Comment these out if you want to keep existing constraints
DROP CONSTRAINT entity_id_unique IF EXISTS;
DROP CONSTRAINT entity_composite_unique IF EXISTS;

// ==================================================================================
// PRIMARY KEY CONSTRAINT
// ==================================================================================
// Create unique constraint on Entity.id (numeric primary key)
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.id IS UNIQUE;

// ==================================================================================
// COMPOSITE UNIQUE CONSTRAINT
// ==================================================================================
// Create unique constraint on (entity_name, clause_id, doc_id)
// This ensures the same entity name can appear in different clauses/documents
CREATE CONSTRAINT entity_composite_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE (e.entity_name, e.clause_id, e.doc_id) IS UNIQUE;

// ==================================================================================
// INDEXES FOR QUERY PERFORMANCE
// ==================================================================================

// Index on entity_name for fast entity lookup
CREATE INDEX entity_name_index IF NOT EXISTS
FOR (e:Entity)
ON (e.entity_name);

// Index on doc_id for document-level queries
CREATE INDEX entity_doc_id_index IF NOT EXISTS
FOR (e:Entity)
ON (e.doc_id);

// Index on clause_id for clause-level queries
CREATE INDEX entity_clause_id_index IF NOT EXISTS
FOR (e:Entity)
ON (e.clause_id);

// Index on entity_type for type-based filtering
CREATE INDEX entity_type_index IF NOT EXISTS
FOR (e:Entity)
ON (e.entity_type);

// Composite index on (doc_id, clause_id) for common query pattern
CREATE INDEX entity_doc_clause_index IF NOT EXISTS
FOR (e:Entity)
ON (e.doc_id, e.clause_id);

// ==================================================================================
// VERIFY CONSTRAINTS AND INDEXES
// ==================================================================================

// Show all constraints
SHOW CONSTRAINTS;

// Show all indexes
SHOW INDEXES;

// ==================================================================================
// EXAMPLE QUERIES
// ==================================================================================

// Query by numeric ID (fastest)
// MATCH (e:Entity {id: 123456789}) RETURN e;

// Query by entity name (will use index)
// MATCH (e:Entity {entity_name: "保险责任"}) RETURN e;

// Query by composite key (will use composite unique constraint)
// MATCH (e:Entity {entity_name: "保险责任", clause_id: 3, doc_id: "doc_001"}) RETURN e;

// Query all entities in a document (will use doc_id index)
// MATCH (e:Entity {doc_id: "doc_001"}) RETURN e;

// Query all entities of a type (will use entity_type index)
// MATCH (e:Entity {entity_type: "CLAUSE"}) RETURN e;

// Count entities per document
// MATCH (e:Entity) RETURN e.doc_id, count(e) AS entity_count ORDER BY entity_count DESC;

// Find duplicate entity names across different clauses
// MATCH (e:Entity)
// WITH e.entity_name AS name, collect({clause_id: e.clause_id, doc_id: e.doc_id}) AS locations
// WHERE size(locations) > 1
// RETURN name, locations;

