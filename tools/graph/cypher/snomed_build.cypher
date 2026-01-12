// -----------------------------------------------------------------------------------------
// Concept: Update a SNOMED_G Graph Database from input CSV files which describe the changes
//          to concepts, descriptions, ISA relationships and defining relationships.
// Input Files:
//          concept_new.csv
//          concept_cn_new.csv (incremental update)
//          concept_nccd_new.csv (incremental update)
//          descrip_new.csv
//          isa_rel_new.csv
//          isa_rel_cn_new.csv (incremental update)
//          maps_to_new.csv (ICD10CN -> SNOMED mapping)
//          maps_to_nccd_new.csv (NCCD -> SNOMED mapping)

// NEXT STEP -- create INDEXES

CREATE CONSTRAINT id_Concept_uniq FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT conceptId_Concept_uniq FOR (c:Concept) REQUIRE c.conceptId IS UNIQUE;
   // id,conceptId index created, requiring uniqueness
   // Note: Can't have "FSN is UNIQUE"" constraint, can have dups (inactive concepts)
   //      for example -- "retired procedure" is FSN of multiple inactive concepts
CREATE INDEX vocabularyId_Concept_index FOR (c:Concept) ON (c.vocabularyId);
   // vocabularyId index for filtering SNOMED vs ICD10CN vs NCCD concepts
CREATE CONSTRAINT id_Desc_uniq FOR (c:Description) REQUIRE c.id IS UNIQUE;
CREATE INDEX conceptId_Desc_index FOR (x:Description) ON (x.conceptId);
  // need index so setting HAS_DESCRIPTION edges doesn't stall
  // there can be more than one description for the same conceptId, conceptId not unique, but id is unique

// ROLE_GROUP nodes.  Index needed for defining relationship assignment.
// CREATE INDEX FOR (x:RoleGroup) ON (x.sctid);

// NEXT STEP -- create CONCEPT nodes in /var/lib/neo4j/import/

RETURN 'Creating NEW SNOMED Concept nodes';
LOAD csv with headers from "file:///concept_new.csv" as line
CALL {
    WITH line
    CREATE (:Concept
        {id:                toInteger(line.id),
        conceptId:          toInteger(line.concept_id),
        FSN:                line.FSN,
        vocabularyId:       "SNOMED",
        effectiveDate:      toInteger(line.valid_start_date),
        active:             toInteger(line.active)
        })
} IN TRANSACTIONS OF 200 ROWS;

// NEXT STEP -- create/update ICD10CN CONCEPT nodes (incremental)
RETURN 'Creating/Updating ICD10CN Concept nodes';
LOAD csv with headers from "file:///concept_cn_new.csv" as line
CALL {
    WITH line
    MERGE (c:Concept { id: toInteger(line.id) })
    SET c.conceptId = toInteger(line.concept_id),
        c.FSN = line.FSN,
        c.vocabularyId = "ICD10CN",
        c.effectiveDate = toInteger(line.valid_start_date),
        c.active = toInteger(line.active)
} IN TRANSACTIONS OF 200 ROWS;

// NEXT STEP -- create NCCD CONCEPT nodes (incremental)
RETURN 'Creating NCCD Concept nodes';
LOAD csv with headers from "file:///concept_nccd_new.csv" as line
CALL {
    WITH line
    CREATE (:Concept
    {id:                toInteger(line.id),
        conceptId:          toInteger(line.concept_id),
        FSN:                line.FSN,
        vocabularyId:       "NCCD",
        effectiveDate:      toInteger(line.valid_start_date),
        active:             toInteger(line.active)
        })
} IN TRANSACTIONS OF 1000 ROWS;

// NEXT STEP -- create DESCRIPTION nodes (info from Language+Description file)

RETURN 'Creating NEW Description nodes';

LOAD csv with headers from "file:///descrip_new.csv" as line
CALL {
    WITH line
    CREATE (:Description   
        {id:                toInteger(line.id),
        conceptId:          toInteger(line.concept_id),
        synonym:            line.concept_synonym_name,
        languageConceptId:  toInteger(line.language_concept_id),
        active:             toInteger(line.active)
        } )
} IN TRANSACTIONS OF 200 ROWS;

// NEXT STEP - create DESCRIPTION edges
RETURN 'Creating HAS_DESCRIPTION edges for new Description nodes related to Concept nodes';

LOAD csv with headers from "file:///descrip_new.csv" as line
CALL {
    WITH line
    MATCH (c:Concept { conceptId: toInteger(line.concept_id) }), (f:Description { id: toInteger(line.id) })
    MERGE (c)-[:HAS_DESCRIPTION]->(f) 
} IN TRANSACTIONS OF 200 ROWS;

// --------------------------------------------------------------------------------------
// NEXT STEP -- create ISA relationships
// --------------------------------------------------------------------------------------

RETURN 'Creating NEW ISA edges';

LOAD csv with headers from "file:///isa_rel_new.csv" as line
CALL {
    WITH line
    MATCH (c1:Concept { id: toInteger(line.source_id) }), (c2:Concept { id: toInteger(line.dest_id) })
    MERGE (c1)-[r:ISA { id: toInteger(line.id) }]->(c2)
    SET r.sourceId = toInteger(line.source_id),
        r.destId = toInteger(line.dest_id),
        r.typeId = toInteger(line.type_id),
        r.effectiveDate = toInteger(line.valid_start_date),
        r.active = toInteger(line.active)
} IN TRANSACTIONS OF 200 ROWS;

// NEXT STEP -- create/update ISA relationships from incremental changes
RETURN 'Creating/Updating ISA relationships from incremental changes';

LOAD csv with headers from "file:///isa_rel_cn_new.csv" as line
CALL {
    WITH line
    MATCH (c1:Concept { id: toInteger(line.source_id) }), (c2:Concept { id: toInteger(line.dest_id) })
    MERGE (c1)-[r:ISA { id: toInteger(line.id) }]->(c2)
    SET r.sourceId = toInteger(line.source_id),
        r.destId = toInteger(line.dest_id),
        r.typeId = toInteger(line.type_id),
        r.effectiveDate = toInteger(line.valid_start_date),
        r.active = toInteger(line.active)
} IN TRANSACTIONS OF 1000 ROWS;

// --------------------------------------------------------------------------------------
// NEXT STEP -- create MAPS_TO relationships (ICD10CN -> SNOMED mapping)
// --------------------------------------------------------------------------------------

RETURN 'Creating MAPS_TO edges (ICD10CN -> SNOMED)';

LOAD csv with headers from "file:///maps_to_new.csv" as line
CALL {
    WITH line
    MATCH (c1:Concept { id: toInteger(line.source_id) }), (c2:Concept { id: toInteger(line.dest_id) })
    MERGE (c1)-[r:MAPS_TO { id: toInteger(line.id) }]->(c2)
    SET r.sourceId = toInteger(line.source_id),
        r.destId = toInteger(line.dest_id),
        r.typeId = toInteger(line.type_id),
        r.effectiveDate = toInteger(line.valid_start_date),
        r.active = toInteger(line.active)
} IN TRANSACTIONS OF 1000 ROWS;

// --------------------------------------------------------------------------------------
// NEXT STEP -- create MAPS_TO relationships (NCCD -> SNOMED mapping)
// --------------------------------------------------------------------------------------

// need to import vocabularyId = RxNorm , or else we got only 36 mappings for now

RETURN 'Creating MAPS_TO edges (NCCD -> SNOMED)';

LOAD csv with headers from "file:///maps_to_nccd_new.csv" as line
CALL {
    WITH line
    MATCH (c1:Concept { id: toInteger(line.source_id) }), (c2:Concept { id: toInteger(line.dest_id) })
    MERGE (c1)-[r:MAPS_TO { id: toInteger(line.id) }]->(c2)
    SET r.sourceId = toInteger(line.source_id),
        r.destId = toInteger(line.dest_id),
        r.typeId = toInteger(line.type_id),
        r.effectiveDate = toInteger(line.valid_start_date),
        r.active = toInteger(line.active)
} IN TRANSACTIONS OF 1000 ROWS;
