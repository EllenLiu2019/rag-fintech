DROP SCHEMA IF EXISTS snomed;
CREATE SCHEMA snomed AUTHORIZATION rag;

ALTER DATABASE rag_fintech SET search_path TO snomed, rag_fintech,public;

-- grant schema permissions
GRANT ALL ON SCHEMA snomed TO rag;
GRANT USAGE ON SCHEMA snomed TO rag;


-- Grant permissions (existing tables/sequence)
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA snomed TO rag;


-- Grant default privileges (for future tables/sequence)
ALTER DEFAULT PRIVILEGES IN SCHEMA snomed 
    GRANT ALL ON TABLES TO rag;
ALTER DEFAULT PRIVILEGES IN SCHEMA snomed 
    GRANT ALL ON SEQUENCES TO rag;

CREATE TABLE snomed.concept (
    concept_id       BIGINT       PRIMARY KEY,
    concept_name     VARCHAR(255) NOT NULL,
    domain_id        VARCHAR(20)  NOT NULL,
    vocabulary_id    VARCHAR(20)  NOT NULL,
    concept_class_id VARCHAR(20)  NOT NULL,
    standard_concept VARCHAR(1)   NULL,
    concept_code     VARCHAR(50)  NOT NULL,
    valid_start_date DATE         NOT NULL,
    valid_end_date   DATE         NOT NULL,
    invalid_reason   VARCHAR(1)   NULL
);

CREATE TABLE snomed.concept_synonym (
    id                     BIGSERIAL PRIMARY KEY,
    concept_id             BIGINT NOT NULL,
	concept_synonym_name   VARCHAR(1000) NOT NULL,
    language_concept_id    BIGINT NOT NULL,
    CONSTRAINT fk_concept_synonym_concept FOREIGN KEY (concept_id) 
        REFERENCES snomed.concept(concept_id) ON DELETE RESTRICT
    CONSTRAINT fk_concept_synonym_language_concept FOREIGN KEY (language_concept_id) 
        REFERENCES snomed.concept(concept_id) ON DELETE RESTRICT
);

CREATE TABLE snomed.concept_relationship (
    id                     BIGSERIAL   PRIMARY KEY,
    concept_id_1           BIGINT      NOT NULL,
    concept_id_2           BIGINT      NOT NULL,
    relationship_id        VARCHAR(20) NOT NULL,
    valid_start_date       DATE        NOT NULL,
    valid_end_date         DATE        NOT NULL,
    invalid_reason         VARCHAR(1)  NULL,
    CONSTRAINT fk_concept_relationship_concept_1 FOREIGN KEY (concept_id_1) 
        REFERENCES snomed.concept(concept_id) ON DELETE RESTRICT
    CONSTRAINT fk_concept_relationship_concept_2 FOREIGN KEY (concept_id_2) 
        REFERENCES snomed.concept(concept_id) ON DELETE RESTRICT
);


CREATE TABLE snomed.concept_ancestor (
    id                       BIGSERIAL  PRIMARY KEY,
    ancestor_concept_id      BIGINT     NOT NULL,
    descendant_concept_id    BIGINT     NOT NULL,
    min_levels_of_separation INTEGER    NOT NULL,
    max_levels_of_separation INTEGER    NOT NULL
    CONSTRAINT fk_concept_ancestor_concept_1 FOREIGN KEY (ancestor_concept_id) 
        REFERENCES snomed.concept(concept_id) ON DELETE RESTRICT
    CONSTRAINT fk_concept_ancestor_concept_2 FOREIGN KEY (descendant_concept_id) 
        REFERENCES snomed.concept(concept_id) ON DELETE RESTRICT
);