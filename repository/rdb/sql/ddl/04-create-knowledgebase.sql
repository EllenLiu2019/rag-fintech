-- connect to rag_fintech database
\c rag_fintech;

-- set schema search path
SET search_path TO rag_fintech, public;

-- drop existing table
DROP TABLE IF EXISTS rag_fintech.knowledgebase CASCADE;

-- create knowledgebase table
CREATE TABLE rag_fintech.knowledgebase (
    id                          BIGSERIAL      PRIMARY KEY,
    kb_name                     VARCHAR(60)    NOT NULL UNIQUE,
    embed_llm_id                BIGINT         NOT NULL,
    doc_num                     INTEGER         DEFAULT 0,
    chunk_num                   INTEGER         DEFAULT 0,
    token_num                   INTEGER         DEFAULT 0,
    update_time                 TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description                 TEXT,
    CONSTRAINT fk_knowledgebase_embed_llm FOREIGN KEY (embed_llm_id) 
        REFERENCES rag_fintech.llm(id) ON DELETE RESTRICT
);

CREATE INDEX idx_knowledgebase_name ON rag_fintech.knowledgebase(kb_name);

