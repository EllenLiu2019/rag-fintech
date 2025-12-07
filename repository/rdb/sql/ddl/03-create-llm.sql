-- connect to rag_fintech database
\c rag_fintech;

-- set schema search path
SET search_path TO rag_fintech, public;

-- drop existing table
DROP TABLE IF EXISTS rag_fintech.llm CASCADE;

-- create llm table
CREATE TABLE rag_fintech.llm (
    id                          BIGSERIAL      PRIMARY KEY,
    llm_provider                VARCHAR(60)    NOT NULL,
    model_name                  VARCHAR(60)    NOT NULL,
    model_type                  VARCHAR(60)    NOT NULL CHECK (model_type IN ('chat', 'reasoner', 'embedding')),
    max_tokens                  INTEGER        NOT NULL,
    max_tokens_context          INTEGER        ,
    created_time                TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description                 TEXT           ,
    CONSTRAINT unique_llm_provider_model_name UNIQUE (llm_provider, model_name)
);

-- create indexes
-- CREATE INDEX idx_llm_provider ON rag_fintech.llm(llm_provider);


