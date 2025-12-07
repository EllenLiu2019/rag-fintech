-- connect to rag_fintech database
\c rag_fintech;

-- set schema search path
SET search_path TO rag_fintech, public;

-- drop existing table
DROP TABLE IF EXISTS rag_fintech.document CASCADE;

-- create document table
CREATE TABLE rag_fintech.document (
    id                          BIGSERIAL      PRIMARY KEY,
    document_id                 VARCHAR(100)   UNIQUE,
    file_name                   VARCHAR(255)   NOT NULL,
    doc_status                  VARCHAR(60)    NOT NULL DEFAULT 'uploaded',
    file_location               VARCHAR(500),
    doc_location                VARCHAR(500),
    content_type                VARCHAR(60),
    page_count                  INTEGER        DEFAULT 0,
    chunk_num                   INTEGER        DEFAULT 0,
    token_num                   INTEGER        DEFAULT 0,
    file_size                   INTEGER        DEFAULT 0,
    upload_time                 TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    kb_name                     VARCHAR(60)    NOT NULL,
    business_data               JSONB,
    confidence                  JSONB,
    CONSTRAINT fk_document_kb FOREIGN KEY (kb_name) REFERENCES rag_fintech.knowledgebase(kb_name) ON DELETE RESTRICT,
    CONSTRAINT chk_doc_status CHECK (doc_status IN ('uploaded', 'processing', 'completed', 'failed'))
);

-- create indexes
CREATE INDEX idx_document_id ON rag_fintech.document(document_id);
--CREATE INDEX idx_document_kb_id ON rag_fintech.document(kb_id);
--CREATE INDEX idx_document_status ON rag_fintech.document(doc_status);
--CREATE INDEX idx_document_upload_time ON rag_fintech.document(upload_time DESC);

-- create indexes for jsonb fields
--CREATE INDEX idx_document_business_data ON rag_fintech.document USING GIN (business_data);
--CREATE INDEX idx_document_confidence ON rag_fintech.document USING GIN (confidence);


