-- connect to rag_fintech database
\c rag_fintech;

-- set schema search path
SET search_path TO rag_fintech, public;

-- truncate table (delete all data)
TRUNCATE TABLE rag_fintech.knowledgebase;

-- insert knowledge base
INSERT INTO rag_fintech.knowledgebase 
(kb_name, embed_llm_id, update_time, description) 
VALUES 
('default_kb', 3, CURRENT_TIMESTAMP, 'Default knowledge base using voyage-3-lite embedding');