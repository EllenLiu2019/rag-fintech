-- connect to rag_fintech database
\c rag_fintech;

-- set schema search path
SET search_path TO rag_fintech, public;

-- truncate table (delete all data)
TRUNCATE TABLE rag_fintech.llm;

-- insert LLM configurations
INSERT INTO rag_fintech.llm 
(provider, model_name, model_type, max_tokens, max_tokens_context, created_time, description) 
VALUES 
('DeepSeek', 'deepseek-reasoner', 'reasoner', 64000, 128000, CURRENT_TIMESTAMP, 'model: DeepSeek-V3.2'),
('Voyage', 'voyage-3-lite', 'embedding', 32000, NULL, CURRENT_TIMESTAMP, 'dimension: 512'),
('Voyage', 'voyage-3.5-lite', 'embedding', 32000, NULL, CURRENT_TIMESTAMP, 'dimension: 1024, 256, 512, 2048'),
('Voyage', 'voyage-3.5', 'embedding', 32000, NULL, CURRENT_TIMESTAMP, 'dimension: 1024, 256, 512, 2048'),
('Voyage', 'voyage-3-large', 'embedding', 32000, NULL, CURRENT_TIMESTAMP, 'dimension: 1024, 256, 512, 2048'),
('Voyage', 'voyage-finance-2', 'embedding', 32000, NULL, CURRENT_TIMESTAMP, 'dimension: 1024, 256, 512, 2048'),
('Voyage', 'voyage-law-2', 'embedding', 16000, NULL, CURRENT_TIMESTAMP, 'dimension: 1024, 256, 512, 2048'),
('BAAI', 'bge-m3', 'embedding', 8192, NULL, CURRENT_TIMESTAMP, 'multilingual; dimension: 1024');