-- connect to rag_fintech database
\c rag_fintech;

-- create schema (if not exists)
DROP SCHEMA IF EXISTS rag_fintech CASCADE;
CREATE SCHEMA rag_fintech AUTHORIZATION rag;

-- set default search path
ALTER DATABASE rag_fintech SET search_path TO rag_fintech, public;

-- grant schema permissions
GRANT ALL ON SCHEMA rag_fintech TO rag;
GRANT USAGE ON SCHEMA rag_fintech TO rag;


-- Grant permissions (existing tables/sequence)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag_fintech TO rag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rag_fintech TO rag;


-- Grant default privileges (for future tables/sequence)
ALTER DEFAULT PRIVILEGES IN SCHEMA rag_fintech 
    GRANT ALL ON TABLES TO rag;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag_fintech 
    GRANT ALL ON SEQUENCES TO rag;