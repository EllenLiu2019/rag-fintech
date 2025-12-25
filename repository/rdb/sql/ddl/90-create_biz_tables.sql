
-- drop existing table
DROP TABLE IF EXISTS rag_fintech.policy_holder;
DROP TABLE IF EXISTS rag_fintech.insured;
DROP TABLE IF EXISTS rag_fintech.coverage;
DROP TABLE IF EXISTS rag_fintech.cvg_premium;
DROP TABLE IF EXISTS rag_fintech.policy;

-- 保单主表
CREATE TABLE rag_fintech.policy (
    id               BIGSERIAL PRIMARY KEY,
    policy_number    VARCHAR(50),
    effective_date   DATE,
    expiry_date      DATE,
    source_file      VARCHAR(200),
    extraction_time  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score FLOAT,
    status           VARCHAR(1) DEFAULT 'A' CHECK (status IN ('A', 'I'))
);

-- 投保人表
CREATE TABLE rag_fintech.policy_holder (
    id                  BIGSERIAL   PRIMARY KEY,
    policy_number       VARCHAR(50),
    name                VARCHAR(100),
    gender              VARCHAR(10),
    birth_date          DATE,
    id_number           VARCHAR(50),
    phone               VARCHAR(100),
    email               VARCHAR(100),
    extraction_time  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status           VARCHAR(1) DEFAULT 'A' CHECK (status IN ('A', 'I'))
);

-- 被保险人表
CREATE TABLE rag_fintech.insured (
    id                     BIGSERIAL   PRIMARY KEY,
    policy_number          VARCHAR(50) ,
    name                   VARCHAR(100),
    gender                 VARCHAR(10),
    birth_date             DATE,
    id_number              VARCHAR(50),
    relationship_to_holder VARCHAR(50),
    extraction_time        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status                 VARCHAR(1) DEFAULT 'A' CHECK (status IN ('A', 'I'))
);

-- 保险条款表
CREATE TABLE rag_fintech.coverage (
    id             BIGSERIAL PRIMARY KEY,
    policy_number  VARCHAR(50),
    cvg_name       VARCHAR(200),
    cvg_type       VARCHAR(200),
    cvg_amt        DECIMAL(15, 2),
    extraction_time  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status         VARCHAR(1) DEFAULT 'A' CHECK (status IN ('A', 'I'))
);

-- 保险费表
CREATE TABLE rag_fintech.cvg_premium (
    id             BIGSERIAL PRIMARY KEY,
    policy_number  VARCHAR(50),
    cvg_name       VARCHAR(200),
    cvg_premium    DECIMAL(15, 2),
    extraction_time  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status         VARCHAR(1) DEFAULT 'A' CHECK (status IN ('A', 'I'))
);

-- 索引优化
CREATE INDEX idx_policy_holder_id_number ON rag_fintech.policy_holder(id_number);
CREATE INDEX idx_policy_holder_policy_number ON rag_fintech.policy_holder(policy_number);
CREATE INDEX idx_insured_id_number ON rag_fintech.insured(id_number);
CREATE INDEX idx_insured_policy_number ON rag_fintech.insured(policy_number);
CREATE INDEX idx_coverage_policy_number ON rag_fintech.coverage(policy_number);
CREATE INDEX idx_cvg_premium_policy_number ON rag_fintech.cvg_premium(policy_number);