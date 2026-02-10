DROP TABLE IF EXISTS rag_fintech.claim_evaluations;
CREATE TABLE IF NOT EXISTS rag_fintech.claim_evaluations (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL,                  -- claim document ID
    patient_id      TEXT,                           -- patient identifier
    entity_index    INTEGER NOT NULL,               -- which entity in the claim
    entity_name     TEXT,                           -- display name (e.g., 甲状腺乳头状癌)
    thread_id       TEXT NOT NULL UNIQUE,           -- LangGraph thread_id
    status          TEXT NOT NULL DEFAULT 'pending', -- pending / reviewing / approved / completed / rejected
    human_decision  JSONB,                          -- the decision made by human reviewer
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for common queries
CREATE INDEX idx_claim_eval_doc_id ON rag_fintech.claim_evaluations(doc_id);
CREATE INDEX idx_claim_eval_thread_id ON rag_fintech.claim_evaluations(thread_id);
CREATE INDEX idx_claim_eval_status ON rag_fintech.claim_evaluations(status);