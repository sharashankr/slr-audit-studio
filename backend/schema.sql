CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS papers (
    paper_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256 TEXT UNIQUE NOT NULL,
    doi TEXT,
    title TEXT,
    abstract TEXT,
    year INT,
    authors JSONB DEFAULT '[]'::jsonb,
    pdf_uri TEXT NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    page_sizes JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    ordinal INT NOT NULL,
    section TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    page_start INT NOT NULL,
    page_end INT NOT NULL,
    bboxes JSONB NOT NULL DEFAULT '[]'::jsonb,
    needs_vlm BOOLEAN NOT NULL DEFAULT FALSE,
    embedding vector(768),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_ord ON chunks(paper_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN(tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_emb ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS rubrics (
    rubric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    spec JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS screening_decisions (
    id BIGSERIAL PRIMARY KEY,
    rubric_id UUID NOT NULL REFERENCES rubrics(rubric_id) ON DELETE CASCADE,
    paper_id UUID NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    decision TEXT NOT NULL,
    exclusion_code TEXT,
    rationale TEXT,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    human_override TEXT,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extraction_cells (
    cell_id BIGSERIAL PRIMARY KEY,
    rubric_id UUID NOT NULL REFERENCES rubrics(rubric_id) ON DELETE CASCADE,
    paper_id UUID NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    criterion_id TEXT NOT NULL,
    value JSONB,
    verbatim_quote TEXT,
    chunk_id UUID REFERENCES chunks(chunk_id),
    page_number INT,
    bbox JSONB,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    critic_status TEXT NOT NULL,
    critic_notes TEXT,
    attempts INT NOT NULL DEFAULT 1,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE OR REPLACE VIEW current_cells AS
SELECT DISTINCT ON (rubric_id, paper_id, criterion_id)
    cell_id, rubric_id, paper_id, criterion_id, value, verbatim_quote,
    chunk_id, page_number, bbox, confidence, critic_status, critic_notes,
    attempts, model_version, created_at
FROM extraction_cells
ORDER BY rubric_id, paper_id, criterion_id, cell_id DESC;

CREATE TABLE IF NOT EXISTS audit_log (
    log_id BIGSERIAL PRIMARY KEY,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    rubric_id UUID,
    paper_id UUID,
    criterion_id TEXT,
    before JSONB,
    after JSONB,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);