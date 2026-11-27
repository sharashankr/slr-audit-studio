-- Migration: add workspace_id to papers and rubrics for multi-user isolation.
-- Safe to run multiple times (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).

ALTER TABLE papers
    ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE rubrics
    ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

-- Index for fast per-workspace paper lookups
CREATE INDEX IF NOT EXISTS idx_papers_workspace ON papers(workspace_id);
CREATE INDEX IF NOT EXISTS idx_rubrics_workspace ON rubrics(workspace_id);
