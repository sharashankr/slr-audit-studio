ALTER TABLE papers ADD COLUMN IF NOT EXISTS pdf_bytes BYTEA;
ALTER TABLE papers DROP CONSTRAINT IF EXISTS papers_sha256_key;
CREATE UNIQUE INDEX IF NOT EXISTS papers_workspace_sha256_key ON papers(workspace_id, sha256);
