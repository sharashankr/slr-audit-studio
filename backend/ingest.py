"""Phase 1 - PDF -> structured chunks with page/bbox provenance -> Postgres.

pip install docling psycopg[binary] pgvector
Deterministic: no LLM calls here. Embeddings are injected (embed_fn).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import psycopg
from docling.document_converter import DocumentConverter

MAX_CHARS = 2400          # ~600 tokens; split long sections on paragraph boundaries
PARSER_NAME = "docling"

_SKIP_EMBED_RE = re.compile(r"(^|\s|>)(references|bibliography|works cited|acknowledg(e)?ments)(\s|$)", re.I)

@dataclass
class Chunk:
    ordinal: int
    section: str
    kind: str                       # text | table | figure | caption
    text: str
    pages: set[int] = field(default_factory=set)
    bboxes: list[dict] = field(default_factory=list)
    needs_vlm: bool = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bboxes(item, page_sizes: dict[int, tuple[float, float]]) -> list[dict]:
    """Docling bboxes -> top-left-origin points, so the PDF viewer needs no conversion."""
    out = []
    for prov in getattr(item, "prov", []) or []:
        w, h = page_sizes[prov.page_no]
        bb = prov.bbox.to_top_left_origin(page_height=h)
        out.append({"page": prov.page_no, "l": bb.l, "t": bb.t, "r": bb.r, "b": bb.b, "w": w, "h": h})
    return out


def parse_pdf(path: Path):
    converter = DocumentConverter()          # enable table-structure + OCR via pipeline options as needed
    result = converter.convert(str(path))
    doc = result.document
    page_sizes = {n: (p.size.width, p.size.height) for n, p in doc.pages.items()}
    return doc, page_sizes


def build_chunks(doc, page_sizes) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    cur: Chunk | None = None

    def section_path() -> str:
        return " > ".join(t for _, t in heading_stack) or "Front matter"

    def flush():
        nonlocal cur
        if cur and cur.text.strip():
            cur.ordinal = len(chunks)
            chunks.append(cur)
        cur = None

    for item, level in doc.iterate_items():
        label = str(getattr(item, "label", "")).lower().split(".")[-1]
        boxes = _bboxes(item, page_sizes)
        pages = {b["page"] for b in boxes}

        if label in ("section_header", "title"):
            flush()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, item.text.strip()))
            continue

        if label == "table":
            flush()
            md = item.export_to_markdown(doc=doc)
            # Heuristic low-confidence layout: empty/degenerate table -> route to VLM on the cropped bbox
            bad = (not md.strip()) or md.count("|") < 6 or len(getattr(item.data, "table_cells", [])) < 4
            chunks.append(Chunk(len(chunks), section_path(), "table", md, pages, boxes, needs_vlm=bad))
            continue

        if label == "picture":
            flush()
            chunks.append(Chunk(len(chunks), section_path(), "figure", "[figure]", pages, boxes, needs_vlm=True))
            continue

        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        kind = "caption" if label == "caption" else "text"
        if kind == "caption":
            flush()
            chunks.append(Chunk(len(chunks), section_path(), "caption", text, pages, boxes))
            continue

        # Start a new chunk on section change or size overflow.
        if cur is None or cur.section != section_path() or len(cur.text) + len(text) > MAX_CHARS:
            flush()
            cur = Chunk(0, section_path(), "text", "")
        cur.text += ("\n\n" if cur.text else "") + text
        cur.pages |= pages
        cur.bboxes += boxes
    flush()
    return chunks


def extract_metadata(doc, chunks: Sequence[Chunk]) -> dict:
    title = next((c.text for c in chunks if c.section == "Front matter"), None)
    abstract = None
    for c in chunks:
        if re.match(r"(?i)^abstract", c.section.split(" > ")[-1]):
            abstract = c.text
            break
    doi = None
    m = re.search(r"\b10\.\d{4,9}/[^\s\"<>]+", " ".join(c.text for c in chunks[:6]))
    if m:
        doi = m.group(0).rstrip(".,;)")
    return {"title": title, "abstract": abstract, "doi": doi}   # prefer GROBID output when available


def ingest_pdf(
    conn: psycopg.Connection,
    path: Path,
    pdf_uri: str,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    workspace_id: str = "default",
) -> str:
    digest = sha256_file(path)
    row = conn.execute("SELECT paper_id FROM papers WHERE sha256=%s AND workspace_id=%s", (digest, workspace_id)).fetchone()
    if row:
        return str(row[0])                       # idempotent: same bytes -> same record

    doc, page_sizes = parse_pdf(path)
    chunks = build_chunks(doc, page_sizes)
    meta = extract_metadata(doc, chunks)
    import docling
    with conn.transaction():
        paper_id = conn.execute(
            """INSERT INTO papers (sha256, doi, title, abstract, pdf_uri, parser_name, parser_version, page_sizes, workspace_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING paper_id""",
            (digest, meta["doi"], meta["title"], meta["abstract"], pdf_uri, PARSER_NAME,
             getattr(docling, "__version__", "unknown"),
             json.dumps({str(k): list(v) for k, v in page_sizes.items()}),
             workspace_id),
        ).fetchone()[0]
        to_embed_idx = [
            i for i, c in enumerate(chunks)
            if c.kind != "figure" and not _SKIP_EMBED_RE.search(c.section)
        ]
        embeddings: list[list[float] | None] = [None] * len(chunks)
        if embed_fn and to_embed_idx:
            vecs = embed_fn([chunks[i].text for i in to_embed_idx])
            for idx, vec in zip(to_embed_idx, vecs):
                embeddings[idx] = vec
        for c, emb in zip(chunks, embeddings):
            conn.execute(
                """INSERT INTO chunks (paper_id, ordinal, section, kind, text, page_start, page_end,
                                       bboxes, needs_vlm, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (paper_id, c.ordinal, c.section, c.kind, c.text,
                 min(c.pages or {1}), max(c.pages or {1}),
                 json.dumps(c.bboxes), c.needs_vlm, emb),
            )
        conn.execute("INSERT INTO audit_log (actor, action, paper_id) VALUES ('system:ingest','ingest',%s)", (paper_id,))
    return str(paper_id)


# ---------- Retrieval helper used by every downstream agent (hybrid keyword + vector, RRF) ----------
HYBRID_SQL = """
WITH kw AS (
  SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %(q)s)) DESC) r
  FROM chunks WHERE paper_id=%(pid)s AND tsv @@ websearch_to_tsquery('english', %(q)s) LIMIT 20),
sem AS (
  SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(qv)s::vector) r
  FROM chunks WHERE paper_id=%(pid)s AND embedding IS NOT NULL LIMIT 20)
SELECT c.chunk_id, c.section, c.kind, c.text, c.page_start, c.bboxes,
       coalesce(1.0/(60+kw.r),0) + coalesce(1.0/(60+sem.r),0) AS score
FROM chunks c LEFT JOIN kw USING (chunk_id) LEFT JOIN sem USING (chunk_id)
WHERE kw.chunk_id IS NOT NULL OR sem.chunk_id IS NOT NULL
ORDER BY score DESC LIMIT %(k)s
"""


def retrieve(conn, paper_id: str, query: str, query_vec: list[float], k: int = 6) -> list[dict]:
    cur = conn.execute(HYBRID_SQL, {"q": query, "pid": paper_id, "qv": query_vec, "k": k})
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
