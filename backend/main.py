"""FastAPI backend entrypoint for SLR-Audit Studio (Free-Tier Gemini Configuration).

Wires the 5 pipeline modules (ingest, rubric_screening, grounded_extraction, export_eval)
to PostgreSQL + pgvector and Google AI Studio's free-tier Gemini 3.x models.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import psycopg
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

from export_eval import draft_synthesis, export_excel, load_matrix
from grounded_extraction import extract_paper
from ingest import ingest_pdf, retrieve
from rubric_screening import (
    Rubric,
    compile_rubric,
    dry_run,
    lint_rubric,
    prisma_counts,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Environment & Free-Tier Model Configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ["DATABASE_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Free-tier eligible models in Google AI Studio (no billing account required):
# - gemini-3.5-flash-lite: high-throughput structured screening & extraction
# - gemini-3.8-flash: independent critic verification & cross-paper synthesis
# - gemini-embedding-001: replaces retired text-embedding-004 (Jan 2026)
def _parse_models(multi_env: str, single_env: str, default: str) -> list[str]:
    raw = os.getenv(multi_env) or os.getenv(single_env) or default
    return [m.strip() for m in raw.split(",") if m.strip()]


EXTRACT_MODELS = _parse_models(
    "GEMINI_EXTRACT_MODELS",
    "GEMINI_EXTRACT_MODEL",
    "gemini-3.5-flash-lite,gemini-2.5-flash-lite,gemini-2.5-flash",
)
VERIFY_MODELS = _parse_models(
    "GEMINI_VERIFY_MODELS",
    "GEMINI_VERIFY_MODEL",
    "gemini-2.5-flash,gemini-3.8-flash,gemini-3.5-flash-lite",
)
MODEL_EXTRACT = EXTRACT_MODELS[0]
MODEL_VERIFY = VERIFY_MODELS[0]
MODEL_EMBED = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Keep concurrency low (default=2) so parallel workers do not burst past
# Google AI Studio's free-tier Requests-Per-Minute (RPM) cap.
EXTRACT_CONCURRENCY = int(os.getenv("EXTRACT_CONCURRENCY", "2"))

# Per-model proactive RPM cap so requests pace cleanly instead of triggering 429 bursts
LLM_RPM = int(os.getenv("GEMINI_RPM_LIMIT", "10"))

PDF_DIR = Path(os.getenv("PDF_DIR", "./data/pdfs"))
PDF_DIR.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(title="SLR-Audit Studio API (Free Tier)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve uploaded PDFs at /pdfs/<filename> so react-pdf can render bounding-box overlays
app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")


def get_conn() -> psycopg.Connection:
    """Open a psycopg3 connection to Neon/Postgres without idle-in-transaction locks."""
    return psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


def _workspace(request: Request) -> str:
    """Return the caller's workspace ID from the X-Workspace-ID header.
    Falls back to 'default' so existing data and the benchmark script still work."""
    return request.headers.get("X-Workspace-ID", "default") or "default"


def _l2_norm(vec: list[float]) -> list[float]:
    """L2-normalize truncated 768-d embeddings (required by gemini-embedding-001)."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


_RATE_STATE: dict[tuple[int, str], tuple[asyncio.Lock, deque[float]]] = {}
_QUERY_VEC_CACHE: dict[tuple[str, str], list[float]] = {}
_EXHAUSTED_MODELS: set[str] = set()


async def _wait_for_rate_slot(model: str, rpm: int = LLM_RPM) -> None:
    """Sliding-window per-model RPM limiter bound to the active event loop."""
    if rpm <= 0:
        return
    key = (id(asyncio.get_running_loop()), model)
    if key not in _RATE_STATE:
        _RATE_STATE[key] = (asyncio.Lock(), deque())
    lock, timestamps = _RATE_STATE[key]

    async with lock:
        while True:
            now = time.monotonic()
            while timestamps and now - timestamps[0] >= 60.0:
                timestamps.popleft()
            if len(timestamps) < rpm:
                timestamps.append(now)
                return
            sleep_for = 60.0 - (now - timestamps[0]) + 0.1
            print(f"[rate] {model} RPM cap ({rpm}); sleeping {sleep_for:.1f}s", flush=True)
            await asyncio.sleep(max(0.1, sleep_for))


# ---------------------------------------------------------------------------
# Free-Tier Gemini Structured Output & Embedding Adapters
# ---------------------------------------------------------------------------
def make_llm_json(models: str | list[str]):
    """Build an async LLMJson callable enforcing Pydantic JSON schema decoding
    with proactive RPM pacing and automatic multi-model failover on 429/503/404.
    """
    model_list = [models] if isinstance(models, str) else list(models)

    def _strip_additional_props(obj):
        if isinstance(obj, dict):
            obj.pop("additionalProperties", None)
            for v in obj.values():
                _strip_additional_props(v)
        elif isinstance(obj, list):
            for item in obj:
                _strip_additional_props(item)
        return obj

    async def _llm_json(system: str, user: str, schema: type[BaseModel]) -> dict:
        clean_schema = _strip_additional_props(schema.model_json_schema())
        last_err: Exception | None = None

        for round_idx in range(4):
            available = [m for m in model_list if m not in _EXHAUSTED_MODELS]
            if not available:
                break

            for idx, active_model in enumerate(available):
                has_fallback = idx < len(available) - 1
                max_tries = 1 if has_fallback else 3
                for attempt in range(max_tries):
                    await _wait_for_rate_slot(active_model)
                    try:
                        resp = await client.aio.models.generate_content(
                            model=active_model,
                            contents=user,
                            config=types.GenerateContentConfig(
                                system_instruction=system,
                                response_mime_type="application/json",
                                response_schema=clean_schema,
                                temperature=0.0,
                            ),
                        )
                        return json.loads(resp.text)
                    except Exception as e:
                        last_err = e
                        err = str(e)
                        err_low = err.lower()
                        is_daily = "perday" in err_low or "per_day" in err_low or "requests per day" in err_low
                        is_missing = "404" in err or "not_found" in err_low or "not found" in err_low
                        if is_daily or is_missing:
                            _EXHAUSTED_MODELS.add(active_model)
                            reason = "daily quota exhausted" if is_daily else "model unavailable (404)"
                            print(f"[llm] {active_model} {reason}; failing over", flush=True)
                            break

                        is_transient = "429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err
                        if is_transient:
                            if has_fallback:
                                print(f"[llm] {active_model} busy (429/503); failing over to {available[idx + 1]}", flush=True)
                                break
                            if attempt < max_tries - 1:
                                wait = min(45.0, (10 * (1.6**attempt))) + random.uniform(1.0, 3.0)
                                print(f"[llm] transient error on {active_model} (attempt {attempt + 1}/{max_tries}), sleeping {wait:.1f}s", flush=True)
                                await asyncio.sleep(wait)
                                continue
                        raise

            # If all remaining models had transient 429/503 in this round, cool down briefly before next round
            if any(m not in _EXHAUSTED_MODELS for m in model_list) and round_idx < 3:
                await asyncio.sleep(10.0)

        if last_err:
            raise last_err
        raise RuntimeError(f"No available models in {model_list}")

    return _llm_json


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed text chunks or queries into 768-d normalized vectors for pgvector."""
    if not texts:
        return []
    out: list[list[float]] = []
    # Slice into batches of 32 to stay within free-tier batch payload limits
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        for attempt in range(5):
            try:
                res = client.models.embed_content(
                    model=MODEL_EMBED,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=768,
                        task_type=task_type,
                    ),
                )
                out.extend(_l2_norm(e.values) for e in res.embeddings)
                break
            except Exception as e:
                err = str(e)
                if ("429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err) and attempt < 4:
                    time.sleep(min(60.0, 5.0 * (2**attempt)))
                    continue
                raise
    return out


async def aembed_query(query: str) -> list[float]:
    """Async cached query embedding so identical rubric queries are embedded at most once."""
    cache_key = (MODEL_EMBED, query)
    if cache_key in _QUERY_VEC_CACHE:
        return _QUERY_VEC_CACHE[cache_key]
    vec = await asyncio.to_thread(lambda: embed_texts([query], task_type="RETRIEVAL_QUERY")[0])
    _QUERY_VEC_CACHE[cache_key] = vec
    return vec


# ---------------------------------------------------------------------------
# API Request Schemas
# ---------------------------------------------------------------------------
class RubricReq(BaseModel):
    yaml_text: str


class RunReq(BaseModel):
    rubric_id: str
    paper_ids: list[str]
    force: bool = False


class ApproveReq(BaseModel):
    rubric_id: str
    cell_ids: list[int]


class OverrideReq(BaseModel):
    cell_id: int
    value: Any
    reason: str


# ---------------------------------------------------------------------------
# Phase 1–5 API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Verify server liveness and active free-tier model configuration."""
    return {
        "status": "ok",
        "models": {
            "extract": MODEL_EXTRACT,
            "verify": MODEL_VERIFY,
            "embed": MODEL_EMBED,
        },
        "extract_concurrency": EXTRACT_CONCURRENCY,
    }


@app.post("/api/ingest")
async def api_ingest(request: Request, file: UploadFile = File(...)):
    """Phase 1: Parse a PDF with Docling, extract bboxes, and index chunks in pgvector."""
    ws = _workspace(request)
    dest = PDF_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    with get_conn() as conn:
        paper_id = ingest_pdf(
            conn,
            dest,
            f"/pdfs/{file.filename}",
            embed_fn=lambda chunks: embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT"),
            workspace_id=ws,
        )
    return {"paper_id": paper_id, "pdf_url": f"/pdfs/{file.filename}"}


@app.post("/api/rubrics")
async def api_create_rubric(req: RubricReq, request: Request):
    """Phase 2a: Compile YAML/PICO criteria into typed rules and lint for ambiguity."""
    ws = _workspace(request)
    rubric = compile_rubric(req.yaml_text)
    spec_json = rubric.model_dump_json()
    with get_conn() as conn:
        existing = conn.execute(
            """SELECT rubric_id FROM rubrics
               WHERE name=%s AND version=%s AND spec = %s::jsonb
               ORDER BY created_at DESC LIMIT 1""",
            (rubric.name, rubric.version, spec_json),
        ).fetchone()
        if existing:
            return {"rubric_id": str(existing[0]), "rubric": rubric.model_dump(), "lint": {"ambiguous": []}}

        lint = await lint_rubric(rubric, make_llm_json(EXTRACT_MODELS))
        rubric_id = conn.execute(
            "INSERT INTO rubrics (name, version, spec, workspace_id) VALUES (%s, %s, %s, %s) RETURNING rubric_id",
            (rubric.name, rubric.version, spec_json, ws),
        ).fetchone()[0]
        conn.commit()
    return {"rubric_id": str(rubric_id), "rubric": rubric.model_dump(), "lint": lint.model_dump()}


@app.post("/api/rubrics/dry-run")
async def api_dry_run(req: RunReq):
    """Phase 2b: Test rubric consistency across multiple trials on 3–5 seed papers."""
    with get_conn() as conn:
        spec = conn.execute("SELECT spec FROM rubrics WHERE rubric_id=%s", (req.rubric_id,)).fetchone()[0]
        rubric = Rubric(**spec)
        report = await dry_run(
            conn, req.rubric_id, rubric, req.paper_ids, make_llm_json(EXTRACT_MODELS), MODEL_EXTRACT
        )
    return {"report": report}


@app.post("/api/screen")
async def api_screen(req: RunReq, request: Request):
    """Phase 2c: Run two-stage Abstract -> Methods screening gates and return PRISMA counts."""
    ws = _workspace(request)
    with get_conn() as conn:
        spec = conn.execute("SELECT spec FROM rubrics WHERE rubric_id=%s", (req.rubric_id,)).fetchone()[0]
        rubric = Rubric(**spec)
        survivors = await run_pipeline(
            conn, req.rubric_id, rubric, req.paper_ids, make_llm_json(EXTRACT_MODELS), MODEL_EXTRACT
        )
        prisma = prisma_counts(conn, req.rubric_id, identified=len(req.paper_ids), duplicates_removed=0)
    return {"survivors": survivors, "prisma": prisma}


@app.post("/api/extract")
async def api_extract(req: RunReq, request: Request):
    """Phase 3: Run grounded Extractor + Critic verification loop across surviving papers."""
    ws = _workspace(request)
    llm_extract = make_llm_json(EXTRACT_MODELS)
    llm_verify = make_llm_json(VERIFY_MODELS)

    # Hybrid keyword + semantic RRF retrieval helper (with cached query embedding)
    async def retrieve_fn(pid: str, query: str, k: int) -> list[dict]:
        qvec = await aembed_query(query)
        with get_conn() as c:
            return retrieve(c, pid, query, qvec, k=k)

    # Full-paper reading-order excluding figure placeholders and bibliography sections
    async def full_text_fn(pid: str) -> list[dict]:
        with get_conn() as c:
            cur = c.execute(
                """SELECT chunk_id, section, kind, text, page_start, bboxes, needs_vlm
                   FROM chunks
                   WHERE paper_id=%s
                     AND kind <> 'figure'
                     AND section !~* '(^|\\s|>)(references|bibliography|works cited|acknowledg(e)?ments)(\\s|$)'
                   ORDER BY ordinal""",
                (pid,),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    with get_conn() as conn:
        spec = conn.execute("SELECT spec FROM rubrics WHERE rubric_id=%s", (req.rubric_id,)).fetchone()[0]
        rubric = Rubric(**spec)
        target_cids = {c.criterion_id for c in rubric.criteria if c.kind in ("extraction", "quality")}
        summary = {}
        for pid in req.paper_ids:
            done_cids: set[str] = set()
            existing_rows: list[dict] = []
            if not req.force:
                cur = conn.execute(
                    """SELECT criterion_id, value, verbatim_quote, chunk_id::text, page_number,
                              bbox, confidence, critic_status, critic_notes, attempts
                       FROM current_cells
                       WHERE rubric_id=%s AND paper_id=%s
                         AND critic_status IN ('verified', 'approved', 'overridden', 'not_reported')""",
                    (req.rubric_id, pid),
                )
                for row in cur.fetchall():
                    done_cids.add(row[0])
                    existing_rows.append({
                        "criterion_id": row[0], "value": row[1], "quote": row[2],
                        "chunk_id": row[3], "page": row[4], "bbox": row[5],
                        "confidence": row[6], "status": row[7], "notes": row[8], "attempts": row[9],
                    })

            new_cells = []
            if not target_cids.issubset(done_cids):
                new_cells = await extract_paper(
                    conn,
                    req.rubric_id,
                    rubric,
                    pid,
                    retrieve_fn,
                    full_text_fn,
                    llm_extract,
                    llm_verify,
                    MODEL_EXTRACT,
                    concurrency=EXTRACT_CONCURRENCY,
                    skip_criterion_ids=done_cids,
                )
            summary[pid] = existing_rows + [c.__dict__ for c in new_cells]
    return {"extracted": summary}


@app.get("/api/matrix/{rubric_id}")
def api_matrix(rubric_id: str, request: Request):
    """Phase 4a: Fetch papers, criteria, and latest cells from the current_cells view."""
    ws = _workspace(request)
    with get_conn() as conn:
        papers = conn.execute(
            "SELECT paper_id::text, title, pdf_uri AS pdf_url FROM papers WHERE workspace_id=%s ORDER BY created_at",
            (ws,),
        ).fetchall()
        spec = conn.execute("SELECT spec FROM rubrics WHERE rubric_id=%s", (rubric_id,)).fetchone()[0]
        criteria = [c["criterion_id"] for c in spec["criteria"] if c["kind"] in ("extraction", "quality")]
        cur = conn.execute(
            """SELECT cell_id, paper_id::text, criterion_id, value, verbatim_quote, page_number,
                      bbox, confidence, critic_status, critic_notes
               FROM current_cells WHERE rubric_id=%s""",
            (rubric_id,),
        )
        cols = [d.name for d in cur.description]
        cells = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {
        "papers": [{"paper_id": p[0], "title": p[1] or p[0], "pdf_url": p[2]} for p in papers],
        "criteria": criteria,
        "cells": cells,
    }


@app.post("/api/cells/approve")
def api_approve(req: ApproveReq):
    """Phase 4b: Append 'approved' cell versions (preserving immutable history) and log audit events."""
    out = []
    with get_conn() as conn, conn.transaction():
        for cid in req.cell_ids:
            old = conn.execute(
                "SELECT rubric_id, paper_id, criterion_id, critic_status FROM extraction_cells WHERE cell_id=%s",
                (cid,),
            ).fetchone()
            cur = conn.execute(
                """INSERT INTO extraction_cells (rubric_id, paper_id, criterion_id, value, verbatim_quote, chunk_id,
                     page_number, bbox, confidence, critic_status, critic_notes, attempts, model_version)
                   SELECT rubric_id, paper_id, criterion_id, value, verbatim_quote, chunk_id,
                          page_number, bbox, confidence, 'approved', critic_notes, attempts, model_version
                   FROM extraction_cells WHERE cell_id=%s
                   RETURNING cell_id, paper_id::text, criterion_id, value, verbatim_quote, page_number,
                             bbox, confidence, critic_status, critic_notes""",
                (cid,),
            )
            cols = [d.name for d in cur.description]
            out.append(dict(zip(cols, cur.fetchone())))
            conn.execute(
                """INSERT INTO audit_log (actor, action, rubric_id, paper_id, criterion_id, before, after)
                   VALUES ('user:reviewer','approve',%s,%s,%s,%s,%s)""",
                (old[0], old[1], old[2], json.dumps({"status": old[3]}), json.dumps({"status": "approved"})),
            )
    return out


@app.post("/api/cells/override")
def api_override(req: OverrideReq):
    """Phase 4c: Append a human-overridden cell value with mandatory rationale and audit entry."""
    if not req.reason.strip():
        raise HTTPException(422, "reason required")
    with get_conn() as conn, conn.transaction():
        old = conn.execute(
            "SELECT rubric_id, paper_id, criterion_id, value, critic_status FROM extraction_cells WHERE cell_id=%s",
            (req.cell_id,),
        ).fetchone()
        cur = conn.execute(
            """INSERT INTO extraction_cells (rubric_id, paper_id, criterion_id, value, verbatim_quote, chunk_id,
                 page_number, bbox, confidence, critic_status, critic_notes, attempts, model_version)
               SELECT rubric_id, paper_id, criterion_id, %s::jsonb, verbatim_quote, chunk_id,
                      page_number, bbox, 1.0, 'overridden', %s, attempts, model_version
               FROM extraction_cells WHERE cell_id=%s
               RETURNING cell_id, paper_id::text, criterion_id, value, verbatim_quote, page_number,
                         bbox, confidence, critic_status, critic_notes""",
            (json.dumps(req.value), f"Override: {req.reason}", req.cell_id),
        )
        cols = [d.name for d in cur.description]
        row = dict(zip(cols, cur.fetchone()))
        conn.execute(
            """INSERT INTO audit_log (actor, action, rubric_id, paper_id, criterion_id, before, after)
               VALUES ('user:reviewer','override',%s,%s,%s,%s,%s)""",
            (
                old[0],
                old[1],
                old[2],
                json.dumps({"value": old[3], "status": old[4]}),
                json.dumps({"value": req.value, "reason": req.reason}),
            ),
        )
    return row


@app.post("/api/synthesize/{rubric_id}")
async def api_synthesize(rubric_id: str):
    """Phase 5a: Generate a citation-verified narrative synthesis across verified matrix cells."""
    with get_conn() as conn:
        df = load_matrix(conn, rubric_id)
    syn = await draft_synthesis(df, make_llm_json(VERIFY_MODELS))
    return syn.model_dump()


@app.get("/api/export/{rubric_id}/xlsx")
def api_export_xlsx(rubric_id: str):
    """Phase 5b: Export a multi-tab Excel workbook (Matrix, Provenance, PRISMA)."""
    with get_conn() as conn:
        df = load_matrix(conn, rubric_id, include_unverified=True)
        total = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
        prisma = prisma_counts(conn, rubric_id, identified=total, duplicates_removed=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    export_excel(df, prisma, tmp.name)
    return FileResponse(tmp.name, filename=f"slr_export_{rubric_id[:8]}.xlsx")