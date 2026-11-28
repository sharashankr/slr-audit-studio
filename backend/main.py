"""FastAPI backend entrypoint for SLR-Audit Studio (Free-Tier Gemini Configuration).

Wires the 5 pipeline modules (ingest, rubric_screening, grounded_extraction, export_eval)
to PostgreSQL + pgvector and Google AI Studio's free-tier Gemini 3.x models.
"""
from __future__ import annotations

import asyncio
import time
from uuid import UUID
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import httpx
import psycopg
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from annotated_pdf import render_annotated_pdf
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
LLM_PROVIDER = os.getenv("SLR_LLM_PROVIDER", "gemini").lower()
if LLM_PROVIDER not in {"gemini", "groq"}:
    raise ValueError("SLR_LLM_PROVIDER must be gemini or groq")
GROQ_API_KEY = os.environ["GROQ_API_KEY"] if LLM_PROVIDER == "groq" else None
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

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
if LLM_PROVIDER == "groq":
    EXTRACT_MODELS = VERIFY_MODELS = [GROQ_MODEL]

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

gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key) if LLM_PROVIDER == "gemini" and gemini_key else None
app = FastAPI(title="SLR-Audit Studio API (Free Tier)")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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
class GroqRateLimitError(RuntimeError):
    """Groq declined the call after the allowed short retries."""


async def groq_llm_json(system: str, user: str, schema: type[BaseModel]) -> dict:
    """Send a JSON-mode request and validate the response against the pipeline schema."""
    rpm = int(os.getenv("GROQ_RPM_LIMIT", "2"))
    temperature = 0.0
    async with httpx.AsyncClient(timeout=120) as http:
        for attempt in range(1, 4):
            await _wait_for_rate_slot(GROQ_MODEL, rpm=rpm)
            response = await http.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": f"{system}\nReturn one JSON object matching this schema: {json.dumps(schema.model_json_schema())}"},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_completion_tokens": 2048,
                    "response_format": {"type": "json_object"},
                },
            )
            if response.is_success:
                try:
                    choice = response.json()["choices"][0]
                    result = schema.model_validate_json(choice["message"]["content"])
                except (ValueError, KeyError, IndexError, TypeError):
                    failure = "response did not match the requested JSON schema"
                else:
                    return result.model_dump(mode="json")
            else:
                try:
                    code = response.json()["error"].get("code")
                except (ValueError, KeyError, AttributeError, TypeError):
                    code = None
                # Log machine-readable codes only, never the raw error payload.
                if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) or code.startswith("gsk_"):
                    code = "unspecified"
                failure = f"HTTP {response.status_code} ({code})"
                if response.status_code == 429:
                    quota_headers = (
                        "retry-after",
                        "x-ratelimit-remaining-requests",
                        "x-ratelimit-reset-requests",
                        "x-ratelimit-remaining-tokens",
                        "x-ratelimit-reset-tokens",
                    )
                    quota = ", ".join(
                        f"{name}={value}" for name in quota_headers
                        if (value := response.headers.get(name))
                        and re.fullmatch(r"[0-9.eE+dhms-]{1,32}", value)
                    )
                    if quota:
                        failure += f" [{quota}]"
                    try:
                        delay = float(response.headers.get("retry-after", "60"))
                    except ValueError:
                        delay = 60
                    if attempt < 3 and 0 <= delay <= 120:
                        print(f"[groq] {failure}; retrying in {delay + 1:.1f}s", flush=True)
                        await asyncio.sleep(delay + 1)
                        continue
                    raise GroqRateLimitError(f"Groq {GROQ_MODEL} {schema.__name__}: {failure}")
                if response.status_code != 400 or code not in {"json_validate_failed", "json_validation_failed"}:
                    raise RuntimeError(f"Groq {GROQ_MODEL} {schema.__name__}: {failure}")

            if attempt < 3:
                print(f"[groq] {schema.__name__}: {failure}; retrying JSON generation ({attempt + 1}/3)", flush=True)
                temperature = 0.2

    raise RuntimeError(f"Groq {GROQ_MODEL} {schema.__name__}: {failure} after 3 attempts")


_browser_llm: ContextVar[Any] = ContextVar("browser_llm", default=None)


def make_llm_json(models: str | list[str]):
    """Build an async LLMJson callable enforcing Pydantic JSON schema decoding
    with proactive RPM pacing and automatic multi-model failover on 429/503/404.
    """
    browser_llm = _browser_llm.get()
    if browser_llm is not None:
        return browser_llm
    if LLM_PROVIDER == "groq":
        return groq_llm_json
    if client is None:
        raise HTTPException(503, "Add a provider key in Models & API keys to run this operation.")

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
        clean_schema = schema.model_json_schema()
        _strip_additional_props(clean_schema)
        last_err: Exception | None = None

        for round_idx in range(4):
            available = [m for m in model_list if m not in _EXHAUSTED_MODELS]
            if not available:
                break

            for idx, active_model in enumerate(available):
                has_fallback = idx < len(available) - 1
                max_tries = 1 if has_fallback else 3
                for attempt in range(max_tries):
                    try:
                        for json_attempt in range(2):
                            await _wait_for_rate_slot(active_model)
                            instruction = system
                            if json_attempt:
                                instruction += (
                                    "\nReturn compact, valid JSON only. Keep source quotes within 300 "
                                    "characters and caveats or explanations to one sentence."
                                )
                            resp = await client.aio.models.generate_content(
                                model=active_model,
                                contents=user,
                                config=types.GenerateContentConfig(
                                    system_instruction=instruction,
                                    response_mime_type="application/json",
                                    response_schema=deepcopy(clean_schema),
                                    temperature=0.0,
                                ),
                            )
                            try:
                                return schema.model_validate_json(resp.text).model_dump(mode="json")
                            except ValueError:
                                if json_attempt:
                                    raise RuntimeError(
                                        f"Gemini {active_model} {schema.__name__}: invalid structured response after retry"
                                    ) from None
                                print(f"[llm] {active_model} {schema.__name__}: invalid structured response; retrying once", flush=True)
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
    dest = PDF_DIR / Path(file.filename).name
    # Stage the upload before replacing the destination in case the benchmark
    # opened the same file as its source.
    with tempfile.NamedTemporaryFile(dir=PDF_DIR, delete=False) as staged:
        staged_path = Path(staged.name)
        try:
            shutil.copyfileobj(file.file, staged)
            staged.close()
            staged_path.replace(dest)
        finally:
            staged_path.unlink(missing_ok=True)
    with get_conn() as conn:
        paper_id = ingest_pdf(
            conn,
            dest,
            f"/pdfs/{dest.name}",
            embed_fn=embed_texts if client is not None and request.headers.get("X-SLR-Retrieval") != "keywords" else None,
            workspace_id=ws,
        )
    return {"paper_id": paper_id, "pdf_url": f"/pdfs/{dest.name}"}


@app.post("/api/rubrics")
async def api_create_rubric(req: RubricReq, request: Request):
    """Phase 2a: Compile YAML/PICO criteria into typed rules and lint for ambiguity."""
    ws = _workspace(request)
    try:
        rubric = compile_rubric(req.yaml_text)
    except (yaml.YAMLError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(422, "Invalid rubric YAML. Check the name and criterion definitions.") from exc
    spec_json = rubric.model_dump_json()
    with get_conn() as conn:
        existing = conn.execute(
            """SELECT rubric_id FROM rubrics
               WHERE name=%s AND version=%s AND spec = %s::jsonb AND workspace_id=%s
               ORDER BY created_at DESC LIMIT 1""",
            (rubric.name, rubric.version, spec_json, ws),
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
    return await extract_run(req, MODEL_EXTRACT)


async def extract_run(
    req: RunReq, model_version: str, *, resume_since: datetime | None = None, examples: dict | None = None,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    browser_provider: str | None = None,
):
    """Phase 3: Run grounded Extractor + Critic verification loop across surviving papers."""
    llm_extract = make_llm_json(EXTRACT_MODELS)
    llm_verify = make_llm_json(VERIFY_MODELS)

    browser_direct = _browser_llm.get() is not None
    bounded_context = browser_provider != "Google Gemini" if browser_direct else LLM_PROVIDER == "groq"
    paper_chunks: dict[str, list[dict]] = {}

    # Hybrid keyword + semantic RRF retrieval helper (with cached query embedding)
    async def retrieve_fn(pid: str, query: str, k: int) -> list[dict]:
        if bounded_context or browser_direct:
            chunks = await full_text_fn(pid)
            terms = set(re.findall(r"\b\w{4,}\b", query.lower()))

            def relevance(chunk: dict) -> int:
                section = set(re.findall(r"\b\w{4,}\b", (chunk["section"] or "").lower()))
                text_terms = set(re.findall(r"\b\w{4,}\b", chunk["text"].lower()))
                return 2 * len(section & terms) + len(text_terms & terms)

            # Keep source text verbatim for quote grounding. Budget by UTF-8
            # bytes so both extract and critic requests have room for metadata.
            evidence = []
            remaining = 8_000
            for chunk in sorted(chunks, key=relevance, reverse=True)[:k]:
                if remaining < 500:
                    break
                excerpt = chunk["text"].encode()[:min(2_500, remaining)].decode(errors="ignore")
                evidence.append({**chunk, "text": excerpt})
                remaining -= len(excerpt.encode())
            return evidence

        qvec = await aembed_query(query)
        with get_conn() as c:
            return retrieve(c, pid, query, qvec, k=k)

    # Full-paper reading-order excluding figure placeholders and bibliography sections
    async def full_text_fn(pid: str) -> list[dict]:
        if pid in paper_chunks:
            return paper_chunks[pid]
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
            paper_chunks[pid] = [dict(zip(cols, r)) for r in cur.fetchall()]
        if not paper_chunks[pid]:
            raise HTTPException(422, "This PDF has no usable text or tables for extraction. Try a text-searchable copy.")
        return paper_chunks[pid]

    with get_conn() as conn:
        spec = conn.execute("SELECT spec FROM rubrics WHERE rubric_id=%s", (req.rubric_id,)).fetchone()[0]
        rubric = Rubric(**spec)
        target_cids = {c.criterion_id for c in rubric.criteria if c.kind in ("extraction", "quality")}
        summary = {}
        for pid in req.paper_ids:
            done_cids: set[str] = set()
            existing_rows: list[dict] = []
            if resume_since is not None:
                cur = conn.execute(
                    """SELECT DISTINCT ON (criterion_id)
                              criterion_id, value, verbatim_quote, chunk_id::text, page_number,
                              bbox, confidence, critic_status, critic_notes, attempts
                       FROM extraction_cells
                       WHERE rubric_id=%s AND paper_id=%s AND model_version=%s AND created_at >= %s
                         AND critic_status NOT IN ('approved', 'overridden')
                       ORDER BY criterion_id, cell_id DESC""",
                    (req.rubric_id, pid, model_version, resume_since),
                )
            elif not req.force:
                cur = conn.execute(
                    """SELECT criterion_id, value, verbatim_quote, chunk_id::text, page_number,
                              bbox, confidence, critic_status, critic_notes, attempts
                       FROM current_cells
                       WHERE rubric_id=%s AND paper_id=%s
                         AND critic_status IN ('verified', 'approved', 'overridden', 'not_reported')""",
                    (req.rubric_id, pid),
                )
            else:
                cur = None
            if cur is not None:
                for row in cur.fetchall():
                    done_cids.add(row[0])
                    existing_rows.append({
                        "criterion_id": row[0], "value": row[1], "quote": row[2],
                        "chunk_id": row[3], "page": row[4], "bbox": row[5],
                        "confidence": row[6], "status": row[7], "notes": row[8], "attempts": row[9],
                    })

            if resume_since is not None:
                print(f"[resume] paper={pid[:8]} restored {len(done_cids)}/{len(target_cids)} criteria", flush=True)
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
                    model_version,
                    concurrency=EXTRACT_CONCURRENCY,
                    skip_criterion_ids=done_cids,
                    allow_full_context=not bounded_context,
                    examples=examples,
                    on_progress=on_progress,
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
        rubric = conn.execute(
            "SELECT name, spec FROM rubrics WHERE rubric_id=%s AND workspace_id IN (%s, 'default')",
            (rubric_id, ws),
        ).fetchone()
        if rubric is None:
            raise HTTPException(404, "Rubric unavailable in this workspace")
        rubric_name, spec = rubric
        criteria = [c["criterion_id"] for c in spec["criteria"] if c["kind"] in ("extraction", "quality")]
        cur = conn.execute(
            """SELECT cell_id, paper_id::text, criterion_id, value, verbatim_quote, page_number,
                      bbox, confidence, critic_status, critic_notes
               FROM current_cells WHERE rubric_id=%s
                 AND paper_id IN (SELECT paper_id FROM papers WHERE workspace_id=%s)""",
            (rubric_id, ws),
        )
        cols = [d.name for d in cur.description]
        cells = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {
        "rubric_name": rubric_name,
        "papers": [
            {"paper_id": p[0], "title": p[1] or p[0], "pdf_url": p[2], "doi": p[3], "year": p[4]}
            for p in papers
        ],
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
    return await synthesize_run(rubric_id)


async def synthesize_run(rubric_id: str, workspace_id: str | None = None):
    """Phase 5a: Generate a citation-verified narrative synthesis across verified matrix cells."""
    with get_conn() as conn:
        df = load_matrix(conn, rubric_id, workspace_id=workspace_id)
    syn = await draft_synthesis(df, make_llm_json(VERIFY_MODELS))
    return syn.model_dump()


class BrowserOperation(BaseModel):
    action: str
    workspace: UUID
    model: str
    provider: Literal["Google Gemini", "Groq", "Mistral"] | None = None
    params: dict[str, Any]


@app.websocket("/api/browser-model")
async def browser_model(socket: WebSocket):
    origin = socket.headers.get("origin")
    if not origin or ("*" not in CORS_ORIGINS and origin not in CORS_ORIGINS):
        await socket.close(code=1008)
        return
    await socket.accept()
    lock = asyncio.Lock()

    async def browser_completion(system: str, user: str, schema: type[BaseModel]) -> dict:
        async with lock:
            for attempt in range(2):
                instruction = system
                if attempt:
                    instruction += "\nYour previous response failed validation. Return compact JSON matching the schema exactly."
                await socket.send_json({
                    "type": "completion", "system": instruction, "user": user,
                    "schema": schema.model_json_schema(),
                })
                try:
                    message = await asyncio.wait_for(socket.receive_json(), timeout=150)
                except asyncio.TimeoutError:
                    raise HTTPException(504, f"Timed out waiting for the model’s {schema.__name__} response.") from None
                if not isinstance(message, dict) or message.get("type") != "completion_result":
                    raise HTTPException(502, "The browser sent an unexpected model response message.")
                if message.get("error") == "invalid_json":
                    continue
                if message.get("error"):
                    raise HTTPException(502, "Direct provider request failed. See the browser message for details.")
                try:
                    return schema.model_validate(message.get("value")).model_dump(mode="json")
                except ValidationError:
                    if attempt:
                        break
            raise HTTPException(502, f"The model returned an invalid {schema.__name__} response after one retry.")

    token = _browser_llm.set(browser_completion)
    try:
        operation = BrowserOperation.model_validate(
            await asyncio.wait_for(socket.receive_json(), timeout=15)
        )
        request = Request({
            **socket.scope, "type": "http",
            "headers": [(b"x-workspace-id", str(operation.workspace).encode())],
        })
        params = operation.params
        if operation.action == "rubric":
            result = await api_create_rubric(RubricReq.model_validate(params), request)
        elif operation.action in {"screen", "extract"}:
            run = RunReq.model_validate(params)
            paper_ids = {UUID(paper_id) for paper_id in run.paper_ids}
            with get_conn() as conn:
                owned = conn.execute(
                    "SELECT count(*) FROM papers WHERE workspace_id=%s AND paper_id=ANY(%s)",
                    (str(operation.workspace), list(paper_ids)),
                ).fetchone()[0]
            if owned != len(paper_ids):
                raise HTTPException(404, "One or more papers are unavailable in this workspace.")
            if operation.action == "screen":
                result = await api_screen(run, request)
            else:
                run.force = True
                async def progress(completed: int, total: int):
                    async with lock:
                        await socket.send_json({"type": "progress", "completed": completed, "total": total})

                result = await extract_run(run, operation.model, on_progress=progress, browser_provider=operation.provider)
        elif operation.action == "synthesize":
            result = await synthesize_run(str(params["rubric_id"]), str(operation.workspace))
        else:
            raise HTTPException(400, "Unknown operation")
        await socket.send_json({"type": "result", "value": result})
        await socket.close()
    except WebSocketDisconnect:
        return
    except HTTPException as exc:
        await socket.send_json({"type": "error", "message": exc.detail})
        await socket.close()
    except (ValueError, KeyError, asyncio.TimeoutError):
        await socket.send_json({"type": "error", "message": "Invalid or timed-out browser model response."})
        await socket.close()
    finally:
        _browser_llm.reset(token)


@app.get("/api/export/{rubric_id}/xlsx")
def api_export_xlsx(rubric_id: str, request: Request):
    """Phase 5b: Export a multi-tab Excel workbook (Matrix, Provenance, PRISMA)."""
    ws = _workspace(request)
    with get_conn() as conn:
        df = load_matrix(conn, rubric_id, include_unverified=True, workspace_id=ws)
        total = conn.execute("SELECT count(*) FROM papers WHERE workspace_id=%s", (ws,)).fetchone()[0]
        prisma = prisma_counts(conn, rubric_id, identified=total, duplicates_removed=0, workspace_id=ws)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    export_excel(df, prisma, tmp.name)
    return FileResponse(tmp.name, filename=f"slr_export_{rubric_id[:8]}.xlsx")


@app.get("/api/export/{rubric_id}/papers/{paper_id}/annotated.pdf")
def api_export_annotated_pdf(rubric_id: UUID, paper_id: UUID, request: Request):
    ws = _workspace(request)
    with get_conn() as conn:
        source = conn.execute(
            """SELECT p.pdf_uri, p.sha256 FROM papers p
               WHERE p.paper_id=%s AND p.workspace_id=%s
               AND EXISTS (SELECT 1 FROM rubrics WHERE rubric_id=%s AND workspace_id IN (%s, 'default'))""",
            (paper_id, ws, rubric_id, ws),
        ).fetchone()
        if source is None:
            raise HTTPException(404, "Paper or rubric unavailable in this workspace")
        uri, digest = source
        cur = conn.execute(
            """SELECT criterion_id, value, verbatim_quote, bbox, critic_status, critic_notes
               FROM current_cells WHERE rubric_id=%s AND paper_id=%s
               AND bbox IS NOT NULL AND length(trim(verbatim_quote)) > 0
               ORDER BY criterion_id""",
            (rubric_id, paper_id),
        )
        columns = [column.name for column in cur.description]
        evidence = [dict(zip(columns, row)) for row in cur.fetchall()]

    filename = uri.removeprefix("/pdfs/")
    if not evidence:
        raise HTTPException(422, "This paper has no extracted quotes with PDF coordinates to annotate")
    if not uri.startswith("/pdfs/") or filename in {"", ".", ".."} or Path(filename).name != filename:
        raise HTTPException(404, "Original PDF unavailable")
    path = PDF_DIR / filename
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "Original PDF unavailable")
    pdf = path.read_bytes()
    if hashlib.sha256(pdf).hexdigest() != digest:
        raise HTTPException(409, "The stored PDF has changed since extraction. Restore the original PDF before exporting.")
    annotated, count = render_annotated_pdf(pdf, evidence)
    if not count:
        raise HTTPException(422, "The stored evidence has no valid PDF coordinates")
    return Response(
        annotated,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="slr-annotated-{str(paper_id)[:8]}.pdf"',
            "Cache-Control": "private, no-store",
        },
    )
