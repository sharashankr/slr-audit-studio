"""Phase 3 - Extractor + Verifier with deterministic citation grounding and targeted retries.

Flow per (paper, criterion):
  retrieve -> extract (LLM) -> GROUND (deterministic substring check) -> verify (LLM critic)
    -> verified | retry with critic feedback (<= MAX_ATTEMPTS) -> full-context fallback -> needs_review
Papers/criteria run concurrently; extractor and verifier are separate prompts (ideally separate models).
"""
from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel

from rubric_screening import Criterion, LLMJson, Rubric

MAX_ATTEMPTS = 3
CONF_REVIEW_THRESHOLD = 0.75


# ---------------- Deterministic grounding ----------------
_LIGATURES = str.maketrans({"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl",
                            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                            "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " "})


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_LIGATURES)
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)      # de-hyphenate line breaks
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def ground_quote(quote: str, chunks: list[dict]) -> dict | None:
    """Return the chunk containing the quote verbatim (after whitespace/ligature normalisation), else None.
    No fuzzy matching: a quote either exists in the source or the extraction is rejected."""
    q = normalize(quote)
    if len(q) < 8:
        return None
    for c in chunks:
        if q in normalize(c["text"]):
            return c
    return None


def value_in_quote(value: Any, quote: str, data_type: str) -> bool:
    """Cheap numeric sanity check: extracted numbers must appear in the quote."""
    if data_type in ("int", "float") and value is not None:
        nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", quote)}
        v = str(value).rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
        return any(v == n.rstrip("0").rstrip(".") if "." in n else v == n for n in nums)
    return True


def _is_absence_or_na_enum(c: Criterion, value: Any) -> bool:
    """Check whether the chosen enum level explicitly encodes 'not mentioned', 'no', or 'NA'."""
    if c.data_type != "enum" or value is None:
        return False
    val = str(value).strip().lower()
    rules = c.scoring_rules.lower()
    if val == "na" and "na =" in rules:
        return True
    if val in ("0", "1"):
        absence_markers = (
            f"{val} = not mentioned",
            f"{val} = no ",
            f"{val} = no,",
            f"{val} = not defined",
            f"{val} = not done",
            f"{val} = not described",
            f"{val} = not discussed",
            f"{val} = no code",
            f"{val} = no analysis",
            f"{val} = no balance",
            f"{val} = no uncertainty",
        )
        return any(m in rules for m in absence_markers)
    return False


# ---------------- LLM output schemas ----------------
class Extraction(BaseModel):
    reported: bool                      # False -> paper does not report this (never invent)
    value: Any | None = None
    unit: str | None = None
    verbatim_quote: str | None = None
    confidence: float = 0.0
    caveats: str | None = None


class Critique(BaseModel):
    verdict: Literal["supported", "unsupported", "unit_mismatch", "missed_caveat", "wrong_value"]
    explanation: str
    suggested_fix: str | None = None


EXTRACT_SYS = (
    "You extract one data point from a scientific paper. Rules: (1) Use ONLY the supplied chunks. "
    "(2) verbatim_quote must be copied character-for-character from one chunk, <= 300 chars, and must support the value. "
    "(3) If the rubric defines an enum level for 'not mentioned', 'no', or 'NA' and the concept is absent, set reported=true, "
    "pick that enum value, and leave verbatim_quote=null (unless explicitly stated in text). "
    "(4) For non-enum fields not reported in the paper, set reported=false; never infer or compute. "
    "(5) Keep units as reported and put them in `unit`. (6) Note caveats (subgroup only, post-hoc, per-arm vs total)."
)
VERIFY_SYS = (
    "You audit an extraction against the source. Check: does the quote (or full-text absence check) support the value; "
    "is the unit right; is it the right population/arm/timepoint; is a caveat missed? Be skeptical. "
    "Return verdict=supported only if all hold."
)


@dataclass
class CellResult:
    criterion_id: str
    value: Any | None
    quote: str | None
    chunk_id: str | None
    page: int | None
    bbox: Any | None
    confidence: float
    status: str
    notes: str
    attempts: int


def _fmt(chunks: list[dict]) -> str:
    return "\n\n".join(f"[chunk {c['chunk_id']}] section={c['section']} page={c['page_start']}\n{c['text']}" for c in chunks)


async def extract_cell(
    c: Criterion,
    paper_id: str,
    retrieve_fn: Callable[[str, str, int], Awaitable[list[dict]]],   # (paper_id, query, k)
    full_text_fn: Callable[[str], Awaitable[list[dict]]],            # all chunks, reading order
    llm_extract: LLMJson,
    llm_verify: LLMJson,
) -> CellResult:
    query = f"{c.question} {' '.join(c.section_hints)}"
    chunks = await retrieve_fn(paper_id, query, 6)
    feedback = ""
    used_full = False
    last: CellResult | None = None

    for attempt in range(1, MAX_ATTEMPTS + 2):                       # last pass = full-context fallback
        if attempt > MAX_ATTEMPTS and not used_full:
            chunks, used_full, feedback = await full_text_fn(paper_id), True, feedback + "\nSearch the full paper."
        elif attempt > MAX_ATTEMPTS:
            break
        user = json.dumps({"criterion": c.model_dump(exclude={"stage"}), "retry_feedback": feedback or None}) + "\n\nCHUNKS:\n" + _fmt(chunks)
        ex = Extraction(**await llm_extract(EXTRACT_SYS, user, Extraction))

        if not ex.reported:
            # Absence claims are riskier than presence claims: only accept after a full-context pass.
            if used_full:
                return CellResult(c.criterion_id, None, None, None, None, None, ex.confidence, "not_reported",
                                  ex.caveats or "Not reported in full text", attempt)
            feedback = "You reported 'not reported' from limited chunks; re-check carefully."
            chunks, used_full = await full_text_fn(paper_id), True
            continue

        if c.data_type == "enum" and str(ex.value) not in c.enum_values:
            feedback = f"REJECTED: value must be one of {c.enum_values}."
            continue

        unquoted_absence = _is_absence_or_na_enum(c, ex.value) and not (ex.verbatim_quote or "").strip()

        if unquoted_absence:
            # Require full-text scan before confirming an unquoted 'not mentioned' / 'NA' enum score
            if not used_full:
                feedback = f"You selected '{ex.value}' (not mentioned / NA) without a quote from limited chunks; verify across the full paper."
                chunks, used_full = await full_text_fn(paper_id), True
                continue
            src = {
                "chunk_id": None,
                "page_start": None,
                "bboxes": None,
                "text": _fmt(chunks),
                "kind": "text",
                "needs_vlm": False,
            }
        else:
            src = ground_quote(ex.verbatim_quote or "", chunks)
            if src is None:
                feedback = "REJECTED: verbatim_quote not found in any chunk. Copy text exactly."
                last = CellResult(c.criterion_id, ex.value, ex.verbatim_quote, None, None, None, 0.0, "rejected", feedback, attempt)
                continue
            if not value_in_quote(ex.value, ex.verbatim_quote or "", c.data_type):
                feedback = "REJECTED: numeric value does not appear in the quote."
                last = CellResult(
                    c.criterion_id, ex.value, ex.verbatim_quote,
                    str(src["chunk_id"]) if src["chunk_id"] else None,
                    src["page_start"], src["bboxes"], 0.0, "rejected", feedback, attempt
                )
                continue

        vuser = json.dumps({"criterion": c.question, "scoring_rules": c.scoring_rules, "extraction": ex.model_dump()}) + "\n\nSOURCE:\n" + src["text"]
        crit = Critique(**await llm_verify(VERIFY_SYS, vuser, Critique))
        chunk_id_str = str(src["chunk_id"]) if src["chunk_id"] else None
        if crit.verdict == "supported":
            low = ex.confidence < CONF_REVIEW_THRESHOLD or src.get("needs_vlm", False) or src["kind"] == "table"
            default_note = "Verified absent across full text" if unquoted_absence else ("table value - spot check" if src["kind"] == "table" else "")
            return CellResult(
                c.criterion_id, ex.value, ex.verbatim_quote, chunk_id_str, src["page_start"], src["bboxes"],
                ex.confidence, "needs_review" if low else "verified",
                ex.caveats or default_note, attempt
            )
        feedback = f"CRITIC: {crit.verdict}: {crit.explanation} Fix: {crit.suggested_fix or 'n/a'}"
        last = CellResult(
            c.criterion_id, ex.value, ex.verbatim_quote, chunk_id_str, src["page_start"], src["bboxes"],
            ex.confidence, "needs_review", feedback, attempt
        )

    return last or CellResult(c.criterion_id, None, None, None, None, None, 0.0, "needs_review", "No grounded extraction after retries", MAX_ATTEMPTS)


async def extract_paper(conn, rubric_id: str, rubric: Rubric, paper_id: str, retrieve_fn, full_text_fn,
                        llm_extract, llm_verify, model_version: str, concurrency: int = 6) -> list[CellResult]:
    sem = asyncio.Semaphore(concurrency)
    todo = [c for c in rubric.criteria if c.kind in ("extraction", "quality")]

    async def one(c: Criterion):
        async with sem:
            return await extract_cell(c, paper_id, retrieve_fn, full_text_fn, llm_extract, llm_verify)

    # 1. Extract STUDY_DESIGN first so RCT papers (value == "2") skip OBS_* criteria
    design_crit = next((c for c in todo if c.criterion_id == "STUDY_DESIGN"), None)
    results: list[CellResult] = []
    is_rct = False

    if design_crit:
        design_res = await one(design_crit)
        results.append(design_res)
        is_rct = str(design_res.value) == "2"
        todo = [c for c in todo if c.criterion_id != "STUDY_DESIGN"]

    # 2. Deterministically assign NA to observational-only criteria when STUDY_DESIGN=2 (RCT)
    remaining: list[Criterion] = []
    for c in todo:
        if is_rct and c.criterion_id.startswith("OBS_"):
            results.append(
                CellResult(
                    c.criterion_id, "NA", None, None, None, None, 1.0, "verified",
                    "Skipped: STUDY_DESIGN=2 (RCT)", 0
                )
            )
        else:
            remaining.append(c)

    # 3. Extract remaining criteria concurrently
    results.extend(await asyncio.gather(*(one(c) for c in remaining)))

    for r in results:
        conn.execute(
            """INSERT INTO extraction_cells (rubric_id, paper_id, criterion_id, value, verbatim_quote, chunk_id, page_number,
                 bbox, confidence, critic_status, critic_notes, attempts, model_version)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (rubric_id, paper_id, r.criterion_id, json.dumps(r.value), r.quote, r.chunk_id, r.page,
             json.dumps(r.bbox), r.confidence, r.status, r.notes, r.attempts, model_version))
        conn.execute("INSERT INTO audit_log (actor, action, rubric_id, paper_id, criterion_id, after, model_version) VALUES ('agent:extractor','extract',%s,%s,%s,%s,%s)",
                     (rubric_id, paper_id, r.criterion_id, json.dumps({"value": r.value, "status": r.status}), model_version))
    conn.commit()
    return results