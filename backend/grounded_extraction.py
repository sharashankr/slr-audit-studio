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

BATCH_SIZE = 7
EXCERPT_BATCH_SIZE = 4
EXCERPT_CHARS = 1000
MAX_FULL_CONTEXT_CHARS = 65_000  # ~16k tokens; larger papers fall back to retrieved chunks


class BatchExtractionItem(BaseModel):
    criterion_id: str
    reported: bool
    value: Any | None = None
    unit: str | None = None
    verbatim_quote: str | None = None
    chunk_ref: str | None = None       # e.g. "C0", "C1"
    confidence: float = 0.0
    caveats: str | None = None


class BatchExtractionOut(BaseModel):
    items: list[BatchExtractionItem]


class BatchCritiqueItem(BaseModel):
    criterion_id: str
    verdict: Literal["supported", "unsupported", "unit_mismatch", "missed_caveat", "wrong_value"]
    explanation: str
    suggested_fix: str | None = None


class BatchCritiqueOut(BaseModel):
    critiques: list[BatchCritiqueItem]


BATCH_EXTRACT_SYS = (
    "You extract multiple data points from one scientific paper. Rules:\n"
    "1. Return one entry in `items` for EVERY criterion_id provided.\n"
    "2. Use ONLY the supplied chunks. verbatim_quote must be copied character-for-character from one chunk, <= 300 chars, and support the value; set chunk_ref to that chunk's tag (e.g. 'C3').\n"
    "3. If the rubric defines an enum level for 'not mentioned', 'no', or 'NA' and the concept is absent, set reported=true, pick that enum value, and leave verbatim_quote=null.\n"
    "4. For non-enum fields not reported anywhere in the supplied paper, set reported=false; never infer or compute.\n"
    "5. Respect every Calibration note in scoring_rules carefully. Note caveats (subgroup only, post-hoc, per-arm vs total)."
)

BATCH_VERIFY_SYS = (
    "You audit extracted data points against source passages from one paper. "
    "For EACH item return verdict=supported only if: (a) the quote or absence claim across the paper supports the value, "
    "(b) the exact enum level matches the scoring_rules and calibration notes, (c) units/population/timepoint match, and (d) no caveat is missed. "
    "Be skeptical and return one entry in `critiques` for EVERY criterion_id."
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

def _fmt_indexed(chunks: list[dict]) -> tuple[str, dict[str, dict]]:
    ref_map: dict[str, dict] = {}
    parts: list[str] = []
    for idx, c in enumerate(chunks):
        ref = f"C{idx}"
        ref_map[ref] = c
        parts.append(f"[{ref}] section={c['section']} page={c['page_start']}\n{c['text']}")
    return "\n\n".join(parts), ref_map


def _with_examples(payload: dict, criteria: list[Criterion], examples: dict | None) -> dict:
    if not examples:
        return payload
    relevant = {c.criterion_id: examples[c.criterion_id] for c in criteria if c.criterion_id in examples}
    if relevant:
        payload["examples_from_other_papers"] = relevant
        payload["example_instructions"] = (
            "Use the reference scores and any reviewer notes only to calibrate how you apply the rubric. "
            "Paper numbers identify examples, not the target paper. Ground answers and quotes only "
            "in the target paper chunks below. Do not copy scores, quotes, or facts from examples."
        )
    return payload


async def extract_cell(
    c: Criterion,
    paper_id: str,
    retrieve_fn: Callable[[str, str, int], Awaitable[list[dict]]],   # (paper_id, query, k)
    full_text_fn: Callable[[str], Awaitable[list[dict]]],            # all chunks, reading order
    llm_extract: LLMJson,
    llm_verify: LLMJson,
    initial_feedback: str = "",
    initial_chunks: list[dict] | None = None,
    allow_full_context: bool = True,
    examples: dict | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> CellResult:
    if initial_chunks is not None:
        chunks, used_full = initial_chunks, allow_full_context
    else:
        query = f"{c.question} {' '.join(c.section_hints)}"
        chunks, used_full = await retrieve_fn(paper_id, query, 6), False
    feedback = initial_feedback
    last: CellResult | None = None


    for attempt in range(1, max_attempts + 2):                       # last pass = full-context fallback
        if attempt > max_attempts and not used_full and allow_full_context:
            chunks, used_full, feedback = await full_text_fn(paper_id), True, feedback + "\nSearch the full paper."
        elif attempt > max_attempts:
            break
        payload = {"criterion": c.model_dump(exclude={"stage"}), "retry_feedback": feedback or None}
        user = json.dumps(_with_examples(payload, [c], examples)) + "\n\nCHUNKS:\n" + _fmt(chunks)
        ex = Extraction(**await llm_extract(EXTRACT_SYS, user, Extraction))

        if not ex.reported:
            # Absence claims are riskier than presence claims: only accept after a full-context pass.
            if used_full:
                return CellResult(c.criterion_id, None, None, None, None, None, ex.confidence, "not_reported",
                                  ex.caveats or "Not reported in full text", attempt)
            if not allow_full_context:
                return CellResult(c.criterion_id, None, None, None, None, None, ex.confidence,
                                  "needs_review", "No evidence in retrieved excerpts; full paper not checked", attempt)
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
                if not allow_full_context:
                    return CellResult(c.criterion_id, None, None, None, None, None, ex.confidence,
                                      "needs_review", "Absence cannot be confirmed from retrieved excerpts", attempt)
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

    return last or CellResult(c.criterion_id, None, None, None, None, None, 0.0, "needs_review", "No grounded extraction after retries", max_attempts)


async def _extract_batch(
    batch: list[Criterion],
    chunks: list[dict],
    paper_text: str,
    ref_map: dict[str, dict],
    used_full: bool,
    llm_extract: LLMJson,
    examples: dict | None = None,
) -> tuple[dict[str, tuple[BatchExtractionItem, dict | None, bool]], list[tuple[Criterion, str]]]:
    """Run one batched extraction + deterministic grounding; return (passed_candidates, retry_entries)."""
    payload = [c.model_dump(exclude={"stage"}) for c in batch]
    user = json.dumps(_with_examples({"criteria": payload}, batch, examples)) + "\n\nCHUNKS:\n" + paper_text
    out = BatchExtractionOut(**await llm_extract(BATCH_EXTRACT_SYS, user, BatchExtractionOut))
    by_id = {item.criterion_id: item for item in out.items}

    candidates: dict[str, tuple[BatchExtractionItem, dict | None, bool]] = {}
    retry: list[tuple[Criterion, str]] = []

    for c in batch:
        item = by_id.get(c.criterion_id)
        if item is None:
            retry.append((c, ""))
            continue
        if not item.reported:
            if used_full:
                candidates[c.criterion_id] = (item, None, False)
            else:
                retry.append((c, "You reported 'not reported' from limited chunks; re-check carefully."))
            continue
        if c.data_type == "enum" and str(item.value) not in c.enum_values:
            retry.append((c, f"REJECTED: value must be one of {c.enum_values}."))
            continue

        unquoted_absence = _is_absence_or_na_enum(c, item.value) and not (item.verbatim_quote or "").strip()
        if unquoted_absence:
            if used_full:
                candidates[c.criterion_id] = (item, None, True)
            else:
                retry.append((c, f"You selected '{item.value}' without a quote; check the next relevant excerpts."))
            continue

        ref_chunk = ref_map.get(item.chunk_ref or "")
        search_pool = [ref_chunk] + chunks if ref_chunk else chunks
        src = ground_quote(item.verbatim_quote or "", search_pool)
        if src is None:
            retry.append((c, "REJECTED: verbatim_quote not found in any chunk. Copy text exactly."))
            continue
        if not value_in_quote(item.value, item.verbatim_quote or "", c.data_type):
            retry.append((c, "REJECTED: numeric value does not appear in the quote."))
            continue

        candidates[c.criterion_id] = (item, src, False)

    return candidates, retry


async def _verify_batch(
    crit_by_id: dict[str, Criterion],
    candidates: dict[str, tuple[BatchExtractionItem, dict | None, bool]],
    full_paper_text: str,
    llm_verify: LLMJson,
) -> tuple[list[CellResult], list[tuple[Criterion, str]]]:
    """Audit all grounded candidates in 1 batched Critic call; return (verified_results, rejected_for_retry)."""
    results: list[CellResult] = []
    to_verify: list[dict] = []
    cited_chunks: dict[str, dict] = {}
    has_unquoted_absence = False

    for cid, (item, src, unquoted_absence) in candidates.items():
        if not item.reported:
            results.append(
                CellResult(cid, None, None, None, None, None, item.confidence, "not_reported",
                           item.caveats or "Not reported in full text", 1)
            )
            continue

        c = crit_by_id[cid]
        if unquoted_absence:
            has_unquoted_absence = True
        elif src is not None:
            cited_chunks[str(src["chunk_id"])] = src

        to_verify.append({
            "criterion_id": cid,
            "question": c.question,
            "scoring_rules": c.scoring_rules,
            "extraction": item.model_dump(exclude={"chunk_ref"}),
            "unquoted_absence_claim": unquoted_absence,
        })

    if not to_verify:
        return results, []

    if has_unquoted_absence:
        source_block = full_paper_text
    else:
        source_block, _ = _fmt_indexed(list(cited_chunks.values()))

    vuser = json.dumps({"items": to_verify}) + "\n\nSOURCE:\n" + source_block
    crit_out = BatchCritiqueOut(**await llm_verify(BATCH_VERIFY_SYS, vuser, BatchCritiqueOut))
    verdicts = {cr.criterion_id: cr for cr in crit_out.critiques}

    rejected: list[tuple[Criterion, str]] = []
    for entry in to_verify:
        cid = entry["criterion_id"]
        c = crit_by_id[cid]
        item, src, unquoted_absence = candidates[cid]
        cr = verdicts.get(cid)
        if cr and cr.verdict == "supported":
            chunk_id_str = str(src["chunk_id"]) if (src and src["chunk_id"]) else None
            page = src["page_start"] if src else None
            bboxes = src["bboxes"] if src else None
            low = item.confidence < CONF_REVIEW_THRESHOLD or (src.get("needs_vlm", False) if src else False) or (src["kind"] == "table" if src else False)
            default_note = "Verified absent across full text" if unquoted_absence else ("table value - spot check" if (src and src["kind"] == "table") else "")
            results.append(
                CellResult(
                    cid, item.value, item.verbatim_quote, chunk_id_str, page, bboxes,
                    item.confidence, "needs_review" if low else "verified",
                    item.caveats or default_note, 1,
                )
            )
        else:
            fb = f"CRITIC: {cr.verdict}: {cr.explanation} Fix: {cr.suggested_fix or 'n/a'}" if cr else "Critic omitted verdict"
            rejected.append((c, fb))

    return results, rejected

def _save_cells(conn, rubric_id: str, paper_id: str, cells: list[CellResult], model_version: str) -> None:
    """Persist completed cells immediately so interrupted papers resume without repeating work."""
    if not cells:
        return
    for r in cells:
        conn.execute(
            """INSERT INTO extraction_cells (rubric_id, paper_id, criterion_id, value, verbatim_quote, chunk_id, page_number,
                 bbox, confidence, critic_status, critic_notes, attempts, model_version)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (rubric_id, paper_id, r.criterion_id, json.dumps(r.value), r.quote, r.chunk_id, r.page,
             json.dumps(r.bbox), r.confidence, r.status, r.notes, r.attempts, model_version),
        )
        conn.execute(
            "INSERT INTO audit_log (actor, action, rubric_id, paper_id, criterion_id, after, model_version) "
            "VALUES ('agent:extractor','extract',%s,%s,%s,%s,%s)",
            (rubric_id, paper_id, r.criterion_id, json.dumps({"value": r.value, "status": r.status}), model_version),
        )
    conn.commit()

async def extract_paper(
    conn,
    rubric_id: str,
    rubric: Rubric,
    paper_id: str,
    retrieve_fn,
    full_text_fn,
    llm_extract,
    llm_verify,
    model_version: str,
    concurrency: int = 6,
    skip_criterion_ids: set[str] | None = None,
    allow_full_context: bool = True,
    examples: dict | None = None,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    excerpt_batches: bool = False,
) -> list[CellResult]:
    skip = skip_criterion_ids or set()
    todo = [c for c in rubric.criteria if c.kind in ("extraction", "quality") and c.criterion_id not in skip]
    if not todo:
        return []

    crit_by_id = {c.criterion_id: c for c in todo}
    full_chunks = await full_text_fn(paper_id)
    full_text, ref_map = _fmt_indexed(full_chunks) if allow_full_context else ("", {})
    use_full_batch = allow_full_context and bool(full_chunks) and len(full_text) <= MAX_FULL_CONTEXT_CHARS
    print(f"[extract] paper={paper_id[:8]} todo={len(todo)} chunks={len(full_chunks)} chars={len(full_text)} full_batch={use_full_batch}", flush=True)

    results: list[CellResult] = []
    candidates: dict[str, tuple[BatchExtractionItem, dict | None, bool]] = {}
    retry_entries: list[tuple[Criterion, str]] = []
    is_rct = False
    followup_chunks: dict[str, list[dict]] = {}

    if use_full_batch:
        # 1. Extract non-OBS batch containing STUDY_DESIGN first so RCT papers skip OBS_*
        non_obs = [c for c in todo if not c.criterion_id.startswith("OBS_")]
        obs = [c for c in todo if c.criterion_id.startswith("OBS_")]

        first_batch = non_obs[:BATCH_SIZE]
        rest_non_obs = non_obs[BATCH_SIZE:]

        if first_batch:
            cands, ret = await _extract_batch(first_batch, full_chunks, full_text, ref_map, True, llm_extract, examples)
            candidates.update(cands)
            retry_entries.extend(ret)

        design_cand = candidates.get("STUDY_DESIGN")
        if design_cand is not None:
            is_rct = str(design_cand[0].value) == "2"
        elif any(c.criterion_id == "STUDY_DESIGN" for c, _ in retry_entries):
            design_crit = crit_by_id["STUDY_DESIGN"]
            retry_entries = [(c, fb) for c, fb in retry_entries if c.criterion_id != "STUDY_DESIGN"]
            design_res = await extract_cell(
                design_crit, paper_id, retrieve_fn, full_text_fn, llm_extract, llm_verify,
                initial_chunks=full_chunks, examples=examples,
            )
            results.append(design_res)
            is_rct = str(design_res.value) == "2"

        if is_rct:
            for c in obs:
                results.append(CellResult(c.criterion_id, "NA", None, None, None, None, 1.0, "verified", "Skipped: STUDY_DESIGN=2 (RCT)", 0))
            remaining = rest_non_obs
        else:
            remaining = obs + rest_non_obs

        # 2. Run remaining groups concurrently (bounded by semaphore)
        sem = asyncio.Semaphore(concurrency)

        async def run_grp(grp: list[Criterion]):
            async with sem:
                return await _extract_batch(grp, full_chunks, full_text, ref_map, True, llm_extract, examples)

        groups = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
        for cands, ret in await asyncio.gather(*(run_grp(g) for g in groups)):
            candidates.update(cands)
            retry_entries.extend(ret)

        # 3. Single batched Critic call for all grounded candidates
        print(f"[verify] paper={paper_id[:8]} auditing {len(candidates)} grounded items, batch_retries={len(retry_entries)}", flush=True)
        verified_res, critic_rejected = await _verify_batch(crit_by_id, candidates, full_text, llm_verify)
        results.extend(verified_res)
        _save_cells(conn, rubric_id, paper_id, verified_res, model_version)
        if on_progress:
            await on_progress(len(results), len(todo))
        retry_entries.extend(critic_rejected)
    elif excerpt_batches:
        design = crit_by_id.get("STUDY_DESIGN")
        ordered = ([design] if design else []) + [c for c in todo if c is not design]
        for start in range(0, len(ordered), EXCERPT_BATCH_SIZE):
            group = ordered[start : start + EXCERPT_BATCH_SIZE]
            finished = [
                CellResult(c.criterion_id, "NA", None, None, None, None, 1.0, "verified", "Skipped: STUDY_DESIGN=2 (RCT)", 0)
                for c in group if is_rct and c.criterion_id.startswith("OBS_")
            ]
            skipped_ids = {cell.criterion_id for cell in finished}
            group = [c for c in group if c.criterion_id not in skipped_ids]
            if group:
                selected = {}
                ranked = {}
                for c in group:
                    query = f"{c.question} {' '.join(c.section_hints)}"
                    ranked[c.criterion_id] = await retrieve_fn(paper_id, query, 6)
                    for chunk in ranked[c.criterion_id][:2]:
                        selected[chunk["chunk_id"]] = {**chunk, "text": chunk["text"][:EXCERPT_CHARS]}
                evidence = list(selected.values())
                text, refs = _fmt_indexed(evidence)
                batch_candidates, ungrounded = await _extract_batch(
                    group, evidence, text, refs, False, llm_extract, examples,
                )
                verified, rejected = await _verify_batch(crit_by_id, batch_candidates, "", llm_verify)
                finished.extend(verified)
                for c, feedback in ungrounded + rejected:
                    # Coverage expands locally by moving to unread chunks. If the
                    # critic rejected a grounded answer, keep its source in view.
                    prior = ranked[c.criterion_id]
                    unread = [chunk for chunk in prior if chunk["chunk_id"] not in selected]
                    candidate = batch_candidates.get(c.criterion_id)
                    source = candidate[1] if candidate else None
                    retry_sources = ([source] if source else []) + unread
                    followup_chunks[c.criterion_id] = [
                        {**chunk, "text": chunk["text"][:EXCERPT_CHARS]} for chunk in (retry_sources or prior)[:3]
                    ]
                    retry_entries.append((c, feedback))
                is_rct = is_rct or any(
                    cell.criterion_id == "STUDY_DESIGN" and str(cell.value) == "2" and cell.status == "verified"
                    for cell in verified
                )
            _save_cells(conn, rubric_id, paper_id, finished, model_version)
            results.extend(finished)
            if on_progress:
                await on_progress(len(results), len(todo))
    else:
        retry_entries = [(c, "") for c in todo]

    # Run targeted extraction for criteria that need an individual pass.
    if retry_entries:
        total = len(retry_entries)
        stage = "retry" if use_full_batch else "extract"
        print(f"[{stage}] paper={paper_id[:8]} queued {total} individual criteria", flush=True)
        completed = 0
        sem = asyncio.Semaphore(concurrency)
        retry_chunks = full_chunks if use_full_batch else None

        async def one(c: Criterion, fb: str):
            nonlocal completed
            async with sem:
                res = await extract_cell(
                    c, paper_id, retrieve_fn, full_text_fn, llm_extract, llm_verify,
                    initial_feedback=fb, initial_chunks=followup_chunks.get(c.criterion_id, retry_chunks),
                    allow_full_context=allow_full_context, examples=examples,
                    max_attempts=1 if excerpt_batches else MAX_ATTEMPTS,
                )
                _save_cells(conn, rubric_id, paper_id, [res], model_version)
                completed += 1
                if on_progress:
                    await on_progress(len(results) + completed, len(todo))
                print(f"[{stage}] paper={paper_id[:8]} saved {completed}/{total} individual criteria", flush=True)
                return res

        results.extend(await asyncio.gather(*(one(c, fb) for c, fb in retry_entries)))

    return results