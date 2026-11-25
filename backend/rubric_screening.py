"""Phase 2 - Rubric compiler, two-stage screener, PRISMA counters, rubric dry-run.

pip install pydantic pyyaml
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# llm_json(system, user, schema) -> dict validated against `schema` (use provider structured outputs)
LLMJson = Callable[[str, str, type[BaseModel]], Awaitable[dict]]


# ---------------- Rubric model ----------------
class Criterion(BaseModel):
    criterion_id: str
    kind: Literal["inclusion", "exclusion", "quality", "extraction"]
    question: str
    data_type: Literal["bool", "enum", "float", "int", "string"] = "string"
    enum_values: list[str] = []
    unit: str | None = None
    scoring_rules: str = ""                    # plain-language pass/fail or scoring guidance
    is_gatekeeper: bool = False                # failing a gatekeeper excludes the paper
    stage: Literal["abstract", "methods", "extraction"] = "extraction"
    section_hints: list[str] = []              # e.g. ["Methods", "Participants"]

    @model_validator(mode="after")
    def _check(self):
        if self.data_type == "enum" and not self.enum_values:
            raise ValueError(f"{self.criterion_id}: enum needs enum_values")
        if self.is_gatekeeper and self.kind not in ("inclusion", "exclusion"):
            raise ValueError(f"{self.criterion_id}: only inclusion/exclusion can be gatekeepers")
        return self


class Rubric(BaseModel):
    name: str
    version: int = 1
    criteria: list[Criterion]

    @model_validator(mode="after")
    def _unique(self):
        ids = [c.criterion_id for c in self.criteria]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate criterion_id: {sorted(dupes)}")
        return self


EXAMPLE_YAML = """
name: demo-slr
version: 1
pico:
  population: "adults with type 2 diabetes"
  intervention: "SGLT2 inhibitors"
  comparison: "placebo or standard care"
  outcome: "HbA1c change"
criteria:
  - {criterion_id: INC_RCT, kind: inclusion, question: "Is the study a randomized controlled trial?", data_type: bool, is_gatekeeper: true, stage: abstract}
  - {criterion_id: EXC_PEDS, kind: exclusion, question: "Are participants under 18?", data_type: bool, is_gatekeeper: true, stage: methods, section_hints: [Methods, Participants]}
  - {criterion_id: N_TOTAL, kind: extraction, question: "Total randomized sample size", data_type: int, section_hints: [Methods, Results]}
"""


def compile_rubric(raw: str | dict) -> Rubric:
    """Deterministic compile: PICO shorthand -> atomic criteria; user criteria validated as-is.
    (An LLM 'rubric linter' can be layered on top to flag vague questions - see lint_rubric.)"""
    spec = yaml.safe_load(raw) if isinstance(raw, str) else dict(raw)
    criteria = list(spec.get("criteria", []))
    pico = spec.get("pico") or {}
    existing = {c["criterion_id"] for c in criteria}
    for key, label in (("population", "P"), ("intervention", "I"), ("comparison", "C"), ("outcome", "O")):
        cid = f"PICO_{label}"
        if pico.get(key) and cid not in existing:
            criteria.append({
                "criterion_id": cid, "kind": "inclusion", "data_type": "bool", "stage": "abstract",
                "is_gatekeeper": key in ("population", "intervention"),
                "question": f"Does the study match the PICO {key}: {pico[key]}?",
                "scoring_rules": "true only if explicitly satisfied; false if clearly not; null if not stated",
            })
    return Rubric(name=spec["name"], version=spec.get("version", 1), criteria=criteria)


class LintIssue(BaseModel):
    criterion_id: str
    problem: str
    suggested_rewrite: str


class LintReport(BaseModel):
    ambiguous: list[LintIssue] = Field(default_factory=list)


async def lint_rubric(rubric: Rubric, llm_json: LLMJson) -> LintReport:
    sys = "You review systematic-review criteria for ambiguity. Flag only criteria two reviewers could grade differently."
    return LintReport(**await llm_json(sys, rubric.model_dump_json(), LintReport))


# ---------------- Screener ----------------
class Verdict(BaseModel):
    criterion_id: str
    result: Literal["pass", "fail", "unclear"]   # pass = criterion satisfied (inclusion) / not triggered (exclusion)
    rationale: str
    quote: str | None = None


class ScreenOut(BaseModel):
    verdicts: list[Verdict]


def _decide(rubric: Rubric, stage: str, verdicts: list[Verdict]) -> tuple[str, str | None, str]:
    by_id = {v.criterion_id: v for v in verdicts}
    for c in rubric.criteria:
        if c.stage != stage or not c.is_gatekeeper:
            continue
        v = by_id.get(c.criterion_id)
        if v is None or v.result == "unclear":
            return "unclear", None, f"{c.criterion_id} could not be resolved: {(v.rationale if v else 'no verdict')}"
        # exclusion criteria: "true" answer means exclude. Prompt asks the *criterion question*, so
        # for exclusion kind a 'pass' verdict means the exclusion condition is NOT met.
        if v.result == "fail":
            return "exclude", c.criterion_id, v.rationale
    return "include", None, "All gatekeeper criteria satisfied"


SCREEN_SYS = (
    "You are a systematic-review screener. For EACH criterion return result=pass|fail|unclear. "
    "Inclusion: pass = the study satisfies it. Exclusion: pass = the exclusion condition is NOT met, fail = it is met. "
    "Use 'unclear' when the text does not state enough. Never guess. Give a one-sentence rationale and, when possible, "
    "a short verbatim quote."
)


async def screen_paper(conn, rubric_id: str, rubric: Rubric, paper_id: str, stage: Literal["abstract", "methods"],
                       llm_json: LLMJson, model_version: str) -> str:
    crits = [c for c in rubric.criteria if c.stage == stage and c.kind in ("inclusion", "exclusion")]
    if not crits:
        return "include"
    if stage == "abstract":
        r = conn.execute("SELECT title, abstract FROM papers WHERE paper_id=%s", (paper_id,)).fetchone()
        context = f"TITLE: {r[0]}\n\nABSTRACT: {r[1] or '(none)'}"
    else:
        rows = conn.execute(
            "SELECT chunk_id, section, text FROM chunks WHERE paper_id=%s AND section ~* '(method|material|design|participant|population|eligib)' ORDER BY ordinal",
            (paper_id,)).fetchall()
        context = "\n\n".join(f"[chunk {a}] ({b})\n{c}" for a, b, c in rows) or "(no methods section found)"
    user = json.dumps({"criteria": [c.model_dump(include={"criterion_id", "kind", "question", "scoring_rules"}) for c in crits]}) \
        + "\n\nSOURCE:\n" + context
    out = ScreenOut(**await llm_json(SCREEN_SYS, user, ScreenOut))
    decision, code, rationale = _decide(rubric, stage, out.verdicts)
    # Abstract-stage 'unclear' advances to full text rather than excluding (standard SLR practice).
    conn.execute(
        """INSERT INTO screening_decisions (rubric_id, paper_id, stage, decision, exclusion_code, rationale, evidence, model_version)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (rubric_id, paper_id, stage, decision, code, rationale,
         json.dumps([v.model_dump() for v in out.verdicts]), model_version))
    conn.commit()
    return decision


async def run_pipeline(conn, rubric_id, rubric, paper_ids, llm_json, model_version) -> list[str]:
    """Returns paper_ids that survive both gates (abstract 'unclear' advances)."""
    survivors = []
    for pid in paper_ids:
        d1 = await screen_paper(conn, rubric_id, rubric, pid, "abstract", llm_json, model_version)
        if d1 == "exclude":
            continue
        d2 = await screen_paper(conn, rubric_id, rubric, pid, "methods", llm_json, model_version)
        if d2 != "exclude":
            survivors.append(pid)       # 'unclear' at full text goes to human review via the UI
    return survivors


# ---------------- PRISMA counters ----------------
PRISMA_SQL = """
WITH latest AS (
  SELECT DISTINCT ON (paper_id, stage) paper_id, stage,
         coalesce(human_override, decision) AS final, exclusion_code
  FROM screening_decisions WHERE rubric_id=%s
  ORDER BY paper_id, stage, id DESC)
SELECT stage, final, coalesce(exclusion_code,'-') code, count(*) FROM latest GROUP BY 1,2,3 ORDER BY 1,2,3
"""


def prisma_counts(conn, rubric_id: str, identified: int, duplicates_removed: int) -> dict[str, Any]:
    rows = conn.execute(PRISMA_SQL, (rubric_id,)).fetchall()
    ex: dict[str, dict[str, int]] = {"abstract": {}, "methods": {}}
    stage_totals = {"abstract": {"include": 0, "exclude": 0, "unclear": 0}, "methods": {"include": 0, "exclude": 0, "unclear": 0}}
    for stage, final, code, n in rows:
        stage_totals[stage][final] += n
        if final == "exclude":
            ex[stage][code] = n
    screened = identified - duplicates_removed
    full_text = stage_totals["abstract"]["include"] + stage_totals["abstract"]["unclear"]
    return {
        "identified": identified, "duplicates_removed": duplicates_removed, "screened": screened,
        "excluded_at_abstract": stage_totals["abstract"]["exclude"], "abstract_exclusion_reasons": ex["abstract"],
        "full_text_assessed": full_text,
        "excluded_at_full_text": stage_totals["methods"]["exclude"], "full_text_exclusion_reasons": ex["methods"],
        "included": stage_totals["methods"]["include"] + stage_totals["methods"]["unclear"],
    }


# ---------------- Rubric dry-run ----------------
async def dry_run(conn, rubric_id, rubric, seed_paper_ids: list[str], llm_json, model_version, trials: int = 3) -> list[dict]:
    """Run each seed paper `trials` times; report criteria where verdicts disagree. Does not write decisions."""
    report = []
    for pid in seed_paper_ids[:5]:
        r = conn.execute("SELECT title, abstract FROM papers WHERE paper_id=%s", (pid,)).fetchone()
        crits = [c for c in rubric.criteria if c.kind in ("inclusion", "exclusion") and c.stage == "abstract"]
        user = json.dumps({"criteria": [c.model_dump(include={"criterion_id", "question", "scoring_rules"}) for c in crits]}) \
            + f"\n\nSOURCE:\nTITLE: {r[0]}\nABSTRACT: {r[1]}"
        seen: dict[str, set[str]] = {}
        for _ in range(trials):
            out = ScreenOut(**await llm_json(SCREEN_SYS, user, ScreenOut))
            for v in out.verdicts:
                seen.setdefault(v.criterion_id, set()).add(v.result)
        report.append({"paper_id": pid, "disagreements": {k: sorted(v) for k, v in seen.items() if len(v) > 1}})
    return report
