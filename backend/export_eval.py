"""Phase 5 - Cross-paper synthesis, CSV/Excel/RIS/PRISMA export, field-level precision/recall harness.

pip install pandas openpyxl
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import pandas as pd
from pydantic import BaseModel

from rubric_screening import LLMJson, prisma_counts


# ---------------- Evidence matrix ----------------
def load_matrix(conn, rubric_id: str, include_unverified: bool = False, *, workspace_id: str | None = None) -> pd.DataFrame:
    ok = "('verified','approved','overridden')" if not include_unverified else "('verified','approved','overridden','needs_review')"
    workspace_filter = "AND p.workspace_id=%s" if workspace_id is not None else ""
    params = (rubric_id, workspace_id) if workspace_id is not None else (rubric_id,)
    rows = conn.execute(f"""
        SELECT p.paper_id, p.doi, p.title, p.year, c.criterion_id, c.value, c.verbatim_quote, c.page_number,
               c.confidence, c.critic_status
        FROM current_cells c JOIN papers p USING (paper_id)
        WHERE c.rubric_id=%s AND c.critic_status IN {ok} {workspace_filter}""", params).fetchall()
    cols = ["paper_id", "doi", "title", "year", "criterion_id", "value", "quote", "page", "confidence", "status"]
    return pd.DataFrame(rows, columns=cols)


def wide(df: pd.DataFrame) -> pd.DataFrame:
    base = df.drop_duplicates("paper_id")[["paper_id", "doi", "title", "year"]].set_index("paper_id")
    vals = df.pivot(index="paper_id", columns="criterion_id", values="value")
    return base.join(vals).reset_index()


# ---------------- Exports ----------------
def export_csv(df: pd.DataFrame, path: str):
    wide(df).to_csv(path, index=False)               # values only
    df.to_csv(path.replace(".csv", "_long.csv"), index=False)   # audit version: quote/page/confidence per cell


def export_excel(df: pd.DataFrame, prisma: dict, path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        wide(df).to_excel(xw, "Matrix", index=False)
        df.to_excel(xw, "Provenance", index=False)
        pd.DataFrame(sorted(prisma.items(), key=lambda kv: kv[0]), columns=["item", "value"]).astype(str).to_excel(xw, "PRISMA", index=False)


def export_ris(conn, rubric_id: str, path: str):
    rows = conn.execute("""SELECT DISTINCT p.title, p.doi, p.year, p.authors FROM papers p
                           JOIN current_cells c USING (paper_id) WHERE c.rubric_id=%s""", (rubric_id,)).fetchall()
    with open(path, "w", encoding="utf-8") as f:
        for title, doi, year, authors in rows:
            f.write("TY  - JOUR\n")
            for a in (authors or []):
                f.write(f"AU  - {a}\n")
            f.write(f"TI  - {title or ''}\n")
            if year: f.write(f"PY  - {year}\n")
            if doi: f.write(f"DO  - {doi}\n")
            f.write("ER  - \n\n")


def export_prisma(conn, rubric_id: str, identified: int, duplicates: int, path: str):
    p = prisma_counts(conn, rubric_id, identified, duplicates)
    with open(path, "w") as f:
        json.dump(p, f, indent=2)
    lines = [f"Records identified: {p['identified']}", f"Duplicates removed: {p['duplicates_removed']}",
             f"Records screened: {p['screened']}", f"Excluded at title/abstract: {p['excluded_at_abstract']}",
             f"Full texts assessed: {p['full_text_assessed']}", f"Excluded at full text: {p['excluded_at_full_text']}"]
    lines += [f"   - {k}: {v}" for k, v in p["full_text_exclusion_reasons"].items()]
    lines.append(f"Studies included: {p['included']}")
    with open(path.replace(".json", ".txt"), "w") as f:
        f.write("\n".join(lines))
    return p


# ---------------- Synthesis ----------------
class Synthesis(BaseModel):
    narrative: str            # every claim must carry [paper_id:criterion_id] tags
    discrepancies: list[str]


SYNTH_SYS = ("Write a narrative synthesis of the evidence matrix. Cite every factual claim as [paper_id:criterion_id]. "
             "Use only supplied values; report heterogeneity and conflicting values explicitly; do not pool or compute statistics.")


async def draft_synthesis(df: pd.DataFrame, llm_json: LLMJson) -> Synthesis:
    payload = df[["paper_id", "criterion_id", "value", "quote", "page"]].to_json(orient="records")
    s = Synthesis(**await llm_json(SYNTH_SYS, payload, Synthesis))
    valid = {f"{r.paper_id}:{r.criterion_id}" for r in df.itertuples()}
    cited = set(re.findall(r"\[([0-9a-f-]{36}:[A-Za-z0-9_]+)\]", s.narrative))
    bad = cited - valid
    if bad:                                   # deterministic check: no citation to a cell that doesn't exist
        raise ValueError(f"Synthesis cites unknown cells: {sorted(bad)[:5]}")
    return s


# ---------------- Evaluation harness ----------------
def norm_value(v: Any, data_type: str = "string") -> Any:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if data_type in ("int", "float"):
        try:
            return round(float(str(v).replace(",", "")), 4)
        except ValueError:
            return str(v).strip().lower()
    if isinstance(v, bool): return v
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def evaluate(pred: dict[tuple[str, str], Any], gold: dict[tuple[str, str], Any],
             dtypes: dict[str, str] | None = None) -> dict[str, dict[str, float]]:
    """pred/gold: {(paper_id, criterion_id): value}. A missing/None gold value means 'not reported'.
      TP: pred non-null and equals gold   FP: pred non-null and (gold null or different)
      FN: gold non-null and (pred null or different)      Different values count as BOTH FP and FN.
    Returns per-field and 'ALL' (micro) precision / recall / F1."""
    dtypes = dtypes or {}
    tally = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for key in set(pred) | set(gold):
        cid = key[1]
        p = norm_value(pred.get(key), dtypes.get(cid, "string"))
        g = norm_value(gold.get(key), dtypes.get(cid, "string"))
        for bucket in (cid, "ALL"):
            t = tally[bucket]
            if p is not None and g is not None and p == g: t["tp"] += 1
            else:
                if p is not None: t["fp"] += 1
                if g is not None: t["fn"] += 1
    out = {}
    for k, t in tally.items():
        prec = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 0.0
        rec = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 0.0
        out[k] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0, **t}
    return out


def grounding_rate(conn, rubric_id: str) -> float:
    """Share of non-null, non-rejected cells whose quote is stored with a chunk anchor (should be 1.0)."""
    r = conn.execute("""SELECT avg((chunk_id IS NOT NULL AND verbatim_quote IS NOT NULL)::int)
                        FROM current_cells WHERE rubric_id=%s AND value IS NOT NULL AND critic_status<>'rejected'""", (rubric_id,)).fetchone()
    return float(r[0] or 0)


def regression_gate(current: dict, baseline: dict, max_drop: float = 0.02) -> list[str]:
    """CI gate: fail if any field's F1 drops more than max_drop vs the saved baseline."""
    return [f"{k}: F1 {baseline[k]['f1']} -> {current.get(k, {}).get('f1', 0)}"
            for k in baseline if baseline[k]["f1"] - current.get(k, {}).get("f1", 0) > max_drop]
