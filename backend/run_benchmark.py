"""Run calibration or held-out evaluation directly from backend/data/gold_standard.xlsx and ~/Documents/Papers."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from main import app

DEFAULT_PAPERS_DIR = Path("/Users/sharathchandrashankar/Documents/Papers")
DEFAULT_XLSX_PATH = BACKEND_DIR / "data" / "gold_standard.xlsx"
RUBRIC_PATH = BACKEND_DIR / "oncology_causal_ml_rubric.yaml"

# Exact column index mapping from the 'Data' sheet in gold_standard.xlsx
COL_MAP = {
    2: "STUDY_DESIGN",
    3: "PATIENT_POPULATION",
    4: "COMPARATORS",
    5: "OBS_CONFOUNDERS_ID",
    6: "OBS_CONFOUNDING_STRATEGY",
    7: "OBS_BALANCE_DIAGNOSTICS",
    8: "OBS_UNMEASURED_CONFOUNDING",
    9: "PREDEFINED_EM",
    10: "ML_METHOD_TYPE",
    11: "ML_METHOD_CLARITY",
    12: "OUTCOME_METHOD_ALIGNMENT",
    13: "HYPERPARAM_TUNING",
    14: "OVERFITTING_CONTROL_BOOL",
    16: "MULTI_MODEL_COMPARISON",
    18: "GENERIC_CAUSAL_ESTIMAND",
    19: "SURVIVAL_CAUSAL_ESTIMAND",
    22: "CLINICAL_DECISION_READINESS",
    23: "UNCERTAINTY_QUANTIFICATION",
    24: "BIOLOGICAL_PLAUSIBILITY",
    25: "POSTHOC_EM",
    26: "CODE_AVAILABILITY",
}
SCORED_CRITERIA = list(COL_MAP.values())


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _clean_score(cid: str, raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "NA"
    s = str(raw).strip()
    if not s or s == "\xa0":
        return "NA"
    if cid == "OVERFITTING_CONTROL_BOOL" and s.lower().startswith("yes"):
        return "2"
    try:
        val = float(s)
        return str(int(round(val + 1e-9)))
    except ValueError:
        return s


def load_gold_workbook(xlsx_path: Path) -> pd.DataFrame:
    df_list = pd.read_excel(xlsx_path, sheet_name="List")
    meta = {}
    for idx in range(2, len(df_list)):
        pid = df_list.iloc[idx, 1]
        if pd.notna(pid):
            meta[int(float(pid))] = {
                "title": str(df_list.iloc[idx, 2]).replace("\n", " ").strip(),
                "author": str(df_list.iloc[idx, 3]).strip(),
            }

    df_data = pd.read_excel(xlsx_path, sheet_name="Data")
    records_by_id: dict[int, dict] = {}

    # Even rows (2, 4, 6...) = scores; odd rows (3, 5, 7...) = reviewer comments for that paper
    for r_idx in range(2, len(df_data), 2):
        raw_id = df_data.iloc[r_idx, 1]
        if raw_id is None or (isinstance(raw_id, float) and pd.isna(raw_id)) or str(raw_id).strip() == "":
            continue
        pid = int(float(raw_id))
        comment_row = df_data.iloc[r_idx + 1] if (r_idx + 1) < len(df_data) else None

        rec = {
            "paper_num": pid,
            "title": meta.get(pid, {}).get("title", ""),
            "author": meta.get(pid, {}).get("author", ""),
        }
        non_na = 0
        for col_idx, cid in COL_MAP.items():
            cleaned = _clean_score(cid, df_data.iloc[r_idx, col_idx])
            rec[cid] = cleaned
            if cleaned != "NA":
                non_na += 1

            comment_val = ""
            if comment_row is not None and col_idx < len(comment_row):
                raw_c = comment_row.iloc[col_idx]
                if pd.notna(raw_c) and str(raw_c).strip() not in ("", "\xa0", "nan"):
                    comment_val = str(raw_c).strip()
            rec[f"{cid}__comment"] = comment_val

        rec["_non_na_count"] = non_na
        if pid not in records_by_id or non_na > records_by_id[pid]["_non_na_count"]:
            records_by_id[pid] = rec

    return pd.DataFrame([records_by_id[k] for k in sorted(records_by_id)])


def _read_page1(pdf: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf))
        if reader.pages:
            return _norm_text(reader.pages[0].extract_text() or "")
    except Exception:
        pass
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf))
        if len(doc) > 0:
            textpage = doc[0].get_textpage()
            return _norm_text(textpage.get_text_range() or "")
    except Exception:
        pass
    return ""


def build_pdf_index(papers_dir: Path) -> list[dict]:
    return [
        {
            "path": pdf,
            "stem_norm": _norm_text(pdf.stem),
            "page1_norm": _read_page1(pdf),
        }
        for pdf in sorted(papers_dir.glob("*.pdf"))
    ]


def match_pdf(row: pd.Series, pdf_index: list[dict]) -> Path | None:
    pid = int(row["paper_num"])
    title_words = [w for w in _norm_text(row["title"]).split() if len(w) >= 5][:6]
    author_last = _norm_text(row["author"]).split()[-1] if row["author"] else ""

    # 1. Exact numeric prefix in filename (e.g., "1.pdf", "01_...", "Paper 1...")
    for item in pdf_index:
        tokens = item["stem_norm"].split()
        if tokens and (tokens[0] == str(pid) or (len(tokens) > 1 and tokens[0] == "paper" and tokens[1] == str(pid))):
            return item["path"]

    # 2. Title keyword + author surname overlap against page 1 text or filename
    best_path, best_score = None, 0
    for item in pdf_index:
        haystack = f"{item['stem_norm']} {item['page1_norm']}"
        score = sum(1 for w in title_words if w in haystack)
        if author_last and len(author_last) >= 4 and author_last in haystack:
            score += 2
        if score > best_score:
            best_score = score
            best_path = item["path"]

    return best_path if best_score >= 3 else None


def score_predictions(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    merged = gold_df.merge(pred_df, on="paper_num", suffixes=("_gold", "_pred"))
    rows = []
    for cid in SCORED_CRITERIA:
        g_col, p_col = f"{cid}_gold", f"{cid}_pred"
        if g_col not in merged.columns or p_col not in merged.columns:
            continue
        g = merged[g_col].astype(str).str.strip()
        p = merged[p_col].astype(str).str.strip()
        exact = (g == p).mean() if len(merged) else 0.0

        labels = sorted(set(g) | set(p))
        f1s, precs, recs = [], [], []
        for lbl in labels:
            tp = ((g == lbl) & (p == lbl)).sum()
            fp = ((g != lbl) & (p == lbl)).sum()
            fn = ((g == lbl) & (p != lbl)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

        rows.append({
            "criterion_id": cid,
            "n": len(merged),
            "exact_acc": round(float(exact), 3),
            "macro_prec": round(sum(precs) / len(precs), 3),
            "macro_rec": round(sum(recs) / len(recs), 3),
            "macro_f1": round(sum(f1s) / len(f1s), 3),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX_PATH)
    parser.add_argument("--split", choices=["calibration", "test", "all"], default="calibration")
    parser.add_argument("--dry-run-match", action="store_true", help="Verify PDF matching without calling APIs")
    args = parser.parse_args()

    gold = load_gold_workbook(args.xlsx)
    pdf_index = build_pdf_index(args.papers_dir)
    gold["matched_pdf"] = gold.apply(lambda r: match_pdf(r, pdf_index), axis=1)

    # Keep only rows that have a PDF on disk; first 5 = calibration, remainder = held-out test
    matched_gold = gold[gold["matched_pdf"].notna()].reset_index(drop=True)
    matched_gold["split"] = ["calibration" if i < 5 else "test" for i in range(len(matched_gold))]

    unmatched = gold[gold["matched_pdf"].isna()]
    if not unmatched.empty:
        print(f"Skipping {len(unmatched)} workbook rows with no PDF on disk: {list(unmatched['paper_num'])}")

    if args.split != "all":
        matched_gold = matched_gold[matched_gold["split"] == args.split].copy()

    print(f"Loaded {len(matched_gold)} matched papers for split='{args.split}' (from {len(pdf_index)} PDFs in {args.papers_dir}):")
    for _, r in matched_gold.iterrows():
        print(f"  Paper {int(r['paper_num']):2d} [{r['split']:11s}] -> {r['matched_pdf'].name} | {r['title'][:65]}")

    if args.dry_run_match:
        return

    with TestClient(app) as client:
        r_resp = client.post("/api/rubrics", json={"yaml_text": RUBRIC_PATH.read_text()})
        r_resp.raise_for_status()
        rubric_id = r_resp.json()["rubric_id"]
        print(f"\nRegistered rubric_id={rubric_id}")

        ckpt_path = args.papers_dir / f".checkpoint_{args.split}.json"
        ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
        done_by_num = {int(item["record"]["paper_num"]): item for item in ckpt.get("completed", [])}

        pred_rows, mismatches = [], []
        for _, row in matched_gold.iterrows():
            pnum = int(row["paper_num"])
            pdf_path: Path = row["matched_pdf"]
            if pnum in done_by_num:
                item = done_by_num[pnum]
                pred_rows.append(item["record"])
                mismatches.extend(item["mismatches"])
                print(f"Resumed Paper {pnum:2d} ({pdf_path.name}) -> paper_id={item['record']['paper_id']}")
                continue

            with pdf_path.open("rb") as f:
                ing = client.post("/api/ingest", files={"file": (pdf_path.name, f, "application/pdf")})
            ing.raise_for_status()
            paper_id = ing.json()["paper_id"]

            ext = client.post("/api/extract", json={"rubric_id": rubric_id, "paper_ids": [paper_id]})
            ext.raise_for_status()
            cells = ext.json()["extracted"][paper_id]

            pred_record = {"paper_num": pnum, "pdf_filename": pdf_path.name, "paper_id": paper_id}
            paper_mismatches = []
            for cell in cells:
                cid = cell["criterion_id"]
                val = str(cell["value"]) if cell["value"] is not None else "NA"
                pred_record[cid] = val
                if cid in SCORED_CRITERIA and cid in row:
                    g_val = str(row[cid]).strip()
                    if val != g_val:
                        paper_mismatches.append({
                            "paper_num": pnum,
                            "pdf": pdf_path.name,
                            "criterion": cid,
                            "gold": g_val,
                            "gold_comment": row.get(f"{cid}__comment", ""),
                            "pred": val,
                            "status": cell.get("critic_status"),
                            "quote": (cell.get("verbatim_quote") or "")[:120],
                        })

            pred_rows.append(pred_record)
            mismatches.extend(paper_mismatches)
            done_by_num[pnum] = {"record": pred_record, "mismatches": paper_mismatches}
            ckpt_path.write_text(json.dumps({"completed": list(done_by_num.values())}, indent=2))
            print(f"Completed Paper {pnum:2d} ({pdf_path.name}) -> paper_id={paper_id}")

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(args.papers_dir / f"predictions_{args.split}.csv", index=False)

    metrics_df = score_predictions(matched_gold, pred_df)
    metrics_df.to_csv(args.papers_dir / f"metrics_{args.split}.csv", index=False)

    print("\n=== PER-CRITERION METRICS ===")
    print(metrics_df.to_string(index=False))
    print(f"\nOverall Mean Exact Accuracy: {metrics_df['exact_acc'].mean():.3f}")
    print(f"Overall Mean Macro F1:       {metrics_df['macro_f1'].mean():.3f}")

    if mismatches:
        mismatch_path = args.papers_dir / f"mismatches_{args.split}.json"
        mismatch_path.write_text(json.dumps(mismatches, indent=2))
        print(f"\nSaved {len(mismatches)} cell discrepancies to {mismatch_path}")


if __name__ == "__main__":
    main()
