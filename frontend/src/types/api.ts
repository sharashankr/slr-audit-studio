export interface BBox {
  page: number;
  l: number;
  t: number;
  r: number;
  b: number;
  w: number;
  h: number;
}

export type CriticStatus =
  | "verified"
  | "approved"
  | "overridden"
  | "flagged"
  | "rejected"
  | "unverified"
  | string;

export interface PaperSummary {
  paper_id: string;
  title: string;
  pdf_url: string;
  doi?: string | null;
  year?: number | null;
}

export interface MatrixCell {
  cell_id: number;
  paper_id: string;
  criterion_id: string;
  value: unknown;
  verbatim_quote: string | null;
  page_number: number | null;
  bbox: BBox[] | BBox | null;
  confidence: number;
  critic_status: CriticStatus;
  critic_notes: string | null;
}

export interface MatrixResponse {
  rubric_name: string;
  papers: PaperSummary[];
  criteria: string[];
  cells: MatrixCell[];
}

export interface LintIssue {
  criterion_id: string;
  problem: string;
  suggested_rewrite: string;
}

export interface RubricCreateResponse {
  rubric_id: string;
  rubric: {
    name: string;
    version: number;
    criteria: Array<{
      criterion_id: string;
      kind: string;
      question: string;
      data_type: string;
      stage: string;
    }>;
  };
  lint: {
    ambiguous: LintIssue[];
  };
}

export interface SynthesisResponse {
  narrative: string;
  discrepancies: string[];
}
