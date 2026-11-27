import type {
  MatrixCell,
  MatrixResponse,
  RubricCreateResponse,
  SynthesisResponse,
} from "../types/api";

export const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export function resolvePdfUrl(pdfUrl: string): string {
  if (!pdfUrl) return "";
  if (pdfUrl.startsWith("http://") || pdfUrl.startsWith("https://")) return pdfUrl;
  return `${API_BASE}${pdfUrl.startsWith("/") ? "" : "/"}${pdfUrl}`;
}

// ---------------------------------------------------------------------------
// Workspace isolation — a random UUID stored in localStorage so each browser
// session gets its own siloed view of papers and rubrics on the shared backend.
// ---------------------------------------------------------------------------
function getWorkspaceId(): string {
  const key = "slr_ws_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      "X-Workspace-ID": getWorkspaceId(),
      ...init?.headers,
    },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  getMatrix: (rubricId: string) =>
    request<MatrixResponse>(`/api/matrix/${encodeURIComponent(rubricId)}`),

  approveCells: (rubricId: string, cellIds: number[]) =>
    request<MatrixCell[]>("/api/cells/approve", {
      method: "POST",
      body: JSON.stringify({ rubric_id: rubricId, cell_ids: cellIds }),
    }),

  overrideCell: (cellId: number, value: unknown, reason: string) =>
    request<MatrixCell>("/api/cells/override", {
      method: "POST",
      body: JSON.stringify({ cell_id: cellId, value, reason }),
    }),

  createRubric: (yamlText: string) =>
    request<RubricCreateResponse>("/api/rubrics", {
      method: "POST",
      body: JSON.stringify({ yaml_text: yamlText }),
    }),

  ingestPdf: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ paper_id: string; pdf_url: string }>("/api/ingest", {
      method: "POST",
      body: fd,
    });
  },

  screenPapers: (rubricId: string, paperIds: string[]) =>
    request<{ survivors: string[]; prisma: Record<string, number> }>("/api/screen", {
      method: "POST",
      body: JSON.stringify({ rubric_id: rubricId, paper_ids: paperIds }),
    }),

  extractPapers: (rubricId: string, paperIds: string[]) =>
    request<{ extracted: Record<string, unknown[]> }>("/api/extract", {
      method: "POST",
      body: JSON.stringify({ rubric_id: rubricId, paper_ids: paperIds }),
    }),

  synthesize: (rubricId: string) =>
    request<SynthesisResponse>(`/api/synthesize/${encodeURIComponent(rubricId)}`, {
      method: "POST",
    }),

  exportXlsxUrl: (rubricId: string) =>
    `${API_BASE}/api/export/${encodeURIComponent(rubricId)}/xlsx`,
};
