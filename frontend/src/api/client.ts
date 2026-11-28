import { runBrowserModel, type BrowserModel } from "./browserModel";
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
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      "X-Workspace-ID": getWorkspaceId(),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function browserOperation<T>(
  action: string, params: object, model: BrowserModel,
  onProgress?: (completed: number, total: number) => void,
) {
  return runBrowserModel<T>(API_BASE, getWorkspaceId(), action, params, model, onProgress);
}

async function download(path: string): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "X-Workspace-ID": getWorkspaceId() },
    });
  } catch {
    throw new Error("Studio could not reach the export endpoint. Check that the backend is running, then try the download again.");
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(typeof detail?.detail === "string" ? detail.detail : `Export failed (HTTP ${response.status}).`);
  }
  return response.blob();
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

  ingestPdf: (file: File, browserModel = false) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ paper_id: string; pdf_url: string }>("/api/ingest", {
      method: "POST",
      body: fd,
      headers: browserModel ? { "X-SLR-Retrieval": "keywords" } : undefined,
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

  exportXlsx: (rubricId: string) => download(`/api/export/${encodeURIComponent(rubricId)}/xlsx`),

  exportAnnotatedPdf: (rubricId: string, paperId: string) =>
    download(`/api/export/${encodeURIComponent(rubricId)}/papers/${encodeURIComponent(paperId)}/annotated.pdf`),
};
