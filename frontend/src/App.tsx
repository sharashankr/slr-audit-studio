import { useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  Download,
  Sparkles,
  RefreshCw,
  CheckCheck,
  Search,
  Upload,
  Play,
  Loader2,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  FilePlus2,
  PanelLeftClose,
  PanelLeftOpen,
  Table2,
  FolderOpen,
  ArrowUpRight,
} from "lucide-react";
import { api } from "./api/client";
import type {
  MatrixCell,
  MatrixResponse,
  RubricCreateResponse,
  SynthesisResponse,
} from "./types/api";
import { InspectorDrawer } from "./components/InspectorDrawer";

const DEFAULT_RUBRIC_ID = "28247498-ee3d-4aeb-98a0-5c588631af49";

const SAMPLE_YAML = `name: Oncology Causal ML Extraction
version: 1
criteria:
  - criterion_id: STUDY_DESIGN
    kind: extraction
    question: What causal inference or observational study design was used?
    data_type: string
    stage: extraction
    section_hints: ["Methods", "Study Design"]
  - criterion_id: PATIENT_POPULATION
    kind: extraction
    question: Describe the cancer cohort, sample size, and eligibility criteria.
    data_type: string
    stage: extraction
    section_hints: ["Methods", "Participants"]
`;

function statusBadge(status: string) {
  switch (status) {
    case "approved":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    case "verified":
    case "supported":
      return "bg-sky-50 text-sky-700 border-sky-200";
    case "overridden":
      return "bg-[#EFF6FF] text-[#1D4ED8] border-blue-200";
    case "flagged":
    case "rejected":
    case "unsupported":
    case "wrong_value":
      return "bg-amber-50 text-amber-800 border-amber-200";
    default:
      return "bg-slate-100 text-slate-600 border-slate-200";
  }
}

function renderCellValue(val: unknown): string {
  if (val === null || val === undefined || val === "") return "—";
  if (typeof val === "boolean") return val ? "Yes" : "No";
  if (typeof val === "object") return JSON.stringify(val);
  return String(val);
}

export function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const matrixSectionRef = useRef<HTMLElement | null>(null);
  const uploadSectionRef = useRef<HTMLElement | null>(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeNav, setActiveNav] = useState<"studio" | "matrix" | "rubric">("studio");

  const [rubricId, setRubricId] = useState(DEFAULT_RUBRIC_ID);
  const [rubricInput, setRubricInput] = useState(DEFAULT_RUBRIC_ID);
  const [matrix, setMatrix] = useState<MatrixResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bannerMsg, setBannerMsg] = useState<string | null>(null);

  // Upload & Pipeline State
  const [dragging, setDragging] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState<string[]>([]);
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [runningPipeline, setRunningPipeline] = useState(false);
  const [pipelineStage, setPipelineStage] = useState<string>("");
  const [prismaCounts, setPrismaCounts] = useState<Record<string, number> | null>(null);

  // Optional Custom YAML Rubric Editor
  const [showRubricEditor, setShowRubricEditor] = useState(false);
  const [yamlText, setYamlText] = useState(SAMPLE_YAML);
  const [rubricResult, setRubricResult] = useState<RubricCreateResponse | null>(null);
  const [compilingRubric, setCompilingRubric] = useState(false);

  // Matrix Filter & Inspector State
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCell, setSelectedCell] = useState<MatrixCell | null>(null);
  const [synthesis, setSynthesis] = useState<SynthesisResponse | null>(null);
  const [synthLoading, setSynthLoading] = useState(false);

  const loadMatrix = async (id: string, autoSelectNew = false) => {
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getMatrix(id.trim());
      setMatrix(data);
      if (autoSelectNew || selectedPaperIds.length === 0) {
        setSelectedPaperIds(data.papers.map((p) => p.paper_id));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadMatrix(rubricId, true);
  }, [rubricId]);

  const handleFilesUpload = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const pdfs = Array.from(fileList).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    );
    if (pdfs.length === 0) {
      setError("Please select one or more .pdf files.");
      return;
    }

    setError(null);
    setBannerMsg(null);
    const addedIds: string[] = [];

    for (const file of pdfs) {
      setUploadingFiles((prev) => [...prev, file.name]);
      try {
        const res = await api.ingestPdf(file);
        addedIds.push(res.paper_id);
      } catch (e) {
        setError(
          `Failed ingesting ${file.name}: ${e instanceof Error ? e.message : String(e)}`
        );
      } finally {
        setUploadingFiles((prev) => prev.filter((n) => n !== file.name));
      }
    }

    if (addedIds.length > 0) {
      await loadMatrix(rubricId);
      setSelectedPaperIds((prev) => Array.from(new Set([...prev, ...addedIds])));
      setBannerMsg(
        `Uploaded & indexed ${addedIds.length} PDF(s). Click "Run Evidence Extraction" below to extract criteria.`
      );
    }
  };

  const handleRunExtractionPipeline = async () => {
    const targets =
      selectedPaperIds.length > 0
        ? selectedPaperIds
        : (matrix?.papers || []).map((p) => p.paper_id);

    if (targets.length === 0) {
      setError("Upload at least one PDF paper before running extraction.");
      return;
    }

    setRunningPipeline(true);
    setError(null);
    setBannerMsg(null);

    try {
      setPipelineStage("Stage 1/2: Running PRISMA Abstract & Methods Screening...");
      const screenRes = await api.screenPapers(rubricId, targets);
      setPrismaCounts(screenRes.prisma);

      const survivors =
        screenRes.survivors.length > 0 ? screenRes.survivors : targets;

      setPipelineStage(
        `Stage 2/2: Extracting & Critic-verifying ${survivors.length} paper(s)...`
      );
      await api.extractPapers(rubricId, survivors);

      await loadMatrix(rubricId);
      setBannerMsg(
        `Extraction complete for ${survivors.length} paper(s). Click any cell below to inspect PDF quote provenance.`
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunningPipeline(false);
      setPipelineStage("");
    }
  };

  const handleCreateRubric = async () => {
    setCompilingRubric(true);
    setError(null);
    try {
      const res = await api.createRubric(yamlText);
      setRubricResult(res);
      setRubricInput(res.rubric_id);
      setRubricId(res.rubric_id);
      setBannerMsg(`Compiled rubric "${res.rubric.name}" (${res.rubric_id}).`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCompilingRubric(false);
    }
  };

  const cellLookup = useMemo(() => {
    const map = new Map<string, MatrixCell>();
    for (const c of matrix?.cells || []) {
      map.set(`${c.paper_id}::${c.criterion_id}`, c);
    }
    return map;
  }, [matrix]);

  const stats = useMemo(() => {
    const cells = matrix?.cells || [];
    const verified = cells.filter((c) =>
      ["verified", "supported", "approved", "overridden"].includes(c.critic_status)
    ).length;
    const flagged = cells.filter((c) =>
      !["verified", "supported", "approved", "overridden"].includes(c.critic_status)
    ).length;
    const approvableIds = cells
      .filter((c) => ["verified", "supported"].includes(c.critic_status))
      .map((c) => c.cell_id);
    return { total: cells.length, verified, flagged, approvableIds };
  }, [matrix]);

  const filteredPapers = useMemo(() => {
    if (!matrix) return [];
    return matrix.papers.filter((p) => {
      if (
        searchQuery &&
        !p.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !p.paper_id.toLowerCase().includes(searchQuery.toLowerCase())
      ) {
        return false;
      }
      if (onlyFlagged) {
        return matrix.criteria.some((cid) => {
          const cell = cellLookup.get(`${p.paper_id}::${cid}`);
          return (
            cell &&
            !["verified", "supported", "approved", "overridden"].includes(
              cell.critic_status
            )
          );
        });
      }
      return true;
    });
  }, [matrix, searchQuery, onlyFlagged, cellLookup]);

  const replaceCellInState = (updated: MatrixCell) => {
    setMatrix((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        cells: prev.cells.map((c) =>
          c.paper_id === updated.paper_id && c.criterion_id === updated.criterion_id
            ? updated
            : c
        ),
      };
    });
    setSelectedCell(updated);
  };

  const handleApproveOne = async (cellId: number) => {
    const [updated] = await api.approveCells(rubricId, [cellId]);
    if (updated) replaceCellInState(updated);
  };

  const handleBulkApproveVerified = async () => {
    if (stats.approvableIds.length === 0) return;
    await api.approveCells(rubricId, stats.approvableIds);
    await loadMatrix(rubricId);
  };

  const handleOverride = async (
    cellId: number,
    value: unknown,
    reason: string
  ) => {
    const updated = await api.overrideCell(cellId, value, reason);
    replaceCellInState(updated);
  };

  const handleSynthesize = async () => {
    setSynthLoading(true);
    try {
      const res = await api.synthesize(rubricId);
      setSynthesis(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSynthLoading(false);
    }
  };

  const paperMap = useMemo(() => {
    return new Map(matrix?.papers.map((p) => [p.paper_id, p]));
  }, [matrix]);

  const togglePaperSelection = (pid: string) => {
    setSelectedPaperIds((prev) =>
      prev.includes(pid) ? prev.filter((id) => id !== pid) : [...prev, pid]
    );
  };

  const scrollToSection = (
    nav: "studio" | "matrix" | "rubric",
    el: HTMLElement | null
  ) => {
    setActiveNav(nav);
    if (nav === "rubric") {
      setShowRubricEditor(true);
    }
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="flex min-h-screen bg-[#F8FAFC] text-[#0F172A]">
      {/* Left Navigation Rail (Scite-inspired) */}
      <aside
        className={`sticky top-0 flex h-screen shrink-0 flex-col justify-between border-r border-[#E2E8F0] bg-white transition-all duration-200 ${
          sidebarCollapsed ? "w-16" : "w-60"
        }`}
      >
        <div className="p-3">
          {/* Brand Wordmark + Collapse Button */}
          <div className="mb-5 flex items-center justify-between px-1.5 pt-1">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[#2563EB] text-white">
                <ShieldCheck className="h-4 w-4" />
              </div>
              {!sidebarCollapsed && (
                <div className="flex items-baseline gap-0.5 truncate">
                  <span className="font-display text-lg font-bold tracking-tight text-[#0F172A]">
                    slr
                  </span>
                  <span className="font-mono text-base font-bold text-[#2563EB]">
                    _audit
                  </span>
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={() => setSidebarCollapsed((v) => !v)}
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              className="rounded-md p-1.5 text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A]"
            >
              {sidebarCollapsed ? (
                <PanelLeftOpen className="h-4 w-4" />
              ) : (
                <PanelLeftClose className="h-4 w-4" />
              )}
            </button>
          </div>

          {/* Primary New Review CTA */}
          <button
            type="button"
            onClick={() => {
              fileInputRef.current?.click();
            }}
            className={`mb-4 flex w-full items-center justify-center gap-2 rounded-lg border border-[#E2E8F0] bg-white py-2 text-xs font-semibold text-[#0F172A] shadow-2xs transition hover:border-[#CBD5E1] hover:bg-[#F8FAFC] ${
              sidebarCollapsed ? "px-0" : "px-3"
            }`}
            title="Upload PDF Paper"
          >
            <Upload className="h-3.5 w-3.5 text-[#2563EB]" />
            {!sidebarCollapsed && <span>Upload PDF Paper</span>}
          </button>

          {/* Navigation Items */}
          <nav className="space-y-1 text-xs font-medium">
            <button
              type="button"
              onClick={() => scrollToSection("studio", uploadSectionRef.current)}
              className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 transition ${
                activeNav === "studio"
                  ? "bg-[#EFF6FF] font-semibold text-[#1D4ED8]"
                  : "text-[#475569] hover:bg-[#F1F5F9] hover:text-[#0F172A]"
              }`}
            >
              <Sparkles className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && <span>Extraction Studio</span>}
            </button>

            <button
              type="button"
              onClick={() => scrollToSection("matrix", matrixSectionRef.current)}
              className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 transition ${
                activeNav === "matrix"
                  ? "bg-[#EFF6FF] font-semibold text-[#1D4ED8]"
                  : "text-[#475569] hover:bg-[#F1F5F9] hover:text-[#0F172A]"
              }`}
            >
              <Table2 className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && (
                <span className="flex-1 text-left">Evidence Matrix</span>
              )}
              {!sidebarCollapsed && (
                <span className="rounded-full bg-[#F1F5F9] px-1.5 py-0.5 font-mono text-[10px] text-[#475569] tabular-nums">
                  {matrix?.papers.length || 0}
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => {
                setActiveNav("rubric");
                setShowRubricEditor((v) => !v);
              }}
              className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 transition ${
                activeNav === "rubric"
                  ? "bg-[#EFF6FF] font-semibold text-[#1D4ED8]"
                  : "text-[#475569] hover:bg-[#F1F5F9] hover:text-[#0F172A]"
              }`}
            >
              <FolderOpen className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && <span>YAML Rubric</span>}
            </button>
          </nav>

          {/* Active Corpus Quick List */}
          {!sidebarCollapsed && (
            <div className="mt-6 border-t border-[#E2E8F0] pt-4">
              <div className="mb-2 flex items-center justify-between px-2">
                <span className="text-[11px] font-semibold tracking-wider text-[#64748B] uppercase">
                  Corpus Papers
                </span>
                <span className="font-mono text-[11px] text-[#64748B] tabular-nums">
                  {selectedPaperIds.length}/{(matrix?.papers || []).length}
                </span>
              </div>
              <div className="max-h-56 space-y-1 overflow-y-auto pr-0.5">
                {(matrix?.papers || []).map((p) => {
                  const checked = selectedPaperIds.includes(p.paper_id);
                  return (
                    <button
                      key={p.paper_id}
                      type="button"
                      onClick={() => togglePaperSelection(p.paper_id)}
                      className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition ${
                        checked
                          ? "bg-[#F8FAFC] text-[#0F172A]"
                          : "text-[#64748B] hover:bg-[#F1F5F9]"
                      }`}
                    >
                      <span
                        className={`h-2 w-2 shrink-0 rounded-full ${
                          checked ? "bg-[#2563EB]" : "bg-[#CBD5E1]"
                        }`}
                      />
                      <span className="truncate">{p.title}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Bottom Workspace Status */}
        {!sidebarCollapsed && (
          <div className="border-t border-[#E2E8F0] p-3 text-[11px] text-[#64748B]">
            <div className="flex items-center justify-between">
              <span>Verified Cells</span>
              <span className="font-mono font-semibold text-[#0F172A] tabular-nums">
                {stats.verified}/{stats.total}
              </span>
            </div>
            <div className="mt-1 flex items-center justify-between">
              <span>Critic Flagged</span>
              <span className="font-mono font-semibold text-amber-700 tabular-nums">
                {stats.flagged}
              </span>
            </div>
          </div>
        )}
      </aside>

      {/* Main Column (Top Bar + Workspace) */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top Utility Header */}
        <header className="sticky top-0 z-20 border-b border-[#E2E8F0] bg-white/95 backdrop-blur-xs">
          <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3 px-6 py-2.5">
            {/* Rubric Switcher */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                setRubricId(rubricInput.trim());
              }}
              className="flex items-center gap-2"
            >
              <span className="text-xs font-medium text-[#64748B]">Rubric</span>
              <input
                type="text"
                value={rubricInput}
                onChange={(e) => setRubricInput(e.target.value)}
                className="w-64 rounded-lg border border-[#E2E8F0] bg-[#F8FAFC] px-2.5 py-1.5 font-mono text-xs text-[#0F172A] focus:border-[#2563EB] focus:bg-white focus:outline-hidden sm:w-72"
              />
              <button
                type="submit"
                className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E8F0] bg-white px-2.5 py-1.5 text-xs font-medium text-[#0F172A] hover:bg-[#F8FAFC]"
              >
                <RefreshCw
                  className={`h-3.5 w-3.5 text-[#64748B] ${
                    loading ? "animate-spin" : ""
                  }`}
                />
                Load
              </button>
            </form>

            {/* Right Action Cluster */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={stats.approvableIds.length === 0}
                onClick={handleBulkApproveVerified}
                className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E8F0] bg-white px-3 py-1.5 text-xs font-medium text-[#0F172A] transition hover:bg-[#F8FAFC] disabled:opacity-40"
              >
                <CheckCheck className="h-3.5 w-3.5 text-emerald-600" />
                <span>Approve Verified ({stats.approvableIds.length})</span>
              </button>

              <button
                type="button"
                disabled={synthLoading || stats.total === 0}
                onClick={handleSynthesize}
                className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E8F0] bg-white px-3 py-1.5 text-xs font-medium text-[#0F172A] transition hover:bg-[#F8FAFC] disabled:opacity-40"
              >
                <Sparkles className="h-3.5 w-3.5 text-[#2563EB]" />
                <span>{synthLoading ? "Synthesizing..." : "Synthesis"}</span>
              </button>

              <a
                href={api.exportXlsxUrl(rubricId)}
                className="inline-flex items-center gap-1.5 rounded-lg bg-[#2563EB] px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-[#1D4ED8]"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Export .xlsx</span>
              </a>
            </div>
          </div>
        </header>

        {/* Main Content Canvas */}
        <main className="mx-auto w-full max-w-[1440px] flex-1 space-y-6 px-6 py-7">
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs font-medium text-rose-800">
              {error}
            </div>
          )}
          {bannerMsg && (
            <div className="flex items-center gap-2 rounded-xl border border-blue-200 bg-[#EFF6FF] px-4 py-3 text-xs font-medium text-[#1D4ED8]">
              <CheckCircle2 className="h-4 w-4 shrink-0 text-[#2563EB]" />
              <span>{bannerMsg}</span>
            </div>
          )}

          {/* Hero Title Row (Scite-style clean centered/editorial framing) */}
          <div className="text-center">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-[#E2E8F0] bg-white px-3 py-1 text-[11px] font-medium text-[#475569]">
              <span className="h-1.5 w-1.5 rounded-full bg-[#2563EB]" />
              Quote-Level Provenance & Critic-Verified Extraction
            </span>
            <h1 className="font-display mt-2.5 text-2xl font-bold tracking-tight text-[#0F172A] sm:text-3xl">
              What evidence are you auditing?
            </h1>
            <p className="mx-auto mt-1.5 max-w-2xl text-sm text-[#475569]">
              Upload research PDFs, run PRISMA screening and grounded extraction, and
              verify every claim against highlighted source pages.
            </p>
          </div>

          {/* Primary Two-Column Workbench: About + Upload & Run */}
          <div
            ref={uploadSectionRef}
            className="grid grid-cols-1 gap-5 lg:grid-cols-12"
          >
            {/* About SLR-Audit Studio */}
            <section className="flex flex-col justify-between rounded-xl border border-[#E2E8F0] bg-white p-6 lg:col-span-5">
              <div>
                <h2 className="font-display text-base font-bold text-[#0F172A]">
                  About SLR-Audit Studio
                </h2>
                <p className="mt-2 text-xs leading-relaxed text-[#475569]">
                  Designed for systematic literature reviews where unverified LLM
                  summaries are unacceptable. Every extracted value pairs a verbatim
                  quote and page bounding box with an independent Critic verdict.
                </p>

                <div className="mt-4 space-y-3 border-t border-[#F1F5F9] pt-4 text-xs">
                  <div className="flex gap-3">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#EFF6FF] font-mono text-[11px] font-semibold text-[#2563EB]">
                      1
                    </span>
                    <div>
                      <p className="font-semibold text-[#0F172A]">
                        Docling Layout & Vector Indexing
                      </p>
                      <p className="text-[#64748B]">
                        Preserves exact page numbers and bounding boxes across text,
                        tables, and figures.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#EFF6FF] font-mono text-[11px] font-semibold text-[#2563EB]">
                      2
                    </span>
                    <div>
                      <p className="font-semibold text-[#0F172A]">
                        Screening, Extraction & Critic Audit
                      </p>
                      <p className="text-[#64748B]">
                        Filters papers via PRISMA gates, extracts rubric fields, and
                        flags unsupported claims.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#EFF6FF] font-mono text-[11px] font-semibold text-[#2563EB]">
                      3
                    </span>
                    <div>
                      <p className="font-semibold text-[#0F172A]">
                        Split-Screen PDF Provenance
                      </p>
                      <p className="text-[#64748B]">
                        Click any cell to jump straight to the highlighted PDF passage,
                        approve, or override.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-5 border-t border-[#F1F5F9] pt-3.5">
                <button
                  type="button"
                  onClick={() => setShowRubricEditor((v) => !v)}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#2563EB] hover:text-[#1D4ED8]"
                >
                  <FilePlus2 className="h-3.5 w-3.5" />
                  <span>
                    {showRubricEditor
                      ? "Hide Custom YAML Rubric Editor"
                      : "Configure Custom YAML Rubric"}
                  </span>
                  {showRubricEditor ? (
                    <ChevronUp className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5" />
                  )}
                </button>
              </div>
            </section>

            {/* Upload Paper & Run Evidence Extraction */}
            <section className="flex flex-col justify-between rounded-xl border border-[#E2E8F0] bg-white p-6 lg:col-span-7">
              <div className="space-y-4">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h2 className="font-display text-base font-bold text-[#0F172A]">
                      Upload Paper & Run Evidence Extraction
                    </h2>
                    <p className="mt-0.5 text-xs text-[#64748B]">
                      Add new PDFs to the corpus, select papers, and run the
                      multi-agent model
                    </p>
                  </div>
                  <span className="rounded-full border border-[#E2E8F0] bg-[#F8FAFC] px-2.5 py-1 font-mono text-[11px] font-medium text-[#475569] tabular-nums">
                    {matrix?.criteria.length || 0} criteria
                  </span>
                </div>

                {/* Dropzone */}
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    void handleFilesUpload(e.dataTransfer.files);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-dashed px-5 py-4 transition ${
                    dragging
                      ? "border-[#2563EB] bg-[#EFF6FF]"
                      : "border-[#CBD5E1] bg-[#F8FAFC] hover:border-[#2563EB] hover:bg-[#EFF6FF]/40"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,application/pdf"
                    multiple
                    className="hidden"
                    onChange={(e) => void handleFilesUpload(e.target.files)}
                  />
                  <div className="flex items-center gap-3.5">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#EFF6FF] text-[#2563EB]">
                      <Upload className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-[#0F172A]">
                        Drop PDF paper(s) here or click to browse
                      </p>
                      <p className="text-[11px] text-[#64748B]">
                        Parses layout coordinates with Docling & indexes embeddings
                      </p>
                    </div>
                  </div>
                  <span className="shrink-0 rounded-lg border border-[#E2E8F0] bg-white px-3 py-1.5 text-xs font-semibold text-[#0F172A]">
                    Browse PDFs
                  </span>
                </div>

                {uploadingFiles.length > 0 && (
                  <div className="space-y-1 rounded-lg border border-blue-200 bg-[#EFF6FF] px-3.5 py-2.5 text-xs text-[#1D4ED8]">
                    {uploadingFiles.map((name) => (
                      <div key={name} className="flex items-center gap-2">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        <span>Ingesting & indexing {name}...</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Paper Selection List */}
                <div>
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <span className="font-semibold text-[#334155] tabular-nums">
                      Target Papers ({selectedPaperIds.length} of{" "}
                      {(matrix?.papers || []).length} selected)
                    </span>
                    {(matrix?.papers.length || 0) > 0 && (
                      <button
                        type="button"
                        onClick={() =>
                          setSelectedPaperIds(
                            selectedPaperIds.length === (matrix?.papers.length || 0)
                              ? []
                              : (matrix?.papers || []).map((p) => p.paper_id)
                          )
                        }
                        className="text-[11px] font-semibold text-[#2563EB] hover:underline"
                      >
                        {selectedPaperIds.length === (matrix?.papers.length || 0)
                          ? "Deselect all"
                          : "Select all"}
                      </button>
                    )}
                  </div>

                  <div className="max-h-32 space-y-1 overflow-y-auto rounded-lg border border-[#E2E8F0] bg-[#F8FAFC] p-2">
                    {(matrix?.papers || []).map((p) => (
                      <label
                        key={p.paper_id}
                        className="flex cursor-pointer items-center gap-2.5 rounded-md border border-transparent bg-white px-2.5 py-1.5 text-xs transition hover:border-[#E2E8F0]"
                      >
                        <input
                          type="checkbox"
                          checked={selectedPaperIds.includes(p.paper_id)}
                          onChange={() => togglePaperSelection(p.paper_id)}
                          className="h-3.5 w-3.5 rounded border-[#CBD5E1] text-[#2563EB] focus:ring-[#2563EB]"
                        />
                        <span className="truncate font-medium text-[#0F172A]">
                          {p.title}
                        </span>
                        <span className="ml-auto font-mono text-[10px] text-[#94A3B8]">
                          {p.paper_id.slice(0, 8)}
                        </span>
                      </label>
                    ))}
                    {(!matrix || matrix.papers.length === 0) && (
                      <p className="py-3 text-center text-xs text-[#64748B]">
                        No papers uploaded yet. Drop a PDF above to begin.
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Run Extraction Footer */}
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-[#F1F5F9] pt-4">
                <div className="text-xs text-[#64748B]">
                  {runningPipeline ? (
                    <span className="inline-flex items-center gap-1.5 font-medium text-[#2563EB]">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      {pipelineStage}
                    </span>
                  ) : prismaCounts ? (
                    <span className="font-mono text-[11px] text-[#475569] tabular-nums">
                      PRISMA:{" "}
                      {Object.entries(prismaCounts)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(" · ")}
                    </span>
                  ) : (
                    <span>PRISMA Screening → Grounded Extractor → Critic</span>
                  )}
                </div>

                <button
                  type="button"
                  disabled={
                    runningPipeline ||
                    uploadingFiles.length > 0 ||
                    selectedPaperIds.length === 0
                  }
                  onClick={() => void handleRunExtractionPipeline()}
                  className="inline-flex items-center gap-2 rounded-lg bg-[#2563EB] px-4 py-2 text-xs font-semibold text-white transition hover:bg-[#1D4ED8] disabled:opacity-45"
                >
                  {runningPipeline ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-3.5 w-3.5 fill-current" />
                  )}
                  <span>
                    {runningPipeline
                      ? "Running Model..."
                      : `Run Evidence Extraction (${selectedPaperIds.length})`}
                  </span>
                </button>
              </div>
            </section>
          </div>

          {/* Collapsible YAML Rubric Editor */}
          {showRubricEditor && (
            <section className="space-y-3 rounded-xl border border-[#E2E8F0] bg-white p-6">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="font-display text-sm font-bold text-[#0F172A]">
                    YAML Rubric Compiler & Ambiguity Linter
                  </h3>
                  <p className="text-xs text-[#64748B]">
                    Define extraction criteria and lint for underspecified or
                    ambiguous rules
                  </p>
                </div>
                <button
                  type="button"
                  disabled={compilingRubric}
                  onClick={() => void handleCreateRubric()}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-[#2563EB] px-3.5 py-2 text-xs font-semibold text-white hover:bg-[#1D4ED8] disabled:opacity-50"
                >
                  {compilingRubric ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <FilePlus2 className="h-3.5 w-3.5" />
                  )}
                  <span>Compile & Activate Rubric</span>
                </button>
              </div>
              <textarea
                rows={7}
                value={yamlText}
                onChange={(e) => setYamlText(e.target.value)}
                className="w-full rounded-lg border border-[#E2E8F0] bg-[#0F172A] p-3.5 font-mono text-xs text-slate-100 focus:border-[#2563EB] focus:outline-hidden"
              />
              {rubricResult && rubricResult.lint.ambiguous.length > 0 && (
                <div className="space-y-2 rounded-lg border border-amber-200 bg-amber-50 p-3.5 text-xs">
                  <div className="flex items-center gap-1.5 font-semibold text-amber-900">
                    <AlertTriangle className="h-4 w-4 text-amber-600" />
                    <span>Rubric Ambiguity Lint Warnings</span>
                  </div>
                  {rubricResult.lint.ambiguous.map((issue) => (
                    <div
                      key={issue.criterion_id}
                      className="rounded-md border border-amber-100 bg-white p-2.5"
                    >
                      <span className="font-mono font-semibold text-amber-900">
                        {issue.criterion_id}:
                      </span>{" "}
                      <span className="text-[#334155]">{issue.problem}</span>
                      <div className="mt-1 text-[11px] font-medium text-emerald-700">
                        Suggested rewrite: {issue.suggested_rewrite}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Grounded Synthesis Output Card */}
          {synthesis && (
            <section className="rounded-xl border border-blue-200 bg-white p-6">
              <div className="mb-2.5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-[#2563EB]" />
                  <h3 className="font-display text-sm font-bold text-[#0F172A]">
                    Grounded Cross-Paper Synthesis
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={() => setSynthesis(null)}
                  className="text-xs font-medium text-[#64748B] hover:text-[#0F172A]"
                >
                  Dismiss
                </button>
              </div>
              <p className="whitespace-pre-line text-xs leading-relaxed text-[#1E293B]">
                {synthesis.narrative}
              </p>
              {synthesis.discrepancies.length > 0 && (
                <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3.5">
                  <p className="text-xs font-semibold text-amber-900">
                    Cross-Paper Discrepancies
                  </p>
                  <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-amber-800">
                    {synthesis.discrepancies.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}

          {/* Section 2: Evidence Extraction Output Matrix */}
          <section ref={matrixSectionRef} className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-[#E2E8F0] bg-white px-4 py-3">
              <div className="flex flex-wrap items-center gap-4 text-xs tabular-nums">
                <span className="font-display text-sm font-bold text-[#0F172A]">
                  Extracted Evidence Matrix
                </span>
                <span className="text-[#64748B]">
                  {matrix?.papers.length || 0} papers ×{" "}
                  {matrix?.criteria.length || 0} criteria
                </span>
                <span className="inline-flex items-center gap-1 font-medium text-emerald-700">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {stats.verified} verified
                </span>
                <span className="inline-flex items-center gap-1 font-medium text-amber-700">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  {stats.flagged} flagged
                </span>
              </div>

              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="pointer-events-none absolute top-2 left-2.5 h-3.5 w-3.5 text-[#94A3B8]" />
                  <input
                    type="text"
                    placeholder="Filter papers..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="rounded-lg border border-[#E2E8F0] bg-[#F8FAFC] py-1.5 pr-3 pl-8 text-xs text-[#0F172A] focus:border-[#2563EB] focus:bg-white focus:outline-hidden"
                  />
                </div>

                <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs font-medium text-[#334155]">
                  <input
                    type="checkbox"
                    checked={onlyFlagged}
                    onChange={(e) => setOnlyFlagged(e.target.checked)}
                    className="h-3.5 w-3.5 rounded border-[#CBD5E1] text-[#2563EB]"
                  />
                  <span>Flagged only</span>
                </label>
              </div>
            </div>

            <div className="overflow-x-auto rounded-xl border border-[#E2E8F0] bg-white">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-[#E2E8F0] bg-[#F8FAFC]">
                    <th className="sticky left-0 z-20 min-w-[260px] border-r border-[#E2E8F0] bg-[#F8FAFC] px-4 py-3 font-semibold text-[#334155]">
                      Paper / Study
                    </th>
                    {(matrix?.criteria || []).map((cid) => (
                      <th
                        key={cid}
                        className="min-w-[196px] border-r border-[#E2E8F0] px-3 py-3 font-mono text-[11px] font-semibold text-[#334155]"
                      >
                        {cid}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E8F0]">
                  {filteredPapers.map((paper) => (
                    <tr
                      key={paper.paper_id}
                      className="transition-colors hover:bg-[#F8FAFC]/80"
                    >
                      <td className="sticky left-0 z-10 border-r border-[#E2E8F0] bg-white px-4 py-3 align-top">
                        <div className="line-clamp-2 font-medium text-[#0F172A]">
                          {paper.title}
                        </div>
                        <div className="mt-1 font-mono text-[10px] text-[#94A3B8]">
                          {paper.paper_id.slice(0, 8)}
                        </div>
                      </td>
                      {(matrix?.criteria || []).map((cid) => {
                        const cell = cellLookup.get(`${paper.paper_id}::${cid}`);
                        if (!cell) {
                          return (
                            <td
                              key={cid}
                              className="border-r border-[#F1F5F9] px-3 py-3 font-mono text-[#CBD5E1]"
                            >
                              —
                            </td>
                          );
                        }
                        return (
                          <td
                            key={cid}
                            onClick={() => setSelectedCell(cell)}
                            className="group cursor-pointer border-r border-[#F1F5F9] px-3 py-2.5 align-top transition-colors hover:bg-[#EFF6FF]/65"
                          >
                            <div className="flex items-start justify-between gap-1.5">
                              <span className="line-clamp-2 font-medium text-[#0F172A]">
                                {renderCellValue(cell.value)}
                              </span>
                              <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-[#94A3B8] opacity-0 transition group-hover:text-[#2563EB] group-hover:opacity-100" />
                            </div>
                            <div className="mt-1.5 flex items-center justify-between gap-1">
                              <span
                                className={`inline-block rounded-md border px-1.5 py-0.5 text-[10px] font-semibold ${statusBadge(
                                  cell.critic_status
                                )}`}
                              >
                                {cell.critic_status}
                              </span>
                              {cell.page_number && (
                                <span className="font-mono text-[10px] text-[#64748B] tabular-nums">
                                  p.{cell.page_number}
                                </span>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                  {!loading && filteredPapers.length === 0 && (
                    <tr>
                      <td
                        colSpan={(matrix?.criteria.length || 0) + 1}
                        className="px-6 py-12 text-center text-xs text-[#64748B]"
                      >
                        Upload a PDF and click{" "}
                        <span className="font-semibold text-[#0F172A]">
                          Run Evidence Extraction
                        </span>{" "}
                        above to populate the matrix.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </main>
      </div>

      {/* Split-Screen PDF Provenance Inspector */}
      <InspectorDrawer
        cell={selectedCell}
        paper={selectedCell ? paperMap.get(selectedCell.paper_id) : undefined}
        onClose={() => setSelectedCell(null)}
        onApprove={handleApproveOne}
        onOverride={handleOverride}
      />
    </div>
  );
}

export default App;
