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
  FilePlus2,
  Table2,
  FolderOpen,
  ArrowUpRight,
  KeyRound,
  Layers,
  SlidersHorizontal,
  FlaskConical,
  Settings2,
  Eye,
  EyeOff,
  Zap,
} from "lucide-react";
import { buildEvidenceCsv } from "./api/evidenceCsv";
import { api, browserOperation, getBetaCode, setBetaCode } from "./api/client";
import type {
  MatrixCell,
  MatrixResponse,
  PaperSummary,
  RubricCreateResponse,
  SynthesisResponse,
} from "./types/api";
import { testBrowserModel, type BrowserModel } from "./api/browserModel";
import { InspectorDrawer } from "./components/InspectorDrawer";

const ACTIVE_RUBRIC_KEY = "slr_active_rubric";

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

// ---------------------------------------------------------------------------
// LLM Provider / Model Configuration
// ---------------------------------------------------------------------------
const LLM_PROVIDERS = ["Google Gemini", "Groq", "Mistral"] as const;
type LLMProvider = (typeof LLM_PROVIDERS)[number];

const PROVIDER_MODELS: Record<LLMProvider, string[]> = {
  "Google Gemini": ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
  Groq: ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
  Mistral: ["mistral-small-latest", "mistral-medium-latest"],
};

type StudioView = "studio" | "calibration" | "training" | "validation" | "matrix" | "settings";

const VIEW_TITLES: Record<StudioView, string> = {
  studio: "Extraction",
  calibration: "Calibration",
  training: "Rubric editor",
  validation: "Validation",
  matrix: "Evidence matrix",
  settings: "Models & API keys",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function statusBadge(status: string) {
  switch (status) {
    case "approved":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    case "verified":
    case "supported":
      return "bg-sky-50 text-sky-700 border-sky-200";
    case "overridden":
      return "bg-indigo-50 text-indigo-700 border-indigo-200";
    case "flagged":
    case "rejected":
    case "unsupported":
    case "wrong_value":
      return "bg-amber-50 text-amber-800 border-amber-200";
    default:
      return "bg-slate-100 text-slate-500 border-slate-200";
  }
}

function renderCellValue(val: unknown): string {
  if (val === null || val === undefined || val === "") return "—";
  if (typeof val === "boolean") return val ? "Yes" : "No";
  if (typeof val === "object") return JSON.stringify(val);
  return String(val);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
function NavItem({
  icon,
  label,
  active,
  badge,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  badge?: string | number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-current={active ? "page" : undefined}
      onClick={onClick}
      className={`group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all ${
        active
          ? "bg-blue-600 text-white shadow-sm"
          : "text-slate-400 hover:bg-white/8 hover:text-white"
      }`}
    >
      <span className={`shrink-0 ${active ? "text-white" : "text-slate-500 group-hover:text-slate-300"}`}>
        {icon}
      </span>
      <span className="flex-1 text-left truncate">{label}</span>
      {badge !== undefined && (
        <span
          className={`rounded-full px-1.5 py-0.5 font-mono text-[10px] tabular-nums ${
            active ? "bg-white/20 text-white" : "bg-white/8 text-slate-400"
          }`}
        >
          {badge}
        </span>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
export function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [activeNav, setActiveNav] = useState<StudioView>("studio");

  // Rubric
  const [rubricId, setRubricId] = useState(() => localStorage.getItem(ACTIVE_RUBRIC_KEY) ?? "");
  const [rubricInput, setRubricInput] = useState(rubricId);
  const [matrix, setMatrix] = useState<MatrixResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bannerMsg, setBannerMsg] = useState<string | null>(null);

  // Upload & Pipeline
  const [dragging, setDragging] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState<string[]>([]);
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [runningPipeline, setRunningPipeline] = useState(false);
  const [pipelineStage, setPipelineStage] = useState("");
  const [pausedRun, setPausedRun] = useState<{
    rubricId: string; model: Pick<BrowserModel, "provider" | "model">; papers: string[]; completed: number;
  } | null>(null);
  const [prismaCounts, setPrismaCounts] = useState<Record<string, number> | null>(null);

  // Rubric Editor
  const [yamlText, setYamlText] = useState(SAMPLE_YAML);
  const [rubricResult, setRubricResult] = useState<RubricCreateResponse | null>(null);
  const [compilingRubric, setCompilingRubric] = useState(false);

  // Matrix Filter & Inspector
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCell, setSelectedCell] = useState<MatrixCell | null>(null);
  const [synthesis, setSynthesis] = useState<SynthesisResponse | null>(null);
  const [synthLoading, setSynthLoading] = useState(false);

  // AI Settings (BYOK)
  const [llmProvider, setLlmProvider] = useState<LLMProvider>("Google Gemini");
  const [llmModel, setLlmModel] = useState(PROVIDER_MODELS["Google Gemini"][0]);
  const [apiKeys, setApiKeys] = useState<Partial<Record<LLMProvider, string>>>({});
  const [showApiKey, setShowApiKey] = useState(false);
  const [betaAccessRequired, setBetaAccessRequired] = useState(false);
  const [betaCode, setBetaCodeInput] = useState(getBetaCode);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectedModel, setConnectedModel] = useState<BrowserModel>();
  const [connectionError, setConnectionError] = useState<string>();
  const apiKey = apiKeys[llmProvider] ?? "";
  const browserModel = apiKey.trim()
    ? { provider: llmProvider, model: llmModel, key: apiKey.trim() }
    : undefined;
  const connected = connectedModel !== undefined && browserModel !== undefined &&
    connectedModel.provider === browserModel.provider && connectedModel.model === browserModel.model &&
    connectedModel.key === browserModel.key;
  const guidedRun = activeNav === "calibration" || activeNav === "validation";

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
    if (rubricId) {
      localStorage.setItem(ACTIVE_RUBRIC_KEY, rubricId);
      void loadMatrix(rubricId, true);
    }
  }, [rubricId]);

  useEffect(() => {
    void api.health().then((result) => setBetaAccessRequired(Boolean(result.beta_access_required))).catch(() => {});
  }, []);

  const handleTestConnection = async () => {
    if (!browserModel) return;
    setTestingConnection(true);
    setConnectionError(undefined);
    setConnectedModel(undefined);
    try {
      setBetaCode(betaCode.trim());
      await api.checkBetaAccess();
      await testBrowserModel(browserModel);
      if (rubricId) await loadMatrix(rubricId);
      setConnectedModel(browserModel);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : "Connection test failed.");
    } finally {
      setTestingConnection(false);
    }
  };

  const handleFilesUpload = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0 || !rubricId || !connected) return;
    const pdfs = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) { setError("Please select one or more .pdf files."); return; }
    setError(null); setBannerMsg(null);
    const addedIds: string[] = [];
    for (const file of pdfs) {
      setUploadingFiles((prev) => [...prev, file.name]);
      try {
        const res = await api.ingestPdf(file, true);
        addedIds.push(res.paper_id);
      } catch (e) {
        setError(`Failed ingesting ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setUploadingFiles((prev) => prev.filter((n) => n !== file.name));
      }
    }
    if (addedIds.length > 0) {
      await loadMatrix(rubricId);
      setSelectedPaperIds((prev) => Array.from(new Set([...prev, ...addedIds])));
      setBannerMsg(`Uploaded & indexed ${addedIds.length} PDF(s). Click "Run Extraction" to begin.`);
    }
  };

  const handleRunExtractionPipeline = async (resume = false) => {
    if (!browserModel || !connected || !rubricId) return;
    const previous = resume ? pausedRun : null;
    if (resume && (!previous || previous.rubricId !== rubricId ||
        previous.model.provider !== browserModel.provider || previous.model.model !== browserModel.model)) return;
    if (!resume && selectedPaperIds.length === 0) return;
    setRunningPipeline(true);
    setError(null);
    setBannerMsg(null);
    let run = previous;
    try {
      if (!run) {
        setPausedRun(null);
        setPipelineStage("Screening selected papers…");
        const screen = await browserOperation<Awaited<ReturnType<typeof api.screenPapers>>>(
          "screen", { rubric_id: rubricId, paper_ids: selectedPaperIds }, browserModel,
        );
        setPrismaCounts(screen.prisma);
        if (!screen.survivors.length) {
          setBannerMsg("Screening excluded all selected papers.");
          return;
        }
        run = { rubricId, model: { provider: browserModel.provider, model: browserModel.model }, papers: screen.survivors, completed: 0 };
      }
      while (run.completed < run.papers.length) {
        const id = run.papers[run.completed];
        const title = paperMap.get(id)?.title ?? "Paper";
        const label = `Paper ${run.completed + 1} of ${run.papers.length} · ${title}`;
        setPipelineStage(label);
        await browserOperation("extract", { rubric_id: rubricId, paper_ids: [id] }, browserModel,
          (completed, total) => setPipelineStage(`${label} · ${completed}/${total} criteria saved`));
        run = { ...run, completed: run.completed + 1 };
      }
      setPausedRun(null);
      await loadMatrix(rubricId);
      setActiveNav("matrix");
      setBannerMsg(`Extraction complete for ${run.completed} paper(s). Choose a cell to inspect its source.`);
    } catch (error) {
      setPausedRun(run);
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      setRunningPipeline(false);
      setPipelineStage("");
    }
  };

  const handleCreateRubric = async () => {
    if (!browserModel || !connected) return;
    setCompilingRubric(true); setError(null);
    try {
      const res = await browserOperation<RubricCreateResponse>("rubric", { yaml_text: yamlText }, browserModel);
      setRubricResult(res);
      setRubricInput(res.rubric_id);
      if (res.rubric_id !== rubricId) {
        setMatrix(null);
        setSelectedPaperIds([]);
        setRubricId(res.rubric_id);
      }
      setBannerMsg(`Activated ${res.rubric.name}.`);
    } catch (e) { setError(e instanceof Error ? e.message : String(e));
    } finally { setCompilingRubric(false); }
  };

  const cellLookup = useMemo(() => {
    const map = new Map<string, MatrixCell>();
    for (const c of matrix?.cells || []) map.set(`${c.paper_id}::${c.criterion_id}`, c);
    return map;
  }, [matrix]);

  const stats = useMemo(() => {
    const cells = matrix?.cells || [];
    const verified = cells.filter((c) => ["verified","supported","approved","overridden"].includes(c.critic_status)).length;
    const flagged = cells.filter((c) => !["verified","supported","approved","overridden"].includes(c.critic_status)).length;
    const approvableIds = cells.filter((c) => ["verified","supported"].includes(c.critic_status)).map((c) => c.cell_id);
    return { total: cells.length, verified, flagged, approvableIds };
  }, [matrix]);

  const filteredPapers = useMemo(() => {
    if (!matrix) return [];
    return matrix.papers.filter((p) => {
      if (searchQuery && !p.title.toLowerCase().includes(searchQuery.toLowerCase()) && !p.paper_id.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      if (onlyFlagged) return matrix.criteria.some((cid) => { const cell = cellLookup.get(`${p.paper_id}::${cid}`); return cell && !["verified","supported","approved","overridden"].includes(cell.critic_status); });
      return true;
    });
  }, [matrix, searchQuery, onlyFlagged, cellLookup]);

  const replaceCellInState = (updated: MatrixCell) => {
    setMatrix((prev) => {
      if (!prev) return prev;
      return { ...prev, cells: prev.cells.map((c) => c.paper_id === updated.paper_id && c.criterion_id === updated.criterion_id ? updated : c) };
    });
    setSelectedCell(updated);
  };

  const saveDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const handleExportPdf = async (paper: PaperSummary) => {
    const name = paper.title.replace(/[<>:"/\\|?*]/g, "-").trim().slice(0, 100);
    const blob = await api.exportAnnotatedPdf(rubricId, paper.paper_id);
    saveDownload(blob, `${name || "paper"}-annotated.pdf`);
  };

  const handleExportCsv = () => {
    if (!matrix) return;
    setError(null);
    saveDownload(new Blob(["\uFEFF", buildEvidenceCsv(matrix)], { type: "text/csv;charset=utf-8" }),
      "slr-extraction-provenance.csv");
  };

  const handleExport = async () => {
    setError(null);
    try {
      saveDownload(await api.exportXlsx(rubricId), "slr-evidence-matrix.xlsx");
    } catch (error) {
      setError(error instanceof Error ? error.message : "Export failed.");
    }
  };

  const handleApproveOne = async (cellId: number) => { const [updated] = await api.approveCells(rubricId, [cellId]); if (updated) replaceCellInState(updated); };
  const handleBulkApproveVerified = async () => { if (!stats.approvableIds.length) return; await api.approveCells(rubricId, stats.approvableIds); await loadMatrix(rubricId); };
  const handleOverride = async (cellId: number, value: unknown, reason: string) => { const updated = await api.overrideCell(cellId, value, reason); replaceCellInState(updated); };
  const handleSynthesize = async () => { setSynthLoading(true); try { const res = browserModel ? await browserOperation<SynthesisResponse>("synthesize", { rubric_id: rubricId }, browserModel) : await api.synthesize(rubricId); setSynthesis(res); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setSynthLoading(false); } };

  const paperMap = useMemo(() => new Map(matrix?.papers.map((p) => [p.paper_id, p])), [matrix]);
  const togglePaperSelection = (pid: string) => setSelectedPaperIds((prev) => prev.includes(pid) ? prev.filter((id) => id !== pid) : [...prev, pid]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="flex h-dvh flex-col overflow-hidden md:flex-row bg-[#0F1117] text-white">
      {/* ── LEFT SIDEBAR ── */}
      <aside className="flex shrink-0 flex-col md:w-56 border-r border-white/8 bg-[#161B27]">
        {/* Brand */}
        <div className="flex items-center gap-2.5 px-4 py-3 border-b border-white/8">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-600">
            <ShieldCheck className="h-4.5 w-4.5 text-white" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-[15px] font-bold tracking-tight text-white">SLR Audit Studio</span>
          </div>
        </div>

        <input ref={fileInputRef} type="file" accept=".pdf,application/pdf" multiple className="hidden" onChange={(e) => {
          void handleFilesUpload(e.target.files);
          e.target.value = "";
        }} />
        <nav aria-label="Studio workspace" className="grid grid-cols-2 gap-1 px-3 py-2 sm:grid-cols-3 md:flex md:flex-col">
          <NavItem icon={<Layers size={16} />} label={VIEW_TITLES.studio} active={activeNav === "studio"} onClick={() => setActiveNav("studio")} />
          <NavItem icon={<SlidersHorizontal size={16} />} label={VIEW_TITLES.calibration} active={activeNav === "calibration"} onClick={() => setActiveNav("calibration")} />
          <NavItem icon={<FolderOpen size={16} />} label={VIEW_TITLES.training} active={activeNav === "training"} onClick={() => setActiveNav("training")} />
          <NavItem icon={<FlaskConical size={16} />} label={VIEW_TITLES.validation} active={activeNav === "validation"} onClick={() => setActiveNav("validation")} />
          <NavItem icon={<Table2 size={16} />} label={VIEW_TITLES.matrix} active={activeNav === "matrix"} onClick={() => setActiveNav("matrix")} />
          <NavItem icon={<KeyRound size={16} />} label={VIEW_TITLES.settings} active={activeNav === "settings"} onClick={() => setActiveNav("settings")} />
        </nav>
        <div className="mt-auto hidden border-t border-white/8 px-4 py-3 text-xs text-slate-400 md:block">
          {matrix?.papers.length ?? 0} papers · {matrix?.criteria.length ?? 0} criteria
        </div>
      </aside>

      {/* ── MAIN AREA ── */}
      <div className="flex min-h-0 flex-1 flex-col min-w-0 bg-[#0F1117]">
        {/* Top Bar */}
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/8 bg-[#161B27]/80 px-4 py-3 sm:px-6 backdrop-blur-sm">
          <h1 className="text-sm font-semibold">{VIEW_TITLES[activeNav]}</h1>
          <button type="button" onClick={() => setActiveNav("settings")} className="flex items-center gap-2 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10">
            <Settings2 size={14} /> {connected ? `${llmProvider} connected` : "Connect model"}
          </button>
        </header>
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-6xl space-y-5 px-4 py-5 sm:px-6">
            {(activeNav !== "settings" && activeNav !== "training") && (
              <div className="flex flex-wrap items-center justify-between gap-3">
                {/* Rubric switcher */}
                <form onSubmit={(e) => { e.preventDefault(); if (rubricInput.trim() && rubricInput.trim() !== rubricId) { setMatrix(null); setSelectedPaperIds([]); setRubricId(rubricInput.trim()); } else if (rubricId) { void loadMatrix(rubricId); } }} className="flex items-center gap-2 min-w-0">
                  <span className="shrink-0 text-xs font-medium text-slate-400">{matrix?.rubric_name ?? "Rubric"}</span>
                  <input
                    type="text"
                    value={rubricInput}
                    onChange={(e) => setRubricInput(e.target.value)}
                    aria-label="Rubric ID"
                    placeholder="Existing rubric ID"
                    className="min-w-0 w-44 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 font-mono text-xs text-slate-300 placeholder-slate-600 focus:border-blue-500 focus:bg-white/8 focus:outline-none sm:w-72"
                  />
                  <button type="submit" disabled={!rubricInput.trim() || loading || runningPipeline || uploadingFiles.length > 0} className="flex shrink-0 items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-white/10 hover:text-white transition">
                    <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
                    Load
                  </button>
                </form>

                {/* Action cluster */}
                {activeNav === "matrix" && <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    disabled={stats.approvableIds.length === 0}
                    onClick={handleBulkApproveVerified}
                    className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
                  >
                    <CheckCheck className="h-3.5 w-3.5 text-emerald-400" />
                    Approve All ({stats.approvableIds.length})
                  </button>
                  <button
                    type="button"
                    disabled={!connected || synthLoading || stats.total === 0}
                    onClick={handleSynthesize}
                    className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:bg-white/10 hover:text-white disabled:opacity-30"
                  >
                    <Sparkles className="h-3.5 w-3.5 text-blue-400" />
                    {synthLoading ? "Synthesizing…" : "Synthesis"}
                  </button>
                  <button
                    type="button"
                    disabled={stats.total === 0}
                    onClick={handleExportCsv}
                    className="flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-white/10 disabled:opacity-40"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Export CSV
                  </button>
                  <button
                    type="button"
                    disabled={!rubricId || stats.total === 0}
                    onClick={() => void handleExport()}
                    className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-blue-500"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Export .xlsx
                  </button>
                </div>}
              </div>
            )}

            {/* Error / Banner */}
            {error && (
              <div role="alert" className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-xs font-medium text-rose-300">
                {error}
              </div>
            )}
            {bannerMsg && (
              <div role="status" className="flex items-center gap-2 rounded-xl border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-xs font-medium text-blue-300">
                <CheckCircle2 className="h-4 w-4 shrink-0 text-blue-400" />
                {bannerMsg}
              </div>
            )}

            {/* ── AI SETTINGS PANEL ── */}
            {activeNav === "settings" && (
              <section className="rounded-2xl border border-white/10 bg-[#161B27] p-6">
                <div className="flex items-center gap-2 mb-5">
                  <Settings2 className="h-4 w-4 text-blue-400" />
                  <h2 className="text-sm font-bold text-white">AI Model Settings</h2>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  {/* Provider */}
                  <div>
                    <label className="mb-1.5 block text-xs font-semibold text-slate-400">Provider</label>
                    <div className="relative">
                      <select
                        aria-label="Provider"
                        disabled={testingConnection || runningPipeline || compilingRubric || synthLoading}
                        value={llmProvider}
                        onChange={(e) => {
                          const provider = e.target.value as LLMProvider;
                          setLlmProvider(provider);
                          setLlmModel(PROVIDER_MODELS[provider][0]);
                          setShowApiKey(false);
                        }}
                        className="w-full appearance-none rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none cursor-pointer"
                      >
                        {LLM_PROVIDERS.map((p) => <option key={p} value={p} className="bg-[#1e2535]">{p}</option>)}
                      </select>
                      <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
                    </div>
                  </div>

                  {/* Model */}
                  <div>
                    <label className="mb-1.5 block text-xs font-semibold text-slate-400">Model</label>
                    <div className="relative">
                      <select
                        aria-label="Model"
                        disabled={testingConnection || runningPipeline || compilingRubric || synthLoading}
                        value={llmModel}
                        onChange={(e) => setLlmModel(e.target.value)}
                        className="w-full appearance-none rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none cursor-pointer"
                      >
                        {PROVIDER_MODELS[llmProvider].map((m) => <option key={m} value={m} className="bg-[#1e2535]">{m}</option>)}
                      </select>
                      <ChevronDown className="pointer-events-none absolute right-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
                    </div>
                  </div>

                  {/* API Key */}
                  <div>
                    <label className="mb-1.5 block text-xs font-semibold text-slate-400">
                      API key
                    </label>
                    <div className="relative">
                      <input
                        type={showApiKey ? "text" : "password"}
                        value={apiKey}
                        onChange={(e) => setApiKeys({ ...apiKeys, [llmProvider]: e.target.value })}
                        placeholder="Enter a provider key"
                        aria-label="Provider API key"
                        disabled={testingConnection || runningPipeline || compilingRubric || synthLoading}
                        autoComplete="off"
                        className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 pr-9 text-xs text-white placeholder-slate-600 focus:border-blue-500 focus:outline-none font-mono"
                      />
                      <button
                        type="button"
                        aria-label={showApiKey ? "Hide API key" : "Show API key"}
                        onClick={() => setShowApiKey((v) => !v)}
                        className="absolute right-2.5 top-2 text-slate-500 hover:text-slate-300 transition"
                      >
                        {showApiKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                  </div>
                </div>

                {betaAccessRequired && (
                  <label className="mt-4 block text-xs font-medium text-slate-300">
                    Studio beta access code
                    <input
                      type="password"
                      autoComplete="off"
                      value={betaCode}
                      disabled={testingConnection || runningPipeline || compilingRubric || synthLoading}
                      onChange={(event) => {
                        setBetaCodeInput(event.target.value);
                        setConnectedModel(undefined);
                        setConnectionError(undefined);
                      }}
                      className="mt-2 block w-full max-w-sm rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white"
                    />
                  </label>
                )}
                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <button type="button" disabled={!browserModel || (betaAccessRequired && !betaCode.trim()) || testingConnection || runningPipeline || compilingRubric || synthLoading}
                    onClick={() => void handleTestConnection()}
                    className="rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold hover:bg-blue-500 disabled:opacity-40">
                    {testingConnection ? "Testing…" : "Test connection"}
                  </button>
                  {connected && <span role="status" className="text-xs text-emerald-300">Model connected</span>}
                  {connected && <button type="button" onClick={() => setActiveNav(rubricId ? "studio" : "training")} className="text-xs font-semibold text-blue-300">Continue</button>}
                  {connectionError && !connected && <span role="alert" className="text-xs text-rose-300">{connectionError}</span>}
                </div>
                {/* Status line */}
                <div className="mt-4 flex items-center gap-2 rounded-lg border border-white/8 bg-white/3 px-3 py-2 text-[11px] text-slate-500">
                  <Zap className="h-3 w-3 text-blue-500 shrink-0" />
                  <span>Your key stays in this tab’s memory until reload. The browser sends it directly to the selected provider. The Studio backend receives prompts and results for the review workflow.</span>
                </div>
              </section>
            )}

            {activeNav === "studio" && (!connected || !rubricId || !matrix?.papers.length) && (
              <section className="rounded-2xl border border-white/10 bg-[#161B27] p-5">
                <h2 className="text-base font-semibold">Start a review</h2>
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  <button type="button" onClick={() => setActiveNav("settings")} className="rounded-lg border border-white/15 px-3 py-2 hover:bg-white/10">{connected ? "✓ Model connected" : "Connect model"}</button>
                  <button type="button" onClick={() => setActiveNav("training")} className="rounded-lg border border-white/15 px-3 py-2 hover:bg-white/10">{rubricId && matrix ? "✓ Rubric active" : "Choose or upload rubric"}</button>
                  <button type="button" disabled={!connected || !rubricId || !matrix || uploadingFiles.length > 0 || runningPipeline} onClick={() => fileInputRef.current?.click()} className="rounded-lg border border-white/15 px-3 py-2 hover:bg-white/10 disabled:opacity-40">Add PDFs</button>
                </div>
              </section>
            )}

            {pausedRun && activeNav === "studio" && (
              <section className="flex flex-wrap items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm">
                <span role="status">Extraction paused. {pausedRun.completed} of {pausedRun.papers.length} papers complete.</span>
                <button type="button"
                  disabled={runningPipeline || !connected || pausedRun.rubricId !== rubricId ||
                    pausedRun.model.provider !== llmProvider || pausedRun.model.model !== llmModel}
                  onClick={() => void handleRunExtractionPipeline(true)}
                  className="rounded-lg border border-white/15 px-3 py-2 text-xs font-semibold disabled:opacity-40">
                  Resume from interrupted paper
                </button>
              </section>
            )}

            {guidedRun && (
              <section className="rounded-2xl border border-white/10 bg-[#161B27] p-5 text-sm text-slate-300">
                Zero-shot and few-shot comparisons are available in the terminal benchmark. Use Extraction for a browser review.
              </section>
            )}

            {activeNav === "studio" && (
              <section className="flex flex-col rounded-2xl border border-white/10 bg-[#161B27] p-5 sm:p-6">
                <div className="flex items-center justify-between gap-2 mb-4">
                  <div>
                    <h2 className="text-sm font-bold text-white">Upload and extract evidence</h2>
                    <p className="mt-0.5 text-xs text-slate-500">Upload PDFs and choose which papers to extract.</p>
                  </div>
                  <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-[11px] text-slate-400 tabular-nums">
                    {matrix?.criteria.length || 0} criteria
                  </span>
                </div>

                {/* Dropzone */}
                <button
                  type="button"
                  disabled={!connected || !rubricId || !matrix || runningPipeline || uploadingFiles.length > 0}
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setDragging(false); if (connected && rubricId && matrix && !runningPipeline && uploadingFiles.length === 0) void handleFilesUpload(e.dataTransfer.files); }}
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex w-full cursor-pointer items-center justify-between gap-4 rounded-xl border-2 border-dashed text-left disabled:opacity-50 px-5 py-4 transition ${
                    dragging ? "border-blue-500 bg-blue-500/10" : "border-white/10 bg-white/3 hover:border-blue-500/50 hover:bg-blue-500/5"
                  }`}
                >
                  <div className="flex items-center gap-3.5">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-600/20">
                      <Upload className="h-5 w-5 text-blue-400" />
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-white">Drop PDF(s) here or click to browse</p>
                      <p className="text-[11px] text-slate-500">Finds text, tables, and page coordinates</p>
                    </div>
                  </div>
                  <span className="shrink-0 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-semibold text-slate-300">Browse</span>
                </button>

                {uploadingFiles.length > 0 && (
                  <div className="mt-3 space-y-1 rounded-lg border border-blue-500/30 bg-blue-500/10 px-3.5 py-2.5">
                    {uploadingFiles.map((name) => (
                      <div key={name} className="flex items-center gap-2 text-xs text-blue-300">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        Ingesting {name}…
                      </div>
                    ))}
                  </div>
                )}

                {/* Paper selection */}
                <div className="mt-4 flex-1">
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-300 tabular-nums">
                      {selectedPaperIds.length} of {(matrix?.papers || []).length} papers selected
                    </span>
                    {(matrix?.papers.length || 0) > 0 && (
                      <button
                        type="button"
                        onClick={() => setSelectedPaperIds(
                          selectedPaperIds.length === (matrix?.papers.length || 0)
                            ? [] : (matrix?.papers || []).map((p) => p.paper_id)
                        )}
                        className="text-[11px] font-semibold text-blue-400 hover:text-blue-300"
                      >
                        {selectedPaperIds.length === (matrix?.papers.length || 0) ? "Deselect all" : "Select all"}
                      </button>
                    )}
                  </div>
                  <div className="max-h-28 space-y-1 overflow-y-auto rounded-lg border border-white/8 bg-black/20 p-2">
                    {(matrix?.papers || []).map((p) => (
                      <label key={p.paper_id} className="flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-1.5 text-xs transition hover:bg-white/5">
                        <input
                          type="checkbox"
                          checked={selectedPaperIds.includes(p.paper_id)}
                          onChange={() => togglePaperSelection(p.paper_id)}
                          className="h-3.5 w-3.5 rounded border-slate-600 bg-transparent text-blue-500 focus:ring-blue-500"
                        />
                        <span className="truncate font-medium text-slate-300">{p.title}</span>
                        <span className="ml-auto font-mono text-[10px] text-slate-600">{p.paper_id.slice(0, 8)}</span>
                      </label>
                    ))}
                    {(!matrix || matrix.papers.length === 0) && (
                      <p className="py-3 text-center text-xs text-slate-600">No papers yet. Drop a PDF above to begin.</p>
                    )}
                  </div>
                </div>

                {/* Run footer */}
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-4">
                  <div className="text-xs text-slate-500">
                    {runningPipeline ? (
                      <span role="status" className="flex items-center gap-1.5 font-medium text-blue-400">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        {pipelineStage}
                      </span>
                    ) : prismaCounts ? (
                      <span className="font-mono text-[11px] text-slate-400 tabular-nums">
                        PRISMA: {Object.entries(prismaCounts).filter(([, value]) => typeof value === "number").map(([k, v]) => `${k}=${v}`).join(" · ")}
                      </span>
                    ) : (
                      <span>PRISMA Screening → Extractor → Critic</span>
                    )}
                  </div>
                  <button
                    type="button"
                    disabled={!connected || !rubricId || loading || compilingRubric || runningPipeline || uploadingFiles.length > 0 || selectedPaperIds.length === 0}
                    onClick={() => void handleRunExtractionPipeline()}
                    className="flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2 text-xs font-semibold text-white transition hover:bg-blue-500 disabled:opacity-40"
                  >
                    {runningPipeline ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-3.5 w-3.5 fill-current" />}
                    {runningPipeline ? "Running…" : `Run Extraction (${selectedPaperIds.length})`}
                  </button>
                </div>
              </section>
            )}

            {activeNav === "training" && (
              <section className="rounded-2xl border border-white/10 bg-[#161B27] p-6 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-bold text-white">Revise rubric prompts</h3>
                    <p className="text-xs text-slate-500">Start with the example rubric below or upload your own YAML. Review the criteria, then activate it.</p>
                  </div>
                  {rubricId && matrix && <button type="button" onClick={() => setActiveNav("studio")} className="text-xs font-semibold text-blue-300">Continue to extraction</button>}
                  <button
                    type="button"
                    disabled={!connected || compilingRubric || runningPipeline || uploadingFiles.length > 0}
                    onClick={() => void handleCreateRubric()}
                    className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-500 disabled:opacity-50 transition"
                  >
                    {compilingRubric ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FilePlus2 className="h-3.5 w-3.5" />}
                    Compile & Activate
                  </button>
                </div>
                <label className="block text-xs font-medium text-slate-300">
                  Upload rubric (.yml or .yaml)
                  <input
                    type="file"
                    accept=".yml,.yaml"
                    disabled={compilingRubric}
                    className="mt-2 block w-full text-xs text-slate-300"
                    onChange={async (event) => {
                      const file = event.currentTarget.files?.[0];
                      event.currentTarget.value = "";
                      if (!file) return;
                      if (!/\.ya?ml$/i.test(file.name) || file.size > 256 * 1024) {
                        setError("Choose a .yml or .yaml file no larger than 256 KB.");
                        return;
                      }
                      try {
                        setYamlText(await file.text());
                        setRubricResult(null);
                        setError(null);
                        setBannerMsg(`Loaded ${file.name}. Review the draft, then choose Compile & Activate.`);
                      } catch {
                        setError("Could not read the rubric file.");
                      }
                    }}
                  />
                </label>
                <textarea
                  aria-label="YAML rubric draft"
                  rows={14}
                  value={yamlText}
                  onChange={(e) => setYamlText(e.target.value)}
                  className="w-full rounded-xl border border-white/10 bg-black/40 p-4 font-mono text-xs text-slate-300 focus:border-blue-500 focus:outline-none resize-none"
                />
                {rubricResult && rubricResult.lint.ambiguous.length > 0 && (
                  <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 space-y-2">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-300">
                      <AlertTriangle className="h-4 w-4" />
                      Rubric Ambiguity Warnings
                    </div>
                    {rubricResult.lint.ambiguous.map((issue) => (
                      <div key={issue.criterion_id} className="rounded-lg border border-white/8 bg-white/3 p-3">
                        <span className="font-mono font-semibold text-amber-300 text-xs">{issue.criterion_id}: </span>
                        <span className="text-xs text-slate-300">{issue.problem}</span>
                        <div className="mt-1 text-[11px] font-medium text-emerald-400">↳ {issue.suggested_rewrite}</div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* ── SYNTHESIS OUTPUT ── */}
            {activeNav === "matrix" && synthesis && (
              <section className="rounded-2xl border border-blue-500/30 bg-[#161B27] p-6">
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-blue-400" />
                    <h3 className="text-sm font-bold text-white">Grounded Cross-Paper Synthesis</h3>
                  </div>
                  <button type="button" onClick={() => setSynthesis(null)} className="text-xs text-slate-500 hover:text-slate-300">Dismiss</button>
                </div>
                <p className="whitespace-pre-line text-xs leading-relaxed text-slate-300">{synthesis.narrative}</p>
                {synthesis.discrepancies.length > 0 && (
                  <div className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
                    <p className="text-xs font-semibold text-amber-300">Cross-Paper Discrepancies</p>
                    <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-amber-200">
                      {synthesis.discrepancies.map((d, i) => <li key={i}>{d}</li>)}
                    </ul>
                  </div>
                )}
              </section>
            )}

            {/* ── EVIDENCE MATRIX ── */}
            {activeNav === "matrix" && <section className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-white/10 bg-[#161B27] px-5 py-3.5">
                <div className="flex flex-wrap items-center gap-4 text-xs tabular-nums">
                  <span className="text-sm font-bold text-white">Evidence Matrix</span>
                  <span className="text-slate-500">{matrix?.papers.length || 0} papers × {matrix?.criteria.length || 0} criteria</span>
                  <span className="flex items-center gap-1 font-medium text-emerald-400">
                    <CheckCircle2 className="h-3.5 w-3.5" />{stats.verified} verified
                  </span>
                  {stats.flagged > 0 && (
                    <span className="flex items-center gap-1 font-medium text-amber-400">
                      <AlertTriangle className="h-3.5 w-3.5" />{stats.flagged} flagged
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-600" />
                    <input
                      type="text"
                      placeholder="Filter papers…"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="rounded-lg border border-white/10 bg-white/5 py-1.5 pl-8 pr-3 text-xs text-slate-300 placeholder-slate-600 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <label className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-400">
                    <input type="checkbox" checked={onlyFlagged} onChange={(e) => setOnlyFlagged(e.target.checked)} className="h-3.5 w-3.5 rounded border-slate-600 bg-transparent text-blue-500" />
                    Flagged only
                  </label>
                </div>
              </div>

              <div className="overflow-x-auto rounded-2xl border border-white/10 bg-[#161B27]">
                <table className="w-full border-collapse text-left text-xs">
                  <thead>
                    <tr className="border-b border-white/8">
                      <th className="sticky left-0 z-20 min-w-[260px] border-r border-white/8 bg-[#161B27] px-4 py-3 font-semibold text-slate-400">
                        Paper / Study
                      </th>
                      {(matrix?.criteria || []).map((cid) => (
                        <th key={cid} className="min-w-[180px] border-r border-white/8 px-3 py-3 font-mono text-[11px] font-semibold text-slate-400">
                          {cid}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {filteredPapers.map((paper) => (
                      <tr key={paper.paper_id} className="transition-colors hover:bg-white/3">
                        <td className="sticky left-0 z-10 border-r border-white/8 bg-[#161B27] px-4 py-3 align-top">
                          <div className="line-clamp-2 font-medium text-slate-200">{paper.title}</div>
                          <div className="mt-1 font-mono text-[10px] text-slate-600">{paper.paper_id.slice(0, 8)}</div>
                        </td>
                        {(matrix?.criteria || []).map((cid) => {
                          const cell = cellLookup.get(`${paper.paper_id}::${cid}`);
                          if (!cell) return <td key={cid} className="border-r border-white/5 px-3 py-3 font-mono text-slate-700">—</td>;
                          return (
                            <td
                              key={cid}
                              onClick={() => setSelectedCell(cell)}
                              className="group cursor-pointer border-r border-white/5 px-3 py-2.5 align-top transition-colors hover:bg-blue-500/8"
                            >
                              <div className="flex items-start justify-between gap-1.5">
                                <span className="line-clamp-2 font-medium text-slate-300">{renderCellValue(cell.value)}</span>
                                <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-slate-700 opacity-0 transition group-hover:text-blue-400 group-hover:opacity-100" />
                              </div>
                              <div className="mt-1.5 flex items-center justify-between gap-1">
                                <span className={`inline-block rounded-md border px-1.5 py-0.5 text-[10px] font-semibold ${statusBadge(cell.critic_status)}`}>
                                  {cell.critic_status}
                                </span>
                                {cell.page_number && (
                                  <span className="font-mono text-[10px] text-slate-600 tabular-nums">p.{cell.page_number}</span>
                                )}
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                    {!loading && filteredPapers.length === 0 && (
                      <tr>
                        <td colSpan={(matrix?.criteria.length || 0) + 1} className="px-6 py-16 text-center text-sm text-slate-600">
                          Upload a PDF and click <span className="font-semibold text-slate-400">Run Extraction</span> to populate the matrix.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>}
          </div>
        </main>
      </div>

      {/* ── INSPECTOR DRAWER ── */}
      <InspectorDrawer
        cell={selectedCell}
        paper={selectedCell ? paperMap.get(selectedCell.paper_id) : undefined}
        onClose={() => setSelectedCell(null)}
        onExportPdf={handleExportPdf}
        onApprove={handleApproveOne}
        onOverride={handleOverride}
      />
    </div>
  );
}

export default App;
