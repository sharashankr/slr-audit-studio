import { useEffect, useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  X,
  Quote,
  Edit3,
  ShieldCheck,
} from "lucide-react";
import type { MatrixCell, PaperSummary } from "../types/api";
import { resolvePdfUrl } from "../api/client";
import { PdfProvenanceViewer } from "./PdfProvenanceViewer";

interface Props {
  cell: MatrixCell | null;
  paper: PaperSummary | undefined;
  onClose: () => void;
  onApprove: (cellId: number) => Promise<void>;
  onOverride: (cellId: number, value: unknown, reason: string) => Promise<void>;
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) return "";
  if (typeof val === "string") return val;
  return JSON.stringify(val);
}

export function InspectorDrawer({
  cell,
  paper,
  onClose,
  onApprove,
  onOverride,
}: Props) {
  const [editValue, setEditValue] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cell) {
      setEditValue(formatValue(cell.value));
      setReason("");
      setError(null);
    }
  }, [cell]);

  if (!cell) return null;

  const parseEditedValue = (raw: string): unknown => {
    const trimmed = raw.trim();
    if (trimmed === "") return null;
    if (trimmed === "true") return true;
    if (trimmed === "false") return false;
    if (!Number.isNaN(Number(trimmed)) && trimmed !== "") return Number(trimmed);
    try {
      return JSON.parse(trimmed);
    } catch {
      return raw;
    }
  };

  const handleApprove = async () => {
    setBusy(true);
    setError(null);
    try {
      await onApprove(cell.cell_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleOverride = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reason.trim()) {
      setError("A rationale is required for the immutable audit log.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onOverride(cell.cell_id, parseEditedValue(editValue), reason.trim());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const isVerified =
    cell.critic_status === "verified" ||
    cell.critic_status === "approved" ||
    cell.critic_status === "supported";

  return (
    <div className="fixed inset-0 z-50 flex bg-[#0F172A]/50 backdrop-blur-xs">
      <div className="ml-auto flex h-full w-full max-w-6xl bg-white shadow-2xl">
        {/* Left Pane: Evidence, Critic Notes & HITL Controls */}
        <div className="flex w-[420px] shrink-0 flex-col border-r border-[#E2E8F0] bg-[#F8FAFC]">
          <div className="flex items-start justify-between border-b border-[#E2E8F0] bg-white px-5 py-4">
            <div>
              <span className="font-mono text-xs font-semibold text-[#2563EB]">
                {cell.criterion_id}
              </span>
              <h2 className="font-display mt-0.5 line-clamp-2 text-sm font-bold text-[#0F172A]">
                {paper?.title || cell.paper_id}
              </h2>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg p-1.5 text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A]"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="flex-1 space-y-5 overflow-y-auto p-5">
            {/* Status & Confidence */}
            <div className="flex items-center justify-between rounded-xl border border-[#E2E8F0] bg-white p-3">
              <div className="flex items-center gap-2">
                {isVerified ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                )}
                <span className="text-xs font-semibold text-[#0F172A] uppercase">
                  {cell.critic_status}
                </span>
              </div>
              <span className="font-mono text-xs text-[#64748B] tabular-nums">
                {(cell.confidence * 100).toFixed(0)}% · #{cell.cell_id}
              </span>
            </div>

            {/* Verbatim Quote */}
            <div>
              <div className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold text-[#334155]">
                <Quote className="h-3.5 w-3.5 text-[#2563EB]" />
                <span>
                  Verbatim Provenance{" "}
                  {cell.page_number ? `(Page ${cell.page_number})` : ""}
                </span>
              </div>
              <blockquote className="rounded-xl border-l-4 border-[#2563EB] bg-[#EFF6FF]/60 p-3.5 text-xs leading-relaxed text-[#1E293B]">
                {cell.verbatim_quote || "No verbatim quote recorded (not reported)."}
              </blockquote>
            </div>

            {/* Critic Notes */}
            {cell.critic_notes && (
              <div>
                <div className="mb-1.5 text-xs font-semibold text-[#334155]">
                  Critic Verification Notes
                </div>
                <div className="rounded-xl border border-[#E2E8F0] bg-white p-3.5 text-xs leading-relaxed text-[#475569]">
                  {cell.critic_notes}
                </div>
              </div>
            )}

            {/* Quick Approve */}
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3.5">
              <button
                type="button"
                disabled={busy}
                onClick={handleApprove}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700 disabled:opacity-50"
              >
                <ShieldCheck className="h-4 w-4" />
                <span>Approve Extracted Value As-Is</span>
              </button>
            </div>

            {/* Override Form */}
            <form
              onSubmit={handleOverride}
              className="space-y-3 rounded-xl border border-[#E2E8F0] bg-white p-4"
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-[#0F172A]">
                <Edit3 className="h-3.5 w-3.5 text-[#2563EB]" />
                <span>Human Override (Audit Trail)</span>
              </div>

              <div>
                <label className="mb-1 block text-[11px] font-medium text-[#475569]">
                  Corrected Value
                </label>
                <input
                  type="text"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  className="w-full rounded-lg border border-[#CBD5E1] bg-[#F8FAFC] px-2.5 py-1.5 font-mono text-xs text-[#0F172A] focus:border-[#2563EB] focus:bg-white focus:outline-hidden"
                />
              </div>

              <div>
                <label className="mb-1 block text-[11px] font-medium text-[#475569]">
                  Audit Rationale (Required)
                </label>
                <textarea
                  rows={2}
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="e.g., Table 2 reports ITT cohort N=412..."
                  className="w-full rounded-lg border border-[#CBD5E1] bg-[#F8FAFC] px-2.5 py-1.5 text-xs text-[#0F172A] focus:border-[#2563EB] focus:bg-white focus:outline-hidden"
                />
              </div>

              {error && (
                <p className="rounded-lg bg-rose-50 p-2 text-xs text-rose-700">
                  {error}
                </p>
              )}

              <button
                type="submit"
                disabled={busy}
                className="w-full rounded-lg bg-[#2563EB] px-3 py-2 text-xs font-semibold text-white transition hover:bg-[#1D4ED8] disabled:opacity-50"
              >
                Save Override & Log Audit Event
              </button>
            </form>
          </div>
        </div>

        {/* Right Pane: Embedded PDF Viewer with BBox Highlight */}
        <div className="flex-1 overflow-hidden">
          <PdfProvenanceViewer
            pdfUrl={paper ? resolvePdfUrl(paper.pdf_url) : ""}
            targetPage={cell.page_number}
            bboxes={cell.bbox}
          />
        </div>
      </div>
    </div>
  );
}
