import { useEffect, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, FileWarning } from "lucide-react";
import type { BBox } from "../types/api";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

interface Props {
  pdfUrl: string;
  targetPage: number | null;
  bboxes: BBox[] | BBox | null;
}

export function PdfProvenanceViewer({ pdfUrl, targetPage, bboxes }: Props) {
  const [numPages, setNumPages] = useState<number>(0);
  const [pageNumber, setPageNumber] = useState<number>(targetPage || 1);
  const [scale, setScale] = useState<number>(1.05);

  useEffect(() => {
    if (targetPage && targetPage >= 1) {
      setPageNumber(targetPage);
    } else {
      setPageNumber(1);
    }
  }, [targetPage, pdfUrl]);

  const normalizedBoxes: BBox[] = Array.isArray(bboxes)
    ? bboxes
    : bboxes
      ? [bboxes]
      : [];

  const pageBoxes = normalizedBoxes.filter(
    (b) => b.page === pageNumber && b.w > 0 && b.h > 0
  );

  if (!pdfUrl) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-slate-400">
        <FileWarning className="h-8 w-8" />
        <p className="text-sm">No PDF URL attached to this paper.</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-slate-900">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-slate-700 bg-slate-800 px-3 py-2 text-xs text-slate-200">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            disabled={pageNumber <= 1}
            onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
            className="rounded p-1 hover:bg-slate-700 disabled:opacity-40"
            title="Previous page"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="font-mono">
            Page {pageNumber} / {numPages || "?"}
          </span>
          <button
            type="button"
            disabled={numPages > 0 && pageNumber >= numPages}
            onClick={() => setPageNumber((p) => Math.min(numPages, p + 1))}
            className="rounded p-1 hover:bg-slate-700 disabled:opacity-40"
            title="Next page"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
          {targetPage && pageNumber !== targetPage && (
            <button
              type="button"
              onClick={() => setPageNumber(targetPage)}
              className="ml-2 rounded bg-amber-500/20 px-2 py-0.5 text-amber-300 hover:bg-amber-500/30"
            >
              Jump to cited p.{targetPage}
            </button>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setScale((s) => Math.max(0.6, +(s - 0.15).toFixed(2)))}
            className="rounded p-1 hover:bg-slate-700"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="w-12 text-center font-mono">{Math.round(scale * 100)}%</span>
          <button
            type="button"
            onClick={() => setScale((s) => Math.min(2.2, +(s + 0.15).toFixed(2)))}
            className="rounded p-1 hover:bg-slate-700"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Scrollable Page Canvas + Docling Top-Left BBox Overlay */}
      <div className="flex-1 overflow-auto p-4">
        <div className="mx-auto w-fit shadow-2xl">
          <Document
            file={pdfUrl}
            onLoadSuccess={({ numPages: total }) => setNumPages(total)}
            loading={
              <div className="p-12 text-sm text-slate-400">Loading PDF document...</div>
            }
            error={
              <div className="p-8 text-sm text-rose-400">
                Failed to load PDF from <code>{pdfUrl}</code>. Confirm FastAPI is running.
              </div>
            }
          >
            <div className="relative inline-block">
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={true}
                renderAnnotationLayer={false}
              />
              {/* Bounding-box provenance overlay */}
              <div className="pointer-events-none absolute inset-0 z-10">
                {pageBoxes.map((box, idx) => {
                  const leftPct = (box.l / box.w) * 100;
                  const topPct = (box.t / box.h) * 100;
                  const widthPct = ((box.r - box.l) / box.w) * 100;
                  const heightPct = ((box.b - box.t) / box.h) * 100;
                  return (
                    <div
                      key={idx}
                      style={{
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                      }}
                      className="absolute rounded-xs border-2 border-amber-500 bg-amber-300/30 shadow-sm transition-all"
                    />
                  );
                })}
              </div>
            </div>
          </Document>
        </div>
      </div>
    </div>
  );
}
