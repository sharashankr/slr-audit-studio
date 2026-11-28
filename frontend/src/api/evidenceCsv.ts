import type { MatrixResponse } from "../types/api";

function csvField(value: unknown): string {
  if (value == null) return '""';
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  // Treat imported text as data when the CSV is opened in a spreadsheet.
  const safe = typeof value === "string" && /^[\s\uFEFF]*[=+\-@]|^[\t\r\n]/u.test(text)
    ? `'${text}`
    : text;
  return `"${safe.replaceAll('"', '""')}"`;
}

export function buildEvidenceCsv(matrix: MatrixResponse): string {
  const papers = new Map(matrix.papers.map((paper) => [paper.paper_id, paper]));
  const rows: unknown[][] = [[
    "Paper ID", "Paper title", "DOI", "Year", "Rubric", "Criterion", "Extracted value",
    "Source quote", "PDF page", "Review status", "Confidence", "Review notes",
  ]];
  for (const cell of matrix.cells) {
    const paper = papers.get(cell.paper_id);
    if (!paper) continue;
    rows.push([
      paper.paper_id, paper.title, paper.doi, paper.year, matrix.rubric_name, cell.criterion_id,
      cell.value, cell.verbatim_quote, cell.page_number, cell.critic_status, cell.confidence,
      cell.critic_notes,
    ]);
  }
  return rows.map((row) => row.map(csvField).join(",")).join("\r\n") + "\r\n";
}
