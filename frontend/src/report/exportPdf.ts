/**
 * PDF export through the backend renderer.
 *
 * Extracted from main.tsx for maintainability.
 */
import { generateMarkdownReport } from "./markdownReport";
import type { CtaAttributionSnapshotResponse } from "../api";
import type { HistoryRecord } from "../storage/historyStorage";
import { downloadAnalysisReportPdf } from "../api";

/** Download a real PDF without relying on popup and print-dialog support. */
export async function exportPdfReport(record: HistoryRecord, attributionSnapshot?: CtaAttributionSnapshotResponse): Promise<void> {
  const md = generateMarkdownReport(record, attributionSnapshot);
  const safeName = (record.product_name || "产品").replace(/[\\/:*?"<>|]/g, "_");
  const blob = await downloadAnalysisReportPdf(record.product_name, md);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${safeName}_报告.pdf`;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

/** @deprecated Kept only so existing imports do not break during migration. */
export function exportPdfReportWithPrintDialog(record: HistoryRecord): void {
  const md = generateMarkdownReport(record);
  const escapeHtml = (value: string): string => value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
  // Convert markdown to HTML. Consecutive table rows must be grouped into a
  // real <table>: browsers strip bare <tr> elements outside a table, which
  // used to glue every cell into one run of text in the printed PDF.
  const rows: string[] = [];
  let inTable = false;
  let firstRowOfTable = false;
  for (const line of md.split("\n")) {
    const isTableRow = line.startsWith("| ");
    const isSeparatorRow =
      isTableRow &&
      line
        .split("|")
        .filter((cell) => cell.trim() !== "")
        .every((cell) => /^:?-+:?$/.test(cell.trim()));
    if (isSeparatorRow) continue;
    if (isTableRow) {
      if (!inTable) {
        rows.push("<table>");
        inTable = true;
        firstRowOfTable = true;
      }
      const cells = line.split("|").filter(Boolean).map((cell) => cell.trim());
      const tag = firstRowOfTable ? "th" : "td";
      rows.push(`<tr>${cells.map((cell) => `<${tag}>${escapeHtml(cell)}</${tag}>`).join("")}</tr>`);
      firstRowOfTable = false;
      continue;
    }
    if (inTable) {
      rows.push("</table>");
      inTable = false;
    }
    if (line.startsWith("## ")) {
      rows.push(`<h2>${escapeHtml(line.slice(3))}</h2>`);
    } else if (line.startsWith("### ")) {
      rows.push(`<h3>${escapeHtml(line.slice(4))}</h3>`);
    } else if (line.startsWith("**")) {
      rows.push(`<p><strong>${escapeHtml(line.replace(/\*\*/g, ""))}</strong></p>`);
    } else if (line.startsWith("---")) {
      rows.push("<hr/>");
    } else if (line.startsWith("*")) {
      rows.push(`<p><em>${escapeHtml(line.replace(/^\*|\*$/g, ""))}</em></p>`);
    } else if (line.trim() !== "") {
      rows.push(`<p>${escapeHtml(line)}</p>`);
    }
  }
  if (inTable) rows.push("</table>");
  const htmlRows = rows.join("\n");

  const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>${record.product_name || "产品"}_分析报告</title>
<style>
  @page { size: A4; margin: 16mm 14mm; }
  * { box-sizing: border-box; }
  body { font-family: "Microsoft YaHei", "SimHei", sans-serif; max-width: 182mm; margin: 0 auto; color: #243b53; line-height: 1.6; }
  header { border-bottom: 2px solid #102a43; margin-bottom: 16px; padding-bottom: 8px; color: #102a43; font-size: 12px; }
  h2 { color: #102a43; border-bottom: 1px solid #9fb3c8; padding-bottom: 6px; margin: 24px 0 12px; break-after: avoid; }
  h3 { color: #243b53; margin: 20px 0 8px; break-after: avoid; }
  p { margin: 7px 0; overflow-wrap: anywhere; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10px; break-inside: auto; }
  thead { display: table-header-group; }
  tr { break-inside: avoid; page-break-inside: avoid; }
  th, td { border: 1px solid #cbd5e1; padding: 5px 7px; text-align: left; vertical-align: top; }
  th { background: #eaf2fb; color: #102a43; }
  hr { border: none; border-top: 1px solid #d9e2ec; margin: 22px 0; }
  @media print { body { max-width: none; } }
</style></head><body><header>${escapeHtml(record.product_name || "未命名产品")} · 产品分析报告</header>${htmlRows}</body></html>`;

  const printWindow = window.open("", "_blank");
  if (!printWindow) return;
  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();
  let printed = false;
  const printWhenReady = (): void => {
    if (printed) return;
    printed = true;
    const fontsReady = printWindow.document.fonts?.ready ?? Promise.resolve();
    void fontsReady.finally(() => {
      setTimeout(() => {
        printWindow.focus();
        printWindow.print();
      }, 150);
    });
  };
  // `document.close()` is not enough on Chromium: printing immediately can
  // capture an unstyled or partially laid-out report.
  printWindow.addEventListener("load", printWhenReady, { once: true });
  setTimeout(printWhenReady, 800);
}
