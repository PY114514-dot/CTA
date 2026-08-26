/**
 * Chart-extraction shared helpers: editable curve rows, series aggregation,
 * CSV export and Y-tick parsing. Extracted from the original single-file
 * ChartExtractDrawer.tsx (image / PDF intake flows).
 */

import type { ExtractTracedPoint } from "../../api";

export const IMAGE_ACCEPT = "image/png,image/jpeg,image/webp";
export const PDF_ACCEPT = "application/pdf";

// ---------------------------------------------------------------------------
// Local types & helpers
// ---------------------------------------------------------------------------

/** One editable curve row in the configuration step. */
export interface CurveRow {
  id: string;
  name: string;
  colorHex: string;
  isBenchmark: boolean;
}

/** Aggregated (date → value) point for tables and export. */
export interface SeriesPoint {
  date: string;
  value: number;
}

let curveIdCounter = 0;
export function nextCurveId(): string {
  curveIdCounter += 1;
  return `curve-${curveIdCounter}`;
}

/** Collapse per-pixel points into one mean value per date, sorted by date. */
export function toSeries(points: ExtractTracedPoint[]): SeriesPoint[] {
  const byDate = new Map<string, number[]>();
  for (const p of points) {
    if (p.value === null || p.value === undefined || !p.date) continue;
    const bucket = byDate.get(p.date);
    if (bucket) bucket.push(p.value);
    else byDate.set(p.date, [p.value]);
  }
  return [...byDate.entries()]
    .map(([date, values]) => ({
      date,
      value: values.reduce((a, b) => a + b, 0) / values.length,
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

export function formatValue(v: number): string {
  return v.toFixed(4);
}

/** Build CSV text from one or more named series (date column + value columns). */
export function buildCsv(seriesByName: Array<{ name: string; series: SeriesPoint[] }>): string {
  if (seriesByName.length === 0) return "";
  const dates = [...new Set(seriesByName.flatMap((s) => s.series.map((p) => p.date)))].sort();
  const header = ["date", ...seriesByName.map((s) => s.name)].join(",");
  const rows = dates.map((date) => {
    const cells = seriesByName.map((s) => {
      const point = s.series.find((p) => p.date === date);
      return point ? point.value.toFixed(6) : "";
    });
    return [date, ...cells].join(",");
  });
  return [header, ...rows].join("\n");
}

export function downloadTextFile(content: string, filename: string, mime: string): void {
  const blob = new Blob(["\uFEFF" + content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * Parse the Y-axis tick input. Accepts either an explicit bottom→top list
 * ("0.995, 1.005, 1.015, …") or a compact "min, max, step" triple
 * ("0.995, 1.075, 0.01") which is expanded into the full evenly-spaced list.
 * The triple is only treated as a range when it yields ≥3 ticks, so a genuine
 * 3-tick axis is never misread as a range.
 */
export function parseYTicksInput(text: string): string[] {
  const tokens = text.split(/[,，\s]+/).map((s) => s.trim()).filter(Boolean);
  if (tokens.length === 3) {
    const min = Number(tokens[0]);
    const max = Number(tokens[1]);
    const step = Number(tokens[2]);
    if (Number.isFinite(min) && Number.isFinite(max) && Number.isFinite(step) && step > 0 && max > min) {
      const steps = Math.round((max - min) / step);
      if (steps >= 2) {
        const decimals = Math.max(...tokens.map((t) => (t.split(".")[1] || "").length));
        const ticks: string[] = [];
        for (let i = 0; i <= steps; i++) {
          ticks.push((min + i * step).toFixed(decimals));
        }
        return ticks;
      }
    }
  }
  return tokens;
}
