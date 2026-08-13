/**
 * Small formatting and validation utilities shared across the app.
 *
 * Extracted from main.tsx to reduce duplication and enable testing.
 */

/** Formats a decimal return (e.g. 0.1234) as a percentage string. */
export function percentage(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

/** Formats a nullable decimal metric with two fixed places. */
export function decimal(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toFixed(2);
}

/** Validates a strict `YYYY-MM-DD` string that is also a real calendar date. */
export function isIsoDate(value: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(new Date(`${value}T00:00:00`).getTime());
}

/** Returns true when `start` is not after `end` (inclusive). */
export function isValidDateRange(start: string, end: string): boolean {
  return isIsoDate(start) && isIsoDate(end) && new Date(start) <= new Date(end);
}

/** Clamps a ratio to [0, 1] so calibration clicks never leave the image bounds. */
export function clampRatio(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/** Formats a cumulative return as a signed percentage with two decimals. */
export function formatCumulativeReturn(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

/** Trigger a browser file download from a string content. */
export function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}
