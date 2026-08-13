/**
 * Axis label parsing utilities for OCR-recognized chart labels.
 *
 * Extracted from main.tsx to allow unit testing and reuse.
 */

/** Try to read a YYYY-MM-DD / YYYY/MM/DD / 中文日期 from an axis label. */
export function parseAxisDate(text: string): string | undefined {
  const normalized = text
    .replace(/[年月]/g, "-")
    .replace(/日/g, "")
    .replace(/\//g, "-")
    // OCR (especially on rotated/vertical labels) often renders "-" as the
    // CJK stroke "一" or an en/em dash, and may insert spaces between tokens.
    .replace(/[一–—]/g, "-")
    .replace(/\s*-\s*/g, "-");
  const match = normalized.match(/\b(\d{4})-(\d{1,2})-(\d{1,2})\b/);
  if (!match) return undefined;
  const [, y, m, d] = match;
  if (y === undefined || m === undefined || d === undefined) return undefined;
  const year = y;
  const month = m.padStart(2, "0");
  const day = d.padStart(2, "0");
  if (Number.isNaN(new Date(`${year}-${month}-${day}T00:00:00`).getTime())) return undefined;
  return `${year}-${month}-${day}`;
}

/** Parse a number (possibly with %) from an axis label. */
export function parseAxisNumber(text: string, mode: "nav" | "cumulative_return"): number | undefined {
  const cleaned = text
    .replace(/,/g, "")
    .replace(/[＋]/g, "+")
    .replace(/[－]/g, "-")
    // OCR (esp. on rotated/vertical labels) may read a leading minus as the
    // CJK stroke "一" or, with the eng model, as an en/em dash "–"/"—";
    // treat any of these as "-" only when a digit/dot follows (with optional
    // whitespace, which the vertical OCR path inserts) so a genuine "一"
    // elsewhere is left alone.  Also drop spaces between a sign and its
    // digits.
    .replace(/[一–—]\s*(?=[\d.])/g, "-")
    .replace(/([+\-])\s+(?=\d)/g, "$1");
  const percentMatch = cleaned.match(/(-?\d+(?:\.\d+)?)\s*%/);
  if (percentMatch) {
    const value = Number(percentMatch[1]);
    return Number.isFinite(value) ? value : undefined;
  }
  const numberMatch = cleaned.match(/(-?\d+(?:\.\d+)?)/);
  if (!numberMatch) return undefined;
  const value = Number(numberMatch[1]);
  if (!Number.isFinite(value)) return undefined;
  // Heuristic: small decimals in cumulative-return mode are likely ratios like 0.12.
  if (mode === "cumulative_return" && Math.abs(value) <= 1) {
    return value * 100;
  }
  return value;
}
