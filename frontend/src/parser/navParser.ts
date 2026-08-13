/**
 * NAV text and spreadsheet parsing utilities.
 *
 * Extracted from main.tsx to allow unit testing and reuse.
 */
import type { NavPoint } from "../api";

/** Calendar-like date formats supported when importing spreadsheets. */
const SPREADSHEET_DATE_PATTERN = /^(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?$/;

/**
 * Parses the review-friendly text input: one `YYYY-MM-DD,value` NAV point per line.
 *
 * Boundary checks:
 * - Empty lines are ignored.
 * - Dates must match ISO format and be parseable.
 * - NAV must be a finite positive number.
 */
export function parseNavText(navText: string): NavPoint[] {
  return navText
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(",").map((item) => item.trim());
      const rawDate = parts[0];
      const rawValue = parts[1];
      if (rawDate === undefined || rawValue === undefined) {
        throw new Error(`无效净值行：${line}`);
      }
      const netAssetValue = Number(rawValue);
      if (!/^\d{4}-\d{2}-\d{2}$/.test(rawDate) || !Number.isFinite(netAssetValue) || netAssetValue <= 0) {
        throw new Error(`无效净值行：${line}`);
      }
      // Defensive: reject clearly impossible dates like 2024-02-30.
      if (Number.isNaN(new Date(`${rawDate}T00:00:00`).getTime())) {
        throw new Error(`无效日期：${rawDate}`);
      }
      return { observation_date: rawDate, net_asset_value: netAssetValue };
    });
}

/** Format the editable review text without exposing image-coordinate noise. */
export function formatNavText(points: NavPoint[], digits = 4): string {
  return points.map((point) => `${point.observation_date},${point.net_asset_value.toFixed(digits)}`).join("\n");
}

/**
 * Keep one observation per Friday-ended calendar week.
 *
 * Chart tracing can return a pixel-derived value for every calendar day. For
 * weekly private-fund disclosures this is false precision, so review opens at
 * weekly cadence and retains the last actually observed value in each week.
 */
export function sampleWeeklyNav(points: NavPoint[]): NavPoint[] {
  const latestInWeek = new Map<string, NavPoint>();
  for (const point of [...points].sort((a, b) => a.observation_date.localeCompare(b.observation_date))) {
    const date = new Date(`${point.observation_date}T00:00:00Z`);
    const weekday = date.getUTCDay();
    const daysUntilFriday = (5 - weekday + 7) % 7;
    const weekEnd = new Date(date);
    weekEnd.setUTCDate(weekEnd.getUTCDate() + daysUntilFriday);
    const key = weekEnd.toISOString().slice(0, 10);
    latestInWeek.set(key, point);
  }
  return [...latestInWeek.values()];
}

/**
 * Normalizes a spreadsheet cell into `YYYY-MM-DD` when possible.
 *
 * Handles Date objects (including Excel dates when `cellDates: true`) and common
 * Chinese date separators. Returns `undefined` when the value cannot be parsed.
 */
export function formatDate(value: unknown): string | undefined {
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    const year = value.getFullYear();
    const month = String(value.getMonth() + 1).padStart(2, "0");
    const day = String(value.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }
  if (typeof value !== "string") return undefined;
  const match = value.trim().match(SPREADSHEET_DATE_PATTERN);
  if (!match) return undefined;
  const [, year, month, day] = match;
  if (year === undefined || month === undefined || day === undefined) return undefined;
  return `${year}-${month.padStart(2, "0")}-${day.padStart(2, "0")}`;
}

/**
 * Reads the first worksheet of a CSV/XLSX file and converts it to reviewable NAV points.
 *
 * Only the first two columns are used (date, unit NAV). The first non-empty row is
 * inspected to decide whether it is a header row. At least two valid data rows are
 * required so that period returns can be computed.
 */
export async function importNavFile(file: File): Promise<NavPoint[]> {
  // Lazy-load the heavy xlsx library only when the user actually imports a file.
  const XLSX = await import("xlsx");
  const workbook = XLSX.read(await file.arrayBuffer(), { cellDates: true });
  const firstSheetName = workbook.SheetNames[0];
  if (!firstSheetName) throw new Error("文件中没有可读取的工作表。");

  const sheet = workbook.Sheets[firstSheetName];
  if (!sheet) throw new Error("文件中没有可读取的工作表。");
  const rows = XLSX.utils.sheet_to_json<unknown[]>(sheet, {
    header: 1,
    defval: "",
    raw: false,
  });
  const firstRow = rows.find((row) => row.some((cell) => String(cell).trim() !== ""));
  if (!firstRow) throw new Error("文件中没有净值数据。");

  const looksLikeHeader = firstRow.some((cell) => /日期|date|净值|nav|value/i.test(String(cell)));
  const headerOffset = looksLikeHeader ? 1 : 0;
  const dataRows = rows.slice(rows.indexOf(firstRow) + headerOffset);

  const points = dataRows
    .filter((row) => row.some((cell) => String(cell).trim() !== ""))
    .map((row, index) => {
      const observationDate = formatDate(row[0]);
      const rawNav = String(row[1]).replace(/,/g, "").trim();
      const netAssetValue = Number(rawNav);
      if (!observationDate || !Number.isFinite(netAssetValue) || netAssetValue <= 0) {
        throw new Error(`第 ${index + 1 + headerOffset} 行无法识别为"日期、单位净值"。`);
      }
      return { observation_date: observationDate, net_asset_value: netAssetValue };
    });

  if (points.length < 2) throw new Error("至少需要两条有效净值记录。");
  return points;
}
