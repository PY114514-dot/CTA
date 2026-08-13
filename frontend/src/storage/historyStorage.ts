/**
 * History record persistence via localStorage.
 *
 * Extracted from main.tsx for testability and reuse.
 */
import type { DataFrequency, ProductStrategyProfileResponse, ReportGenerateResponse } from "../api";

export interface HistoryRecord {
  id: string;
  saved_at: string;
  product_name: string;
  frequency: DataFrequency;
  nav_count: number;
  nav_text: string;
  metrics: {
    cumulative_return: number;
    annualized_return: number;
    annualized_volatility: number;
    sharpe_ratio: number | null;
    maximum_drawdown: number;
    calmar_ratio: number | null;
  };
  /** Raw OCR / supplementary disclosure text the profile was derived from. */
  source_text?: string;
  /** Keyword-evidence strategy profile built from the OCR text. */
  strategy_profile?: ProductStrategyProfileResponse;
  /** AI attribution analysis report (summary + structured steps). */
  ai_report?: ReportGenerateResponse;
}

export const HISTORY_STORAGE_KEY = "cta_research_history";

export function loadHistory(): HistoryRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveHistory(records: HistoryRecord[]): void {
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(records));
}
