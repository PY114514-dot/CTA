/**
 * useHistory — manages persisted analysis history records.
 *
 * Primary storage is the backend API (/api/history).  localStorage is kept
 * as a write-through cache so the UI stays responsive even when the backend
 * is unreachable.
 */
import { useCallback, useEffect, useState } from "react";
import type { NavAnalysisResponse, ReportGenerateResponse, ProductStrategyProfileResponse } from "../api";
import type { DataFrequency } from "../api";
import {
  listHistoryRecords,
  createHistoryRecord,
  deleteHistoryRecord,
  clearHistoryRecords,
  syncHistoryRecords,
} from "../api";
import { type HistoryRecord, loadHistory, saveHistory } from "../storage/historyStorage";

export interface HistoryState {
  historyRecords: HistoryRecord[];
  isHistoryOpen: boolean;
  setIsHistoryOpen: React.Dispatch<React.SetStateAction<boolean>>;
  handleSaveHistory: (params: {
    productName: string;
    frequency: DataFrequency;
    navCount: number;
    navText: string;
    analysisResult: NavAnalysisResponse;
    sourceText?: string;
    strategyProfile?: ProductStrategyProfileResponse;
    aiReport?: ReportGenerateResponse;
  }) => void;
  handleDeleteHistory: (id: string) => void;
  handleClearAllHistory: () => void;
}

export function useHistory(): HistoryState {
  const [historyRecords, setHistoryRecords] = useState<HistoryRecord[]>(loadHistory);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  // On mount, try to load history from the backend.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const remote = await listHistoryRecords();
        if (cancelled) return;
        if (remote.length > 0) {
          const mapped: HistoryRecord[] = remote.map((r) => ({
            id: r.id,
            saved_at: r.saved_at,
            product_name: r.product_name,
            frequency: r.frequency,
            nav_count: r.nav_count,
            nav_text: r.nav_text,
            metrics: r.metrics,
            source_text: r.source_text,
            strategy_profile: r.strategy_profile as HistoryRecord["strategy_profile"],
            ai_report: r.ai_report as HistoryRecord["ai_report"],
          }));
          setHistoryRecords(mapped);
          saveHistory(mapped); // update localStorage cache
        }
      } catch {
        // Backend unreachable — localStorage is the fallback.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const handleSaveHistory = useCallback((params: {
    productName: string;
    frequency: DataFrequency;
    navCount: number;
    navText: string;
    analysisResult: NavAnalysisResponse;
    sourceText?: string;
    strategyProfile?: ProductStrategyProfileResponse;
    aiReport?: ReportGenerateResponse;
  }): void => {
    const record: HistoryRecord = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      saved_at: new Date().toLocaleString("zh-CN"),
      product_name: params.productName.trim() || "未命名产品",
      frequency: params.frequency,
      nav_count: params.navCount,
      nav_text: params.navText,
      metrics: params.analysisResult.metrics,
      source_text: params.sourceText?.trim() || undefined,
      strategy_profile: params.strategyProfile,
      ai_report: params.aiReport,
    };
    const updated = [record, ...historyRecords].slice(0, 50);
    setHistoryRecords(updated);
    saveHistory(updated);

    // Fire-and-forget backend sync.
    createHistoryRecord({
      id: record.id,
      product_name: record.product_name,
      frequency: record.frequency,
      nav_count: record.nav_count,
      nav_text: record.nav_text,
      metrics: record.metrics,
      source_text: record.source_text,
      strategy_profile: record.strategy_profile as unknown as Record<string, unknown> | undefined,
      ai_report: record.ai_report as unknown as Record<string, unknown> | undefined,
    }).catch(() => { /* offline — localStorage is authoritative */ });
  }, [historyRecords]);

  const handleDeleteHistory = useCallback((id: string): void => {
    const updated = historyRecords.filter((r) => r.id !== id);
    setHistoryRecords(updated);
    saveHistory(updated);
    deleteHistoryRecord(id).catch(() => { /* best-effort */ });
  }, [historyRecords]);

  const handleClearAllHistory = useCallback((): void => {
    setHistoryRecords([]);
    saveHistory([]);
    clearHistoryRecords().catch(() => { /* best-effort */ });
  }, []);

  return {
    historyRecords,
    isHistoryOpen, setIsHistoryOpen,
    handleSaveHistory, handleDeleteHistory, handleClearAllHistory,
  };
}
