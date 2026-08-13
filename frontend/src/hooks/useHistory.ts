/**
 * useHistory — manages persisted analysis history records (localStorage-backed).
 */
import { useState } from "react";
import type { NavAnalysisResponse, ReportGenerateResponse, ProductStrategyProfileResponse } from "../api";
import type { DataFrequency } from "../api";
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

  function handleSaveHistory(params: {
    productName: string;
    frequency: DataFrequency;
    navCount: number;
    navText: string;
    analysisResult: NavAnalysisResponse;
    sourceText?: string;
    strategyProfile?: ProductStrategyProfileResponse;
    aiReport?: ReportGenerateResponse;
  }): void {
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
  }

  function handleDeleteHistory(id: string): void {
    const updated = historyRecords.filter((r) => r.id !== id);
    setHistoryRecords(updated);
    saveHistory(updated);
  }

  function handleClearAllHistory(): void {
    setHistoryRecords([]);
    saveHistory([]);
  }

  return {
    historyRecords,
    isHistoryOpen, setIsHistoryOpen,
    handleSaveHistory, handleDeleteHistory, handleClearAllHistory,
  };
}
