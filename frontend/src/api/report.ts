/**
 * Report domain — AI report generation, PDF export, LLM config,
 * multi-product report extraction, and PaddleOCR document parsing.
 */

import type {
  NavPoint,
  DataFrequency,
  StrategyConfirmation,
  ReportGenerateResponse,
  LlmConfigResponse,
  ModelConnectionTest,
  MultiProductReportResponse,
  PaddleOcrDocumentResponse,
} from "./types";
import { API_BASE_URL, errorMessage, requestJson } from "./client";

// ---------------------------------------------------------------------------
// AI report generation
// ---------------------------------------------------------------------------

/** Download a server-rendered PDF report without relying on a popup or print dialog. */
export async function downloadAnalysisReportPdf(productName: string, markdown: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/analysis/report-pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ product_name: productName, markdown }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "PDF 导出失败"));
  return response.blob();
}

/** One-click full pipeline: classify → factors → varieties → report. */
export async function generateAnalysisReport(
  navPoints: NavPoint[],
  frequency: DataFrequency,
  dataSource: "api" | "upload" = "api",
  rollingWindow: number = 12,
  topNVarities: number = 8,
  strategyHint?: string,
  strategyConfirmation: StrategyConfirmation = "auto",
  qualityOverrideConfirmed: boolean = false,
  signal?: AbortSignal,
): Promise<ReportGenerateResponse> {
  const response = await fetch(`${API_BASE_URL}/analysis/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      nav_points: navPoints,
      frequency,
      data_source: dataSource,
      rolling_window: rollingWindow,
      top_n_varieties: topNVarities,
      strategy_hint: strategyHint ?? "",
      strategy_confirmation: strategyConfirmation,
      quality_override_confirmed: qualityOverrideConfirmed,
    }),
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response, "AI 策略归因分析失败"));
  return response.json() as Promise<ReportGenerateResponse>;
}

// ---------------------------------------------------------------------------
// LLM config
// ---------------------------------------------------------------------------

/** Get current LLM configuration. */
export async function getLlmConfig(): Promise<LlmConfigResponse> {
  return requestJson("/analysis/config", "获取 LLM 配置失败");
}

/** Update LLM configuration. */
export async function updateLlmConfig(config: {
  api_base: string;
  api_key: string;
  model: string;
  enabled: boolean;
  thinking_enabled: boolean;
  max_tokens: number;
}): Promise<LlmConfigResponse> {
  const response = await fetch(`${API_BASE_URL}/analysis/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "更新 LLM 配置失败"));
  return response.json() as Promise<LlmConfigResponse>;
}

/** Send a short real request to the configured text model. */
export async function testLlmConfig(): Promise<ModelConnectionTest> {
  return requestJson("/analysis/config/test", "LLM 连通性测试失败", { method: "POST" });
}

// ---------------------------------------------------------------------------
// Multi-product report extraction & PaddleOCR
// ---------------------------------------------------------------------------

export async function extractMultiProductReport(image: File): Promise<MultiProductReportResponse> {
  const formData = new FormData();
  formData.set("image", image);
  const response = await fetch(`${API_BASE_URL}/report/extract-multiple`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "多产品报告识别失败"));
  return response.json() as Promise<MultiProductReportResponse>;
}

/** Parse a PDF or image with the server-side PaddleOCR-VL integration. */
export async function extractDocumentWithPaddleOcr(document: File): Promise<PaddleOcrDocumentResponse> {
  const formData = new FormData();
  formData.set("document", document);
  const response = await fetch(`${API_BASE_URL}/document/paddleocr`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "PaddleOCR 文档解析失败"));
  return response.json() as Promise<PaddleOcrDocumentResponse>;
}
