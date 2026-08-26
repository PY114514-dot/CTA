/**
 * CTA domain — factor attribution, factor library management, external factors,
 * sector breakdown, CTA ranking, CTA scores, and CTA-Fama audit.
 */

import type {
  NavPoint,
  DataFrequency,
  FactorBetaItem,
  RollingSnapshot,
  FactorAttributionResponse,
  WeeklyReportSnapshot,
  FactorValidationResponse,
  FactorMetaItem,
  FactorDetail,
  BuildLogEntry,
  BuildJobResult,
  BuildJobStatus,
  FactorPerformanceItem,
  RiskOverlayMeta,
  SectorContribution,
  FactorSectorBreakdown,
  SectorBreakdownResponse,
  ExternalFactorObservation,
  CtaRankingAttributionEvidencePayload,
  CtaRankingRequestPayload,
  CtaRankingDimensionScore,
  CtaRankingProductResult,
  CtaRankingResponse,
  CtaRankingSnapshotSummary,
  CtaRankingSnapshotResponse,
  CtaPortfolioPositionPayload,
  CtaProductScoreRequestPayload,
  CtaConfidenceScore,
  CtaQualityHistoryPoint,
  CtaProductScoreItem,
  CtaAllocationScore,
  CtaProductScoreResponse,
  CtaProductScoreSnapshot,
  CtaFamaReadinessRequestPayload,
  CtaFamaReadinessResponse,
  CtaFamaAuditState,
} from "./types";
import { API_BASE_URL, errorMessage, requestJson } from "./client";

// ---------------------------------------------------------------------------
// External factor library (国泰君安 etc.)
// ---------------------------------------------------------------------------

export async function getGuotaiJunanExternalFactors(): Promise<{ source: string; latest_as_of_date: string | null; observations: ExternalFactorObservation[] }> {
  const response = await fetch(`${API_BASE_URL}/external-factor-library/guotai-junan/observations`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取国泰君安因子库失败"));
  return response.json();
}

export function externalFactorExportUrl(format: "csv" | "json" | "xlsx"): string {
  return `${API_BASE_URL}/external-factor-library/guotai-junan/export/${format}`;
}

export async function importGuotaiJunanExternalFactorReport(file: File): Promise<{ as_of_date: string; rows_added: number; revision: number; duplicate: boolean }> {
  const body = new FormData(); body.set("file", file);
  const response = await fetch(`${API_BASE_URL}/external-factor-library/guotai-junan/import`, { method: "POST", body });
  if (!response.ok) throw new Error(await errorMessage(response, "周报因子导入失败"));
  return response.json();
}

export async function importExternalFactorReport(file: File, organizationName?: string): Promise<{ organization_name: string; as_of_date: string; rows_added: number; revision: number; duplicate: boolean }> {
  const body = new FormData(); body.set("file", file); if (organizationName?.trim()) body.set("organization_name", organizationName.trim());
  const response = await fetch(`${API_BASE_URL}/external-factor-library/import`, { method: "POST", body });
  if (!response.ok) throw new Error(await errorMessage(response, "外部周报导入失败"));
  return response.json();
}

export async function verifyGuotaiJunanExternalFactors(asOfDate: string): Promise<{ as_of_date: string; verified_rows: number }> {
  const response = await fetch(`${API_BASE_URL}/external-factor-library/guotai-junan/verify`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ as_of_date: asOfDate, verified_by: "user" }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "核验状态更新失败"));
  return response.json();
}

// ---------------------------------------------------------------------------
// Factor Library — Quantitative L1 Attribution
// ---------------------------------------------------------------------------

export async function parseWeeklyReport(file: File): Promise<WeeklyReportSnapshot> {
  const body = new FormData();
  body.set("file", file);
  const response = await fetch(`${API_BASE_URL}/report/parse-weekly`, { method: "POST", body });
  if (!response.ok) throw new Error(await errorMessage(response, "周报解析失败"));
  return response.json() as Promise<WeeklyReportSnapshot>;
}

export async function validateFactorAttribution(params: {
  regressionResult: FactorAttributionResponse;
  reportSnapshot: WeeklyReportSnapshot;
  periodStart: string;
  periodEnd: string;
}): Promise<FactorValidationResponse> {
  const response = await fetch(`${API_BASE_URL}/factor-library/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      regression_result: params.regressionResult,
      report_snapshot: params.reportSnapshot,
      period_start: params.periodStart,
      period_end: params.periodEnd,
    }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "L1 与周报校验失败"));
  return response.json() as Promise<FactorValidationResponse>;
}

/** Run quantitative factor attribution (OLS regression against factor library). */
export async function runFactorAttribution(
  navPoints: NavPoint[],
  frequency: DataFrequency,
  rollingWindow?: number,
  factorNames?: string[],
  signal?: AbortSignal,
  bootstrapReps = 1000,
  oosTrainWindow?: number,
): Promise<FactorAttributionResponse> {
  const body: Record<string, unknown> = {
    nav_points: navPoints.map((p) => ({ date: p.observation_date, nav: p.net_asset_value })),
    frequency,
  };
  if (rollingWindow) body.rolling_window = rollingWindow;
  if (factorNames && factorNames.length > 0) body.factor_names = factorNames;
  body.bootstrap_reps = bootstrapReps;
  if (oosTrainWindow) body.oos_train_window = oosTrainWindow;

  const response = await fetch(`${API_BASE_URL}/factor-library/attribute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response, "定量因子归因失败"));
  return response.json() as Promise<FactorAttributionResponse>;
}

/** Check factor library build status. */
export async function getFactorLibraryStatus(): Promise<{
  cached_factors: Record<string, { built_at: string; rows: number; start: string; end: string }>;
  providers: { akshare_available: boolean; csv_symbols: string[] };
  core_varieties: string[];
}> {
  const response = await fetch(`${API_BASE_URL}/factor-library/status`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取因子库状态失败"));
  return response.json();
}

// ---------------------------------------------------------------------------
// Factor library management
// ---------------------------------------------------------------------------

/** List all registered factors with metadata. */
export async function listFactors(): Promise<{
  factors: FactorMetaItem[];
  core_varieties: string[];
  sector_coverage: Record<string, string[]>;
}> {
  const response = await fetch(`${API_BASE_URL}/factor-library/factors`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取因子列表失败"));
  return response.json();
}

/** Fetch rich detail (signal rule, LaTeX formula, derivation, code) for one factor. */
export async function getFactorDetail(name: string): Promise<FactorDetail> {
  const response = await fetch(`${API_BASE_URL}/factor-library/factors/${encodeURIComponent(name)}/detail`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await errorMessage(response, "获取因子详情失败"));
  return response.json();
}

/** Start an async factor library build; returns a job id for polling. */
export async function startFactorBuild(params: {
  start: string;
  end: string;
  factors?: string[];
  useCache?: boolean;
}): Promise<{ job_id: string }> {
  const query = new URLSearchParams({ start: params.start, end: params.end });
  if (params.factors && params.factors.length > 0) query.set("factors", params.factors.join(","));
  query.set("use_cache", String(params.useCache ?? true));
  const response = await fetch(`${API_BASE_URL}/factor-library/build-async?${query.toString()}`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await errorMessage(response, "启动因子构建失败"));
  return response.json();
}

/** Poll an async build job for progress / logs / result. */
export async function getBuildJob(jobId: string): Promise<BuildJobStatus> {
  return requestJson(`/factor-library/build-job/${jobId}`, "查询构建任务失败");
}

/** Performance statistics for all cached factors. */
export async function getFactorPerformance(
  riskProfile: RiskOverlayMeta["profile"] = "baseline",
): Promise<{ performance: FactorPerformanceItem[]; risk_overlay: RiskOverlayMeta }> {
  const response = await fetch(`${API_BASE_URL}/factor-library/performance?risk_profile=${riskProfile}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取因子绩效失败"));
  return response.json();
}

/** Export factor library catalog + performance as CSV or Markdown text. */
export async function exportFactorLibrary(format: "csv" | "markdown"): Promise<{
  filename: string;
  content_type: string;
  content: string;
}> {
  const response = await fetch(`${API_BASE_URL}/factor-library/export?format=${format}`);
  if (!response.ok) throw new Error(await errorMessage(response, "导出因子库失败"));
  return response.json();
}

// ---------------------------------------------------------------------------
// Sector breakdown (板块盈亏归因)
// ---------------------------------------------------------------------------

/** Fetch sector-level P&L attribution for factors over a date range. */
export async function getSectorBreakdown(
  start: string,
  end: string,
  factorNames?: string[],
): Promise<SectorBreakdownResponse> {
  const response = await fetch(`${API_BASE_URL}/factor-library/sector-breakdown`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start, end, factor_names: factorNames || null }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "获取板块盈亏归因失败"));
  return response.json();
}

// ---------------------------------------------------------------------------
// CODEX CTA ranking
// ---------------------------------------------------------------------------

export async function persistCtaRankingSnapshot(request: CtaRankingRequestPayload): Promise<CtaRankingSnapshotResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-ranking/snapshots`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "CODEX 排名生成失败"));
  return response.json() as Promise<CtaRankingSnapshotResponse>;
}

export async function listCtaRankingSnapshots(limit = 12): Promise<CtaRankingSnapshotSummary[]> {
  const response = await fetch(`${API_BASE_URL}/cta-ranking/snapshots?limit=${limit}`);
  if (!response.ok) throw new Error(await errorMessage(response, "加载 CODEX 排名历史失败"));
  return response.json() as Promise<CtaRankingSnapshotSummary[]>;
}

export async function getCtaRankingSnapshot(snapshotId: string): Promise<CtaRankingSnapshotResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-ranking/snapshots/${snapshotId}`);
  if (!response.ok) throw new Error(await errorMessage(response, "加载 CODEX 排名快照失败"));
  return response.json() as Promise<CtaRankingSnapshotResponse>;
}

export async function deleteCtaRankingSnapshot(snapshotId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/cta-ranking/snapshots/${snapshotId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response, "删除 CODEX 排名快照失败"));
}

export async function getCtaProductScoreProfile(request: CtaProductScoreRequestPayload): Promise<CtaProductScoreResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-scores/profile`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA 产品评分失败"));
  return response.json() as Promise<CtaProductScoreResponse>;
}

export async function persistCtaProductScoreSnapshot(request: CtaProductScoreRequestPayload): Promise<CtaProductScoreSnapshot> {
  const response = await fetch(`${API_BASE_URL}/cta-scores/snapshots`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "生成 CTA 产品评分失败"));
  return response.json() as Promise<CtaProductScoreSnapshot>;
}

export async function getLatestCtaProductScoreSnapshot(): Promise<CtaProductScoreSnapshot> {
  const response = await fetch(`${API_BASE_URL}/cta-scores/snapshots/latest`);
  if (!response.ok) throw new Error(await errorMessage(response, "暂无已固化的 CTA 产品评分"));
  return response.json() as Promise<CtaProductScoreSnapshot>;
}

export async function getLatestCtaProductScoreItem(productId: string): Promise<{ as_of_date: string; item: CtaProductScoreItem }> {
  const response = await fetch(`${API_BASE_URL}/cta-scores/snapshots/latest/${productId}`);
  if (!response.ok) throw new Error(await errorMessage(response, "暂无该产品的已固化评分"));
  return response.json() as Promise<{ as_of_date: string; item: CtaProductScoreItem }>;
}

export type ProductArchiveItem = { product_id: string; name: string; manager: string | null; strategy: string | null; nav_count: number; annual_return: number | null; annual_volatility: number; sharpe: number | null; maximum_drawdown: number; data_status: "正常" | "高回撤"; rank: number | null; quality_score: number | null; confidence_score: number | null; attribution_quality_score: number | null; r_squared: number | null; oos_r_squared: number | null };
export async function getProductArchive(): Promise<{ items: ProductArchiveItem[]; ranking_as_of: string | null; score_as_of: string | null }> {
  const response = await fetch(`${API_BASE_URL}/product-archive/products`);
  if (!response.ok) throw new Error(await errorMessage(response, "加载产品研究库失败"));
  return response.json();
}
export async function getProductPeers(productId: string): Promise<{ similar: Array<{ product_id: string; name: string; manager: string | null; correlation: number; overlap: number }>; diversifiers: Array<{ product_id: string; name: string; manager: string | null; correlation: number; overlap: number }> }> {
  const response = await fetch(`${API_BASE_URL}/product-archive/products/${productId}/peers`);
  if (!response.ok) throw new Error(await errorMessage(response, "加载同类产品失败"));
  return response.json();
}

export async function getCtaAllocationScore(request: CtaProductScoreRequestPayload): Promise<{
  model_version: string;
  quality_model_version: string;
  as_of_date: string;
  allocation: CtaAllocationScore;
  warnings: string[];
}> {
  const response = await fetch(`${API_BASE_URL}/cta-scores/allocation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA 配置分失败"));
  return response.json();
}

// ---------------------------------------------------------------------------
// CTA-Fama audit
// ---------------------------------------------------------------------------

export async function getCtaFamaAudit(): Promise<CtaFamaAuditState> {
  const response = await fetch(`${API_BASE_URL}/cta-fama/audit`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA-Fama 审计失败"));
  return response.json() as Promise<CtaFamaAuditState>;
}

export async function updateCtaFamaGlobalAudit(payload: Omit<CtaFamaAuditState["global"], "reviewed_by" | "updated_at"> & { reviewed_by?: string }): Promise<CtaFamaAuditState["global"]> {
  const response = await fetch(`${API_BASE_URL}/cta-fama/audit/global`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存 CTA-Fama 全局审计失败"));
  return response.json();
}

export async function updateCtaFamaProductAudit(productId: string, payload: { source_group?: string | null; point_in_time_verified: boolean; reviewed_by?: string }): Promise<CtaFamaAuditState["products"][number]> {
  const response = await fetch(`${API_BASE_URL}/cta-fama/audit/products/${productId}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存 CTA-Fama 产品审计失败"));
  return response.json();
}

export async function getCtaFamaReadiness(request: CtaFamaReadinessRequestPayload): Promise<CtaFamaReadinessResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-fama/readiness`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA-Fama 数据门槛失败"));
  return response.json() as Promise<CtaFamaReadinessResponse>;
}
