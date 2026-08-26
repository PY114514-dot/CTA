/** CTA attribution transport, separated from the general research client. */

import {
  API_BASE_URL,
  errorMessage,
  type CtaAttributionPhase,
  type CtaAttributionResult,
  type CtaAttributionSnapshotResponse,
  type CtaAttributionSnapshotSummary,
  type CtaPhaseAResponse,
  type CtaPhaseBResponse,
  type CtaPhaseCResponse,
  type CtaPhaseDResponse,
  type CtaRankingAttributionEvidencePayload,
  type ModelApplicability,
} from "../api";

export async function getCtaPhaseA(productId: string): Promise<CtaPhaseAResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-a`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 已审核净值归因失败"));
  return response.json() as Promise<CtaPhaseAResponse>;
}

export async function getModelApplicability(productId: string): Promise<ModelApplicability> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/applicability`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取模型适用性失败"));
  return response.json() as Promise<ModelApplicability>;
}

export async function uploadAttributionFactorCsv(pool: "equity_quant" | "options_volatility", file: File, source: string, dataVersion: string): Promise<void> {
  const body = new FormData();
  body.set("file", file);
  body.set("source", source);
  body.set("data_version", dataVersion);
  const response = await fetch(`${API_BASE_URL}/attribution-factors/${pool}/csv`, { method: "POST", body });
  if (!response.ok) throw new Error(await errorMessage(response, "因子 CSV 上传失败"));
}

export async function importEquityFactorsFromAkShare(start: string, end: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/attribution-factors/equity_quant/akshare`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start, end }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "AKShare 股票因子获取失败"));
}

export async function getCtaPhaseB(productId: string): Promise<CtaPhaseBResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-b`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 动态 Beta 归因失败"));
  return response.json() as Promise<CtaPhaseBResponse>;
}

export async function getCtaPhaseC(productId: string): Promise<CtaPhaseCResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-c`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 状态归因失败"));
  return response.json() as Promise<CtaPhaseCResponse>;
}

export async function getCtaPhaseD(productId: string): Promise<CtaPhaseDResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-d`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 非线性增量归因失败"));
  return response.json() as Promise<CtaPhaseDResponse>;
}

export async function createCtaAttributionSnapshot(productId: string, phase: CtaAttributionPhase): Promise<{
  snapshot_id: string; created_at: string | null; snapshot_type: "cta_dynamic_attribution"; phase: CtaAttributionPhase;
  model_version: string; product_id: string; as_of_date: string; nav_fingerprint: string;
  factor_data_version: string; result: CtaAttributionResult; idempotent: boolean;
}> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ product_id: productId, phase }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存 CTA 归因快照失败"));
  return response.json();
}

export async function listCtaAttributionSnapshots(productId: string, limit = 20): Promise<CtaAttributionSnapshotSummary[]> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots?${new URLSearchParams({ product_id: productId, limit: String(limit) })}`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA 归因快照失败"));
  return response.json();
}

export async function getCtaAttributionSnapshot(snapshotId: string): Promise<CtaAttributionSnapshotResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots/${snapshotId}`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA 归因快照详情失败"));
  return response.json();
}

export async function getLatestCtaPhaseDEvidence(productIds: string[], asOfDate?: string): Promise<Record<string, CtaRankingAttributionEvidencePayload>> {
  const query = new URLSearchParams();
  if (productIds.length) query.set("product_ids", productIds.join(","));
  if (asOfDate) query.set("as_of_date", asOfDate);
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots/latest-phase-d?${query}`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取 Phase D 冻结证据失败"));
  return response.json() as Promise<Record<string, CtaRankingAttributionEvidencePayload>>;
}
