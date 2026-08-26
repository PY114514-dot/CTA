/** FOF workbench, material research, library, and allocation functions. */

import type {
  FofMaterialResearchResponse,
  FofLibraryProduct,
  FofRiskProfile,
  FofFundInput,
  FofRecommendationResponse,
  ChatMessage,
} from "./types";
import { API_BASE_URL, errorMessage, requestJson } from "./client";

// ---------------------------------------------------------------------------
// FOF library — material research
// ---------------------------------------------------------------------------

/** Upload one source image into the immutable FOF library, then run the evidence-first Agent. */
export async function analyzeFofMaterial(image: File): Promise<FofMaterialResearchResponse> {
  const formData = new FormData();
  formData.append("files", image);
  formData.append("source_label", "FOF 工作台单图研究");
  const upload = await fetch(`${API_BASE_URL}/fof-library/materials`, { method: "POST", body: formData });
  if (!upload.ok) throw new Error(await errorMessage(upload, "素材上传失败"));
  const uploaded = await upload.json() as { materials: Array<{ material_id: string }> };
  const materialId = uploaded.materials[0]?.material_id;
  if (!materialId) throw new Error("素材上传成功，但没有返回素材编号");
  const analysis = await fetch(`${API_BASE_URL}/fof-library/materials/${materialId}/analyze`, { method: "POST" });
  if (!analysis.ok) throw new Error(await errorMessage(analysis, "FOF Agent 单图研究失败"));
  return analysis.json() as Promise<FofMaterialResearchResponse>;
}

export async function listFofLibraryProducts(): Promise<FofLibraryProduct[]> {
  return requestJson("/fof-library/products", "加载 FOF 候选产品失败");
}

/** FOF workbench chat reads the FOF evidence store, not the legacy CTA sample library. */
export async function fofLibraryChat(query: string): Promise<ChatMessage> {
  const response = await fetch(`${API_BASE_URL}/fof-library/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "FOF Agent 对话失败"));
  return response.json() as Promise<ChatMessage>;
}

// ---------------------------------------------------------------------------
// FOF agent runtime — allocation
// ---------------------------------------------------------------------------

export async function recommendFof(request: {
  funds: FofFundInput[];
  risk_profile: FofRiskProfile;
  max_recommendations?: number;
  max_single_fund_weight?: number;
  session_id?: string;
}): Promise<FofRecommendationResponse> {
  const response = await fetch(`${API_BASE_URL}/fof/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "FOF Agent 执行失败"));
  return response.json() as Promise<FofRecommendationResponse>;
}
