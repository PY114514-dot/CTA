/** Product identity, knowledge base CRUD, agent chat, and investment committee functions. */

import type {
  NavPoint,
  DataFrequency,
  ConfirmationStatus,
  ParsingStatus,
  ProductImageRecognitionResponse,
  ProductStrategyProfileResponse,
  KbFile,
  KbBindFileResponse,
  KbProduct,
  KbProductDetail,
  KbProductUpdate,
  KbNavPoint,
  KbNavCandidate,
  KbReviewPageLocation,
  ChatMessage,
  DecisionStatus,
  KbDecision,
  KbDecisionDetail,
  KbTrackingRecord,
  KbReviewTask,
  KbInvestmentMemo,
} from "./types";
import { API_BASE_URL, errorMessage } from "./client";

// ---------------------------------------------------------------------------
// Product identity
// ---------------------------------------------------------------------------

export async function recognizeProductImage(image: File): Promise<ProductImageRecognitionResponse> {
  const formData = new FormData();
  formData.set("image", image);
  const response = await fetch(`${API_BASE_URL}/product/recognize-image`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "产品名称识别失败"));
  return response.json() as Promise<ProductImageRecognitionResponse>;
}

export async function buildProductStrategyProfile(productName: string, sourceText: string): Promise<ProductStrategyProfileResponse> {
  const response = await fetch(`${API_BASE_URL}/product/strategy-profile`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ product_name: productName, source_text: sourceText }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "策略画像生成失败"));
  return response.json() as Promise<ProductStrategyProfileResponse>;
}

// ---------------------------------------------------------------------------
// Knowledge base — files
// ---------------------------------------------------------------------------

export async function kbUploadFile(file: File, materialNature?: string): Promise<{ id: string; file_hash: string; size_bytes: number; version: number; parsing_status: ParsingStatus }> {
  const formData = new FormData();
  formData.append("file", file);
  if (materialNature) formData.append("material_nature", materialNature);
  const response = await fetch(`${API_BASE_URL}/kb/files/upload`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "文件上传失败"));
  return response.json();
}

export async function kbImportFuturesWeeklySqlite(file: File): Promise<{ products_selected: number; products_created: number; products_existing: number; nav_observations_added: number }> {
  const formData = new FormData();
  formData.set("file", file);
  formData.set("limit", "100");
  formData.set("confirm_products", "true");
  const response = await fetch(`${API_BASE_URL}/kb/files/import-futures-weekly-sqlite`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "SQLite 导入失败"));
  return response.json();
}

export async function kbReparseFile(fileId: string): Promise<{ id: string; parsing_status: ParsingStatus }> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}/reparse`, { method: "POST" });
  if (!response.ok) throw new Error(await errorMessage(response, "重新解析失败"));
  return response.json();
}

export async function kbDeleteFile(fileId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response, "删除文件失败"));
}

export async function kbUpdateFileMaterialNature(fileId: string, materialNature: "CTA策略介绍" | "CTA研究/市场资料" | "忽略"): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}/material-nature`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ material_nature: materialNature }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "更新资料用途失败"));
}

/** Resolve the product identity of an uploaded file (or create a pending one). */
export async function kbBindFile(
  fileId: string,
  payload: {
    product_id?: string;
    product_name?: string;
    manager_name?: string;
    strategy?: string;
  },
): Promise<KbBindFileResponse> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}/bind`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await errorMessage(response, "绑定产品失败");
    if (response.status === 404 && detail === "Not Found") {
      throw new Error("后端绑定接口尚未加载，请先重启 start.py 后再试");
    }
    throw new Error(detail);
  }
  return response.json() as Promise<KbBindFileResponse>;
}

/** Create a pending product for a separately reviewed curve. */
export async function kbCreateProduct(payload: { standard_name: string; manager_name?: string; strategy?: string }): Promise<{ id: string; standard_name: string; confirmation_status: ConfirmationStatus }> {
  const response = await fetch(`${API_BASE_URL}/kb/products`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "新建产品失败"));
  return response.json();
}

/** Confirm different product targets for individual curves in one source file. */
export async function kbBindFileCurves(
  fileId: string,
  bindings: Array<{ fragment_id: string; product_id: string }>,
): Promise<{ file_id: string; bound_curves: number; moved_nav: number }> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}/curve-bindings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bindings }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "批量绑定曲线失败"));
  return response.json();
}

export async function kbListFiles(params?: { parsingStatus?: ParsingStatus; pendingOnly?: boolean }): Promise<KbFile[]> {
  const query = new URLSearchParams();
  if (params?.parsingStatus) query.set("parsing_status", params.parsingStatus);
  if (params?.pendingOnly) query.set("pending_only", "true");
  const suffix = query.size ? `?${query.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/kb/files${suffix}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取文件列表失败"));
  return response.json() as Promise<KbFile[]>;
}

export interface IngestionQueueSnapshot {
  max_concurrency: number;
  active_count: number;
  active_file_ids: string[];
  processing_count: number;
  queued_count: number;
}

/** Describe the async parse queue so the upload card can show capacity and progress. */
export async function kbGetIngestionQueue(): Promise<IngestionQueueSnapshot> {
  const response = await fetch(`${API_BASE_URL}/kb/files/ingestion-queue`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取解析队列状态失败"));
  return response.json() as Promise<IngestionQueueSnapshot>;
}

/** Load the original uploaded material so the user can calibrate it in-place. */
export async function kbGetSourceFile(fileId: string): Promise<File> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}/content`);
  if (!response.ok) throw new Error(await errorMessage(response, "无法读取原始图片"));
  const blob = await response.blob();
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = disposition.match(/filename\*?=(?:UTF-8''|\")?([^;\"]+)/i);
  const filename = decodeURIComponent(match?.[1]?.replace(/[\"]/g, "") || "source-image");
  return new File([blob], filename, { type: blob.type || "image/png" });
}

/** Load an image preview for review; PDF sources are rendered to a PNG page. */
export async function kbGetSourcePreview(fileId: string, page = 0): Promise<File> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}/preview?page=${page}`);
  if (!response.ok) throw new Error(await errorMessage(response, "无法生成原始资料预览"));
  const blob = await response.blob();
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = disposition.match(/filename\*?=(?:UTF-8''|\")?([^;\"]+)/i);
  const filename = decodeURIComponent(match?.[1]?.replace(/[\"]/g, "") || "source-preview.png");
  return new File([blob], filename, { type: "image/png" });
}

/** Locate the product performance page before rendering and invoking the image VLM. */
export async function kbGetReviewPage(fileId: string, productId: string): Promise<KbReviewPageLocation> {
  const response = await fetch(`${API_BASE_URL}/kb/files/${fileId}/review-page?product_id=${encodeURIComponent(productId)}`);
  if (!response.ok) throw new Error(await errorMessage(response, "无法定位产品业绩页"));
  return response.json() as Promise<KbReviewPageLocation>;
}

// ---------------------------------------------------------------------------
// Knowledge base — products
// ---------------------------------------------------------------------------

export async function kbListProducts(params?: {
  confirmation_status?: ConfirmationStatus;
  strategy?: string;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<KbProduct[]> {
  const query = new URLSearchParams();
  if (params?.confirmation_status) query.set("confirmation_status", params.confirmation_status);
  if (params?.strategy) query.set("strategy", params.strategy);
  if (params?.search) query.set("search", params.search);
  query.set("limit", String(params?.limit ?? 50));
  if (params?.offset) query.set("offset", String(params.offset));
  const response = await fetch(`${API_BASE_URL}/kb/products?${query.toString()}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取产品列表失败"));
  return response.json() as Promise<KbProduct[]>;
}

export async function kbCountProducts(search?: string): Promise<number> {
  const query = search ? `?search=${encodeURIComponent(search)}` : "";
  const response = await fetch(`${API_BASE_URL}/kb/products/count${query}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取产品数量失败"));
  return (await response.json() as { total: number }).total;
}

export async function kbGetProduct(productId: string): Promise<KbProductDetail> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取产品详情失败"));
  return response.json() as Promise<KbProductDetail>;
}

export async function kbConfirmProduct(productId: string, confirmedBy = "user"): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed_by: confirmedBy }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "确认产品失败"));
}

export async function kbRejectProduct(productId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}/reject`, { method: "POST" });
  if (!response.ok) throw new Error(await errorMessage(response, "拒绝产品失败"));
}

export async function kbUpdateProduct(productId: string, fields: KbProductUpdate): Promise<KbProduct> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "更新产品失败"));
  return response.json() as Promise<KbProduct>;
}

export async function kbDeleteProduct(productId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response, "删除产品失败"));
}

export async function kbGetNavSeries(productId: string, reviewedOnly = false): Promise<KbNavPoint[]> {
  const query = reviewedOnly ? "?reviewed_only=true" : "";
  const response = await fetch(`${API_BASE_URL}/kb/nav/${productId}${query}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取净值序列失败"));
  return response.json() as Promise<KbNavPoint[]>;
}

export async function kbGetNavSeriesBulk(productIds: string[], reviewedOnly = false): Promise<Record<string, KbNavPoint[]>> {
  const response = await fetch(`${API_BASE_URL}/kb/nav/bulk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ product_ids: productIds, reviewed_only: reviewedOnly }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "批量获取净值序列失败"));
  return response.json() as Promise<Record<string, KbNavPoint[]>>;
}

export async function kbListNavCandidates(productId: string): Promise<KbNavCandidate[]> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}/nav-candidates`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取候选净值失败"));
  return response.json() as Promise<KbNavCandidate[]>;
}

export async function kbMachineReviewProducts(productIds?: string[]): Promise<{
  machine_reviewed: number;
  human_review_required: number;
  results: Array<{
    product_id: string;
    product_name: string;
    updated: number;
    status: "machine_reviewed" | "human_review_required";
    machine_reviewed: boolean;
    confidence: number | null;
    reasons: string[];
    scope: string;
  }>;
}> {
  const response = await fetch(`${API_BASE_URL}/kb/machine-review/products`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(productIds ? { product_ids: productIds } : {}),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "机器复核失败"));
  return response.json();
}

/** Replace traced candidate values with a user-reviewed calibrated NAV series. */
export async function kbReplaceNavSeries(
  productId: string,
  points: NavPoint[],
  options?: { frequency?: string; sourceFileId?: string; sourceFragmentId?: string },
): Promise<{ product_id: string; saved: number; review_status: "reviewed"; review_finalization?: Record<string, unknown> | null }> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}/nav`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: points.map((point) => ({ observation_date: point.observation_date, nav: point.net_asset_value })),
      frequency: options?.frequency,
      source_file_id: options?.sourceFileId,
      source_fragment_id: options?.sourceFragmentId,
    }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存校准净值失败"));
  return response.json();
}

// ---------------------------------------------------------------------------
// Agent conversation (P2)
// ---------------------------------------------------------------------------

export async function kbChat(request: {
  query: string;
  product_ids?: string[];
  session_id?: string;
}, signal?: AbortSignal): Promise<ChatMessage> {
  const response = await fetch(`${API_BASE_URL}/kb/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response, "Agent 对话失败"));
  return response.json() as Promise<ChatMessage>;
}

export async function kbSaveAllocationDraft(draftId: string): Promise<{ id: string; status: "draft" }> {
  const response = await fetch(`${API_BASE_URL}/kb/chat/allocation-drafts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ draft_id: draftId }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存配置草案失败"));
  return response.json() as Promise<{ id: string; status: "draft" }>;
}

export async function kbSubmitAllocationDraft(decisionId: string): Promise<{ id: string; status: "pending_review" }> {
  const response = await fetch(`${API_BASE_URL}/kb/chat/allocation-drafts/${decisionId}/submit`, { method: "POST" });
  if (!response.ok) throw new Error(await errorMessage(response, "提交人工确认失败"));
  return response.json() as Promise<{ id: string; status: "pending_review" }>;
}

// ---------------------------------------------------------------------------
// Investment committee & outer-loop tracking (P4)
// ---------------------------------------------------------------------------

export async function kbListDecisions(status?: DecisionStatus): Promise<KbDecision[]> {
  const query = status ? `?status=${status}` : "";
  const response = await fetch(`${API_BASE_URL}/kb/decisions${query}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取决策列表失败"));
  return response.json() as Promise<KbDecision[]>;
}

export async function kbGetDecisionDetail(decisionId: string): Promise<KbDecisionDetail> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}/detail`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取决策详情失败"));
  return response.json() as Promise<KbDecisionDetail>;
}

export async function kbDeleteAllocationDraft(decisionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response, "删除配置草案失败"));
}

export async function kbInterpretAllocation(decisionId: string): Promise<{ decision_id: string; interpretation: string }> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}/interpretation`, { method: "POST" });
  if (!response.ok) throw new Error(await errorMessage(response, "生成配置解读失败"));
  return response.json() as Promise<{ decision_id: string; interpretation: string }>;
}

export async function kbApproveDecision(
  decisionId: string,
  reviewer: string,
  comment?: string,
  adjustment?: Record<string, unknown>,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviewer, comment, adjustment }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "审批通过失败"));
}

export async function kbVetoDecision(decisionId: string, reviewer: string, reason: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}/veto`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviewer, reason }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "否决失败"));
}

export async function kbReopenDecision(decisionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}/reopen`, { method: "POST" });
  if (!response.ok) throw new Error(await errorMessage(response, "重新审议失败"));
}

export async function kbExportMemo(decisionId: string): Promise<KbInvestmentMemo> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}/memo`);
  if (!response.ok) throw new Error(await errorMessage(response, "导出备忘录失败"));
  return response.json() as Promise<KbInvestmentMemo>;
}

export async function kbRunTracking(
  decisionId: string,
  reviewDate?: string,
): Promise<KbTrackingRecord & { triggered_task: KbReviewTask | null }> {
  const response = await fetch(`${API_BASE_URL}/kb/decisions/${decisionId}/track`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_date: reviewDate ?? null }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "推荐后追踪失败"));
  return response.json();
}

export async function kbListReviewTasks(status?: string): Promise<KbReviewTask[]> {
  const query = status ? `?status=${status}` : "";
  const response = await fetch(`${API_BASE_URL}/kb/review-tasks${query}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取复核任务失败"));
  return response.json() as Promise<KbReviewTask[]>;
}

export async function kbResolveReviewTask(
  taskId: string,
  resolution: string,
  resolvedBy: string,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/review-tasks/${taskId}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resolution, resolved_by: resolvedBy }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "处理复核任务失败"));
}
