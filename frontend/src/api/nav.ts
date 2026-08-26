/** NAV analysis, image digitization, chart extraction, and OCR functions. */

import type {
  NavPoint,
  DataFrequency,
  ImageExtractionFrequency,
  NavAnalysisResponse,
  NavImageDigitizationResponse,
  OcrRegionResponse,
  ExtractCurveSpec,
  ExtractSingleImageResponse,
  ExtractPdfResponse,
  ExtractJobStatus,
  ExtractJobPreview,
  VlmConfig,
  ModelConnectionTest,
} from "./types";
import { API_BASE_URL, errorMessage, requestJson } from "./client";

// ---------------------------------------------------------------------------
// NAV analysis & digitization
// ---------------------------------------------------------------------------

/** Submit reviewed NAV data and return transparent performance metrics. */
export async function analyzeNav(
  navPoints: NavPoint[],
  frequency: DataFrequency,
  annualRiskFreeRate: number,
): Promise<NavAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/nav/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      nav_points: navPoints,
      frequency,
      annual_risk_free_rate: annualRiskFreeRate,
    }),
  });

  if (!response.ok) {
    throw new Error(await errorMessage(response, "净值分析请求失败"));
  }
  return response.json() as Promise<NavAnalysisResponse>;
}

/** Upload a chart image and return reviewable, manually calibrated NAV candidates. */
export async function digitizeNavImage(
  image: File,
  startDate: string,
  endDate: string,
  navMin: number,
  navMax: number,
  valueMode: "nav" | "cumulative_return",
  frequency: ImageExtractionFrequency,
  startXRatio?: number,
  endXRatio?: number,
  lineKind: "product" | "benchmark" = "product",
  topYRatio?: number,
  bottomYRatio?: number,
  lineColor?: string,
): Promise<NavImageDigitizationResponse> {
  const formData = new FormData();
  formData.set("image", image);
  formData.set("start_date", startDate);
  formData.set("end_date", endDate);
  formData.set("nav_min", String(navMin));
  formData.set("nav_max", String(navMax));
  formData.set("value_mode", valueMode);
  formData.set("frequency", frequency);
  formData.set("line_kind", lineKind);
  if (lineColor) formData.set("line_color", lineColor);
  if (startXRatio !== undefined && endXRatio !== undefined) {
    formData.set("start_x_ratio", String(startXRatio));
    formData.set("end_x_ratio", String(endXRatio));
  }
  if (topYRatio !== undefined && bottomYRatio !== undefined) {
    formData.set("top_y_ratio", String(topYRatio));
    formData.set("bottom_y_ratio", String(bottomYRatio));
  }
  const response = await fetch(`${API_BASE_URL}/nav/digitize-image`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(await errorMessage(response, "图片识别失败"));
  }
  return response.json() as Promise<NavImageDigitizationResponse>;
}

/** OCR a rectangular region of an image for axis-label auto-fill. */
export async function ocrImageRegion(
  image: File,
  leftRatio: number,
  topRatio: number,
  rightRatio: number,
  bottomRatio: number,
): Promise<OcrRegionResponse> {
  const formData = new FormData();
  formData.set("image", image);
  formData.set("left_ratio", String(leftRatio));
  formData.set("top_ratio", String(topRatio));
  formData.set("right_ratio", String(rightRatio));
  formData.set("bottom_ratio", String(bottomRatio));
  const response = await fetch(`${API_BASE_URL}/image/ocr-region`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "区域 OCR 失败"));
  return response.json() as Promise<OcrRegionResponse>;
}

// ---------------------------------------------------------------------------
// Chart extraction — CV pixel tracing + optional VLM structure reading
// ---------------------------------------------------------------------------

/**
 * Extract chart data from a single image.
 *
 * Manual mode: pass `curves` (and optional axis anchors) — VLM is skipped.
 * Auto mode: leave curves empty and set useVlm=true.
 */
export async function extractSingleImage(
  image: File,
  curves?: ExtractCurveSpec[],
  opts?: {
    yAnchors?: Array<{ px: number; value: number; label?: string }>;
    xAnchors?: Array<{ px: number; value: number; label?: string }>;
    xLabels?: string[];
    yTicks?: string[];
    useVlm?: boolean;
  },
): Promise<ExtractSingleImageResponse> {
  const formData = new FormData();
  formData.set("file", image);
  if (curves && curves.length > 0) {
    formData.set("curves", JSON.stringify(curves));
  }
  if (opts?.yAnchors && opts.yAnchors.length > 0) {
    formData.set("y_anchors", JSON.stringify(opts.yAnchors));
  }
  if (opts?.xAnchors && opts.xAnchors.length > 0) {
    formData.set("x_anchors", JSON.stringify(opts.xAnchors));
  }
  if (opts?.xLabels && opts.xLabels.length > 0) {
    formData.set("x_labels", JSON.stringify(opts.xLabels));
  }
  if (opts?.yTicks && opts.yTicks.length > 0) {
    formData.set("y_ticks", JSON.stringify(opts.yTicks));
  }
  formData.set("use_vlm", String(opts?.useVlm ?? (curves && curves.length > 0 ? false : true)));

  const response = await fetch(`${API_BASE_URL}/extract/single-image`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(await errorMessage(response, "图表提取失败"));
  return response.json() as Promise<ExtractSingleImageResponse>;
}

/** Sample the dominant color at a clicked pixel (click-to-pick workflow). */
export async function sampleColorAtPixel(
  image: File,
  x: number,
  y: number,
  radius = 3,
): Promise<{ x: number; y: number; color_hex: string }> {
  const formData = new FormData();
  formData.set("file", image);
  formData.set("x", String(Math.round(x)));
  formData.set("y", String(Math.round(y)));
  formData.set("radius", String(radius));
  const response = await fetch(`${API_BASE_URL}/extract/sample-color`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(await errorMessage(response, "取色失败"));
  return response.json();
}

/** Upload a PDF and run the full extraction pipeline (returns a job id). */
export async function extractPdf(
  file: File,
  opts?: { maxPages?: number; useVlm?: boolean; curveColorHex?: string },
): Promise<ExtractPdfResponse> {
  const formData = new FormData();
  formData.set("file", file);
  if (opts?.maxPages) formData.set("max_pages", String(opts.maxPages));
  formData.set("use_vlm", String(opts?.useVlm ?? true));
  if (opts?.curveColorHex) formData.set("curve_color_hex", opts.curveColorHex);
  const response = await fetch(`${API_BASE_URL}/extract/pdf`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(await errorMessage(response, "PDF 提取失败"));
  return response.json() as Promise<ExtractPdfResponse>;
}

/** Poll a PDF extraction job's progress. */
export async function getExtractJobStatus(jobId: string): Promise<ExtractJobStatus> {
  return requestJson(`/extract/status/${jobId}`, "查询提取任务失败");
}

/** Fetch full results of a completed extraction job. */
export async function getExtractJobPreview(jobId: string): Promise<ExtractJobPreview> {
  return requestJson(`/extract/preview/${jobId}`, "获取提取结果失败");
}

/** URL for a saved chart-region crop image (for preview thumbnails). */
export function extractRegionImageUrl(jobId: string, pageIndex: number, regionIndex: number): string {
  return `${API_BASE_URL}/extract/preview/${jobId}/image/${pageIndex}/${regionIndex}`;
}

/** Get current VLM configuration. */
export async function getVlmConfig(): Promise<VlmConfig> {
  return requestJson("/config/vlm", "获取 VLM 配置失败");
}

/** Update VLM configuration for the running backend process. */
export async function updateVlmConfig(config: {
  provider?: string;
  model?: string;
  api_key?: string;
  base_url?: string;
}): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/config/vlm`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "更新 VLM 配置失败"));
  return response.json();
}

/** Send a tiny image to the configured VLM to verify vision input is accepted. */
export async function testVlmConfig(): Promise<ModelConnectionTest> {
  return requestJson("/config/vlm/test", "VLM 连通性测试失败", { method: "POST" });
}
