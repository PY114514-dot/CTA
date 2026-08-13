/** HTTP client and API contracts for the research backend. */

export type DataFrequency = "daily" | "weekly" | "monthly";
export type ImageExtractionFrequency = "auto" | DataFrequency;
export type StrategyConfirmation = "auto" | "commodity_cta" | "equity_cta" | "mixed";

export interface NavPoint {
  observation_date: string;
  net_asset_value: number;
}

export interface PerformanceMetrics {
  cumulative_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number | null;
  maximum_drawdown: number;
  calmar_ratio: number | null;
}

export interface NavAnalysisResponse {
  metrics: PerformanceMetrics;
  periodic_returns: number[];
  annualization_factor: number;
}

export interface NavImageDigitizationResponse {
  nav_points: NavPoint[];
  confidence: number;
  image_width: number;
  image_height: number;
  ocr_text: string | null;
  warnings: string[];
  extraction_frequency: DataFrequency;
  /** CV trace coordinates normalized to the uploaded source image. */
  candidate_curve: Array<{ x_ratio: number; y_ratio: number }>;
}

export interface StrategyEvidence {
  label: string;
  confidence: number;
  evidence: string[];
}

export interface ProductImageRecognitionResponse {
  product_name_candidate: string | null;
  confidence: number;
  ocr_text: string | null;
  warnings: string[];
}

export interface ProductStrategyProfileResponse {
  product_name: string;
  strategy_hypothesis: StrategyEvidence;
  factor_hypotheses: StrategyEvidence[];
  futures_categories: StrategyEvidence[];
  disclaimer: string;
}

export interface ReportDisclosedMetrics {
  product_id: string;
  product_name: string | null;
  strategy: string | null;
  start_date: string;
  end_date: string;
  cumulative_return: number;
  annualized_return: number;
  maximum_drawdown: number;
  maximum_drawdown_disclosed?: boolean;
}

export interface ReportProductIdentity {
  product_name: string | null;
  manager_name: string | null;
  strategy: string | null;
  confidence: number;
  method: string;
}

export interface ReportCurveCandidate {
  curve_index: number;
  product_id: string | null;
  left_ratio: number;
  top_ratio: number;
  right_ratio: number;
  bottom_ratio: number;
  color_hex?: string | null;
  legend_label?: string | null;
  binding_confidence?: number;
  binding_evidence?: string[];
}

export interface ReportProductCard {
  product_id: string;
  product_name: string | null;
  left_ratio: number;
  top_ratio: number;
  right_ratio: number;
  bottom_ratio: number;
  confidence: number;
  evidence: string[];
}

export interface ReportLegendItem {
  label: string;
  color_hex?: string | null;
  left_ratio: number;
  top_ratio: number;
  right_ratio: number;
  bottom_ratio: number;
  confidence: number;
}

export interface ReportCurveBinding {
  curve_index: number;
  product_id?: string | null;
  status: "matched" | "unmatched" | "needs_confirmation";
  confidence: number;
  evidence: string[];
}

export interface MultiProductReportResponse {
  disclosed_metrics: ReportDisclosedMetrics[];
  product_curve_candidates: ReportCurveCandidate[];
  product_cards?: ReportProductCard[];
  legend_items?: ReportLegendItem[];
  curve_bindings?: ReportCurveBinding[];
  product_identity?: ReportProductIdentity;
  warnings: string[];
}

export interface OcrRegionResponse {
  text: string | null;
}

export interface PaddleOcrPage {
  page_number: number;
  markdown: string;
}

export interface PaddleOcrDocumentResponse {
  pages: PaddleOcrPage[];
  model: string;
  warnings: string[];
}

export interface ExternalFactorObservation {
  as_of_date: string;
  source: string;
  factor_id: string;
  source_factor_name: string;
  horizon: "1w" | "1m" | "3m" | "6m" | "1y";
  return_pct: string;
  revision: string;
  verified: string;
  extraction_method: string;
}

/** Same-origin proxy avoids browser CORS and direct-port drift in local development. */
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

export interface FofMaterialEvidence {
  evidence_id: string;
  material_id: string;
  claim_type: string;
  claim_value: string;
  location_hint: string | null;
  confidence: number;
  verification_status: "pending" | "verified" | "rejected";
}

export interface FofMaterialResearchResponse {
  task_id: string;
  material_id: string;
  product_candidates: Array<{
    product_name: string;
    top_ratio: number;
    bottom_ratio: number;
    metrics: Record<string, string>;
    confidence: number;
  }>;
  evidence: FofMaterialEvidence[];
  report: string;
}

export interface FofLibraryProduct {
  product_id: string;
  name: string;
  manager_name: string | null;
  strategy: string | null;
  verification_status: "pending" | "verified" | "rejected";
  material_count: number;
  evidence_count: number;
}

async function errorMessage(response: Response, fallback: string): Promise<string> {
  const payload: unknown = await response.json().catch(() => undefined);
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => typeof item === "object" && item !== null && "msg" in item ? String((item as { msg: unknown }).msg) : "")
        .filter(Boolean);
      if (messages.length) return messages.join("；");
    }
  }
  return fallback;
}

async function requestJson<T>(path: string, fallback: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) throw new Error(await errorMessage(response, fallback));
  return response.json() as Promise<T>;
}

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

export async function extractMultiProductReport(image: File): Promise<MultiProductReportResponse> {
  const formData = new FormData();
  formData.set("image", image);
  const response = await fetch(`${API_BASE_URL}/report/extract-multiple`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "多产品报告识别失败"));
  return response.json() as Promise<MultiProductReportResponse>;
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

/** Parse a PDF or image with the server-side PaddleOCR-VL integration. */
export async function extractDocumentWithPaddleOcr(document: File): Promise<PaddleOcrDocumentResponse> {
  const formData = new FormData();
  formData.set("document", document);
  const response = await fetch(`${API_BASE_URL}/document/paddleocr`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "PaddleOCR 文档解析失败"));
  return response.json() as Promise<PaddleOcrDocumentResponse>;
}

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

// ---------------------------------------------------------------------------
// AI Strategy Attribution Analysis
// ---------------------------------------------------------------------------

export interface StrategyClassifyResponse {
  strategy_type: string;
  confidence_pct: number;
  confidence_label: string;
  evidence: string[];
  correlations: Record<string, number>;
  details: Record<string, unknown>;
}

export interface FactorExposureItem {
  factor_name: string;
  factor_label: string;
  exposure_beta: number;
  correlation: number;
  t_statistic: number;
  p_value: number;
  confidence_pct: number;
  confidence_label: string;
  /** Rolling out-of-sample direction agreement; not a trading win rate. */
  out_of_sample_hit_rate: number | null;
  hit_rate_observations: number;
  hit_rate_p_value: number | null;
  evidence: string[];
  rolling_exposure: number[];
}

export interface FactorAnalysisResponse {
  factors: FactorExposureItem[];
  r_squared: number;
  adj_r_squared: number;
  rolling_r_squared: number[];
  evidence: string[];
  warnings: string[];
  collinearity: {
    factor_names: string[];
    correlation_matrix: Record<string, Record<string, number>>;
    covariance_matrix: Record<string, Record<string, number>>;
    condition_number: number | null;
    max_abs_correlation: number | null;
    high_correlation_pairs: string[];
    ridge_applied: boolean;
    ridge_alpha: number | null;
  };
  risk_profile: {
    downside_betas: Record<string, number | null>;
    tail_correlations: Record<string, number | null>;
    regime_returns: Record<string, {
      downside_periods: number;
      product_mean_return_when_factor_down: number | null;
      upside_periods: number;
      product_mean_return_when_factor_up: number | null;
    }>;
    variance_contributions_pct: Record<string, number | null>;
    principal_components: Array<{
      component: number;
      explained_variance_pct: number;
      product_loading: number;
      dominant_factors: Array<{ factor_name: string; factor_label: string; loading: number }>;
    }>;
    warnings: string[];
  };
  charts_data?: {
    factor_explainability?: {
      series: Array<{
        date: string;
        product_return: number;
        predicted_return: number;
        product_index: number;
        predicted_index: number;
        residual: number;
        factor_contributions: Record<string, number>;
      }>;
      factor_labels: Record<string, string>;
      description?: string;
    };
  };
  method_provenance?: Record<string, unknown>;
}

export interface VarietyCandidateItem {
  symbol: string;
  name: string;
  sector: string;
  probability_pct: number;
  correlation: number;
  lasso_coefficient: number;
  rolling_hit_rate: number;
  evidence: string[];
}

export interface SectorExposureItem {
  sector: string;
  color: string;
  total_probability: number;
  top_variety: string;
  variety_count: number;
}

export interface VarietyIdentifyResponse {
  top_varieties: VarietyCandidateItem[];
  sector_exposure: SectorExposureItem[];
  method_stability: number;
  evidence: string[];
  warnings: string[];
}

export interface ReportGenerateResponse {
  strategy_summary: string;
  structured: {
    nav_profile?: {
      point_count: number;
      start_date: string;
      end_date: string;
      frequency: string;
      cumulative_return: number;
      annualized_return: number | null;
      annualized_volatility: number | null;
      maximum_drawdown: number;
    };
    classification?: StrategyClassifyResponse;
    factors?: FactorAnalysisResponse;
    varieties?: VarietyIdentifyResponse;
    external_factors?: {
      available: boolean;
      data_role?: string;
      latest_as_of_date?: string;
      selected_as_of_date?: string | null;
      latest_available_as_of_date?: string | null;
      factor_count?: number;
      observation_count?: number;
      horizons?: string[];
      verified?: boolean;
      verified_rows?: number;
      product_start?: string | null;
      product_end?: string | null;
      date_relation?: string;
      usable_for_regression?: boolean;
      warnings?: string[];
    };
    market_reference?: {
      available: boolean;
      source: string;
      start_date: string;
      end_date: string;
      frequency: DataFrequency;
      references: Record<string, number>;
      data_role: string;
      note: string;
    };
    deep_attribution?: DeepAttributionResponse;
  };
  charts_data: {
    factor_radar?: { indicators: string[]; values: number[] };
    rolling_r_squared?: number[];
    rolling_exposure?: Record<string, number[]>;
    variety_bar?: { names: string[]; probabilities: number[]; title?: string };
    sector_pie?: { name: string; value: number; color: string }[];
    correlations?: Record<string, number>;
    product_nav_curve?: Array<{ date: string; nav: number; normalized_nav: number | null }>;
    factor_explainability?: {
      series: Array<{
        date: string;
        product_return: number;
        predicted_return: number;
        product_index: number;
        predicted_index: number;
        residual: number;
        factor_contributions: Record<string, number>;
      }>;
      factor_labels: Record<string, string>;
      description?: string;
    };
    chart_notes?: { frequency?: string; factor_alignment?: string };
  };
  disclaimer: string;
  generated_at: string;
  engine: string;
  warnings: string[];
  method_provenance?: Record<string, {
    method?: string;
    model?: string | null;
    data_source?: string;
    data_frequency?: string;
    strategy_scope?: string;
    nav_fingerprint?: string;
    point_count?: number;
    frequency?: string;
    detail?: string;
  }>;
}

export interface DeepAttributionResponse {
  asset_class: {
    most_likely?: string;
    most_likely_label?: string;
    confidence_pct?: number;
    requested_scope?: string;
    candidates?: Array<{
      asset_class: string;
      label: string;
      candidate_probability_pct: number;
      score_pct: number;
      direction: string;
      correlation: number;
      stability_pct: number;
      evidence: string[];
    }>;
    interpretation?: string;
  };
  sector_exposures: Array<{
    sector: string;
    label: string;
    candidate_probability_pct: number;
    score_pct: number;
    direction: string;
    correlation: number;
    standardized_coefficient: number;
    stability_pct: number;
    evidence: string[];
    counter_evidence: string[];
    interpretation: string;
  }>;
  strategy_fingerprints: Array<{
    strategy: string;
    label: string;
    evidence_score_pct: number;
    direction: string;
    stability_pct: number;
    evidence: string[];
    counter_evidence: string[];
    status: string;
    interpretation: string;
  }>;
  state_analysis: Array<{
    state: string;
    periods: number;
    product_mean_return: number;
    product_positive_rate: number;
    market_mean_return: number;
    interpretation: string;
  }>;
  diagnostics: {
    observation_count?: number;
    proxy_count?: number;
    factor_count?: number;
    linear_model?: string;
    nonlinear_model?: string;
    linear_oos_r2?: number | null;
    nonlinear_oos_r2?: number | null;
    nonlinear_uplift?: number | null;
    mean_bootstrap_sign_consistency?: number;
    reliability_label?: string;
    carry_data_available?: boolean;
  };
  evidence: string[];
  warnings: string[];
  method_provenance: Record<string, unknown>;
  disclaimer: string;
}

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

export interface LlmConfigResponse {
  enabled: boolean;
  provider: string;
  model: string;
  api_base_configured: boolean;
}

export interface ModelConnectionTest {
  ok: boolean;
  model: string;
  latency_ms: number;
  detail: string;
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
// Factor Library — Quantitative L1 Attribution
// ---------------------------------------------------------------------------

export interface FactorBetaItem {
  name: string;
  display_name: string;
  beta: number;
  std_error: number;
  t_stat: number;
  p_value: number;
  contribution_pct: number;
  significant: boolean;
  factor_group: string;
  ordinary_std_error: number;
  bootstrap_ci_low: number | null;
  bootstrap_ci_high: number | null;
}

export interface RollingSnapshot {
  date: string;
  r_squared: number;
  betas: Record<string, number>;
}

export interface FactorAttributionResponse {
  intercept: number;
  annualized_alpha: number;
  alpha_t_stat: number;
  alpha_p_value: number;
  annualized_alpha_bootstrap_ci_low: number | null;
  annualized_alpha_bootstrap_ci_high: number | null;
  r_squared: number;
  adj_r_squared: number;
  f_statistic: number;
  f_p_value: number;
  n_observations: number;
  factors: FactorBetaItem[];
  residual: {
    annual_vol: number;
    skewness: number;
    kurtosis: number;
  };
  rolling_r_squared: RollingSnapshot[];
  lasso_selected: string[];
  frequency: string;
  start_date: string;
  end_date: string;
  warnings: string[];
  inference_method: string;
  hac_max_lags: number | null;
  diagnostics: {
    ljung_box_pvalue_6: number | null;
    ljung_box_pvalue_12: number | null;
    arch_lm_pvalue_6: number | null;
    jarque_bera_pvalue: number | null;
    residual_autocorrelation_detected: boolean;
    volatility_clustering_detected: boolean;
    non_normality_detected: boolean;
    [key: string]: unknown;
  };
  bootstrap: {
    method?: string;
    reps: number;
    successful_reps: number;
    block_length: number | null;
    seed?: number;
    confidence_level?: number;
  };
  factor_groups: Record<string, string[]>;
  factor_group_contributions: Record<string, number>;
  factor_risk_contributions: Record<string, number | null>;
  joint_hac: {
    statistic: number | null;
    p_value: number | null;
    degrees_of_freedom: number;
    error?: string;
  };
  collinearity: {
    factor_names: string[];
    condition_number: number | null;
    max_abs_correlation: number | null;
    high_correlation_pairs: string[];
    correlation_matrix?: Record<string, Record<string, number>>;
  };
  out_of_sample: {
    method: string;
    train_window: number;
    observations: number;
    r_squared: number | null;
    correlation: number | null;
    tracking_error_annual: number | null;
    warnings?: string[];
  };
}

export interface WeeklyReportSnapshot {
  product_name: string;
  report_date: string;
  factor_contribution: Record<string, number | null>;
  parse_confidence: number;
  warnings: string[];
  [key: string]: unknown;
}

export interface FactorValidationResponse {
  product_name: string;
  report_date: string;
  r_squared: number;
  r_gap_interpretation: string;
  direction_agreement_pct: number;
  comparison_period: { start: string; end: string } | null;
  factor_comparisons: Array<{
    factor_name: string;
    display_name: string;
    l1_beta: number;
    l1_t_stat: number;
    l1_factor_period_return: number | null;
    l1_predicted_contribution: number | null;
    l4_contribution: number;
    direction_match: boolean | null;
    magnitude_ratio: number | null;
  }>;
  summary: string;
  warnings: string[];
}

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

export interface FactorMetaItem {
  name: string;
  display_name: string;
  category: string;
  description: string;
  params: Record<string, number>;
  cached: boolean;
}

/** Rich detail for one factor (signal rule, LaTeX formula, derivation, code). */
export interface FactorDetail {
  name: string;
  display_name: string;
  category: string;
  description: string;
  signal_rule: string;
  formula: string;
  derivation: string;
  params: Record<string, number>;
  code: string;
}

export interface BuildLogEntry {
  time: string;
  message: string;
}

export interface BuildJobResult {
  results: Record<string, { rows: number; start: string | null; end: string | null; status: string; note?: string }>;
  data_summary: Record<string, unknown>;
  errors: Record<string, string>;
}

export interface BuildJobStatus {
  job_id: string;
  status: "running" | "done" | "error";
  progress: number;
  logs: BuildLogEntry[];
  result: BuildJobResult | null;
  error: string | null;
  started_at: string;
  params: { start: string; end: string; factors: string[] | null; use_cache: boolean };
}

export interface FactorPerformanceItem {
  name: string;
  display_name: string;
  category: string;
  annualized_return: number;
  annualized_vol: number;
  sharpe: number;
  max_drawdown: number;
  calmar: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  payoff_ratio: number;
  worst_month: number;
  risk_profile: "baseline" | "vol_target" | "drawdown_control";
  rows: number;
  start: string;
  end: string;
  built_at: string;
}

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
export interface RiskOverlayMeta {
  profile: "baseline" | "vol_target" | "drawdown_control";
  display_name: string;
  description: string;
  params: Record<string, number>;
}

export async function getFactorPerformance(
  riskProfile: RiskOverlayMeta["profile"] = "baseline",
): Promise<{ performance: FactorPerformanceItem[]; risk_overlay: RiskOverlayMeta }> {
  const response = await fetch(`${API_BASE_URL}/factor-library/performance?risk_profile=${riskProfile}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取因子绩效失败"));
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

export interface SectorContribution {
  sector: string;
  display_sector: string;
  contribution: number; // cumulative % over the period
  direction: "盈利" | "亏损" | "持平";
}

export interface FactorSectorBreakdown {
  factor_name: string;
  display_name: string;
  period_return: number; // cumulative % over the period
  start: string;
  end: string;
  sectors: SectorContribution[];
  prose: string;
}

export interface SectorBreakdownResponse {
  start: string;
  end: string;
  factors: FactorSectorBreakdown[];
  count: number;
}

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
// Chart extraction — CV pixel tracing + optional VLM structure reading
// ---------------------------------------------------------------------------

/** A curve the user wants traced (manual spec or VLM-detected). */
export interface ExtractCurveSpec {
  name: string;
  color_hex: string;
  color_name?: string;
  is_benchmark: boolean;
  /** Clicked anchor pixel [x, y] on the curve (optional). */
  anchor_px?: [number, number];
}

/** A single traced data point (pixel → calibrated value). */
export interface ExtractTracedPoint {
  x_px: number;
  y_px: number;
  date: string;
  value: number | null;
}

/** One fully traced curve with calibrated values. */
export interface ExtractTracedCurve {
  name: string;
  color_hex: string;
  is_benchmark: boolean;
  points: ExtractTracedPoint[];
  num_pixels_traced: number;
}

/** Chart structure as read by the VLM (or entered manually). */
export interface ExtractChartStructure {
  curves: Array<Record<string, unknown>>;
  x_ticks: string[];
  y_ticks: string[];
  y_axis_label: string;
  chart_title: string;
  frequency: string;
  y_range?: number[];
  /** VLM drawable chart frame, normalized to 0–1000 source-image coordinates. */
  plot_bbox_1000?: number[];
  has_grid_lines: boolean;
  source: string;
}

/** Plot region in pixel coordinates. */
export interface ExtractPlotArea {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** Full result of extracting one chart image. */
export interface ExtractChartResult {
  structure: ExtractChartStructure | null;
  plot_area: ExtractPlotArea | null;
  plot_area_source?: "vlm" | "cv" | null;
  curves: ExtractTracedCurve[];
  frequency: string;
  confidence: number;
  error: string | null;
  /** VLM located the chart but needs one user click on the product line. */
  needs_color_pick?: boolean;
  /** (x, y) offset of an auto-cropped chart region within the original image. */
  crop_offset?: [number, number];
  /** True only when the backend actually attempted a visual-model request. */
  vlm_attempted?: boolean;
  /** True only when that request returned a parsed chart structure. */
  vlm_succeeded?: boolean;
  vlm_provider?: string | null;
  vlm_model?: string | null;
  vlm_error?: string | null;
  warnings?: string[];
}

/** Response from the single-image extraction endpoint. */
export interface ExtractSingleImageResponse {
  status: string;
  result: ExtractChartResult;
}

/** Response from the PDF extraction endpoint. */
export interface ExtractPdfResponse {
  job_id: string;
  filename: string;
  status: string;
  total_pages: number;
  message: string;
}

/** A detected chart region within a rendered PDF page. */
export interface ExtractPageRegion {
  page_index: number;
  region_index: number;
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
  image_path: string | null;
}

/** Job status while a PDF extraction runs. */
export interface ExtractJobStatus {
  job_id: string;
  filename: string;
  status: string;
  progress: number;
  total_pages: number;
  pages_with_charts: number[];
  num_results: number;
  error: string | null;
}

/** Full preview payload once a PDF extraction job completes. */
export interface ExtractJobPreview {
  job_id: string;
  filename: string;
  status: string;
  total_pages: number;
  pages_with_charts: number[];
  regions: ExtractPageRegion[];
  results: ExtractChartResult[];
  error: string | null;
  progress?: number;
  message?: string;
}

/** VLM provider configuration. */
export interface VlmConfig {
  provider: string;
  model: string;
  api_key_set: boolean;
  base_url: string;
}

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

// ---------------------------------------------------------------------------
// FOF agent runtime
// ---------------------------------------------------------------------------

export type FofRiskProfile = "conservative" | "balanced" | "growth";

export interface FofFundInput {
  fund_id: string;
  fund_name: string;
  nav_points: NavPoint[];
  frequency: DataFrequency;
  strategy?: string;
}

export interface FofCandidateEvaluation {
  fund_id: string;
  fund_name: string;
  eligible: boolean;
  score: number;
  metrics: PerformanceMetrics;
  warnings: string[];
}

export interface FofRecommendationResponse {
  session_id: string;
  evaluations: FofCandidateEvaluation[];
  recommendations: Array<{ fund_id: string; fund_name: string; weight: number; rationale: string }>;
  agent_trace: Array<{ phase: string; action: string; status: string; iteration: number; detail: string }>;
  reflection: { passed: boolean; checks: string[]; warnings: string[] };
  disclaimer: string;
  method_provenance?: Record<string, { method?: string; detail?: string }>;
}

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

// ---------------------------------------------------------------------------
// Knowledge base (P0 data layer)
// ---------------------------------------------------------------------------

export type ConfirmationStatus = "pending" | "confirmed" | "rejected";
export type ParsingStatus = "pending" | "processing" | "completed" | "completed_no_nav" | "failed";
export type ReviewStatus = "pending" | "reviewed" | "rejected";

export interface KbFile {
  id: string;
  filename: string;
  file_hash: string;
  mime_type: string | null;
  size_bytes: number | null;
  source: string | null;
  report_period: string | null;
  version: number;
  parsing_status: ParsingStatus;
  parsing_error: string | null;
  uploaded_at: string | null;
  extraction_audit?: Record<string, unknown> | null;
  workflow?: {
    identity?: "resolved" | "needs_review";
    layout?: "resolved" | "not_found";
    binding?: "resolved" | "needs_review";
    trace?: "candidate_created" | "needs_calibration";
    next_action?: "review_candidate" | "confirm_product_binding" | "recover_chart_calibration";
  };
  workflow_errors?: string[];
  ingestion_context?: {
    material_nature?: string | null;
    product_name_hint?: string | null;
    manager_name_hint?: string | null;
  } | null;
  fragment_count?: number;
  linked_product_ids?: string[];
  linked_product_names?: string[];
  nav_count?: number;
  unbound_fragment_count?: number;
  unbound_curve_count?: number;
  curve_bindings?: Array<{
    fragment_id: string;
    curve_index: number;
    product_id: string | null;
    product_name: string | null;
    binding_status: "matched" | "unmatched" | "needs_confirmation";
    binding_confidence: number;
    binding_evidence: string[];
    legend_label: string | null;
    color_hex: string | null;
    bbox: { left_ratio?: number; top_ratio?: number; right_ratio?: number; bottom_ratio?: number } | null;
  }>;
}

export interface KbBindFileResponse {
  file_id: string;
  product_id: string;
  product_name: string;
  bound_fragments: number;
  created_manual_fragment: boolean;
}

export interface KbProduct {
  id: string;
  standard_name: string;
  manager_name: string | null;
  strategy: string | null;
  inception_date: string | null;
  nav_frequency: string | null;
  status: string;
  confirmation_status: ConfirmationStatus;
  merged_into_id: string | null;
  nav_count: number;
  fact_count: number;
  nav_start: string | null;
  nav_end: string | null;
  reviewed_nav_count: number;
  source_files: string[];
  created_at: string | null;
  fragment_count?: number;
  nav_methods?: string[];
  source_file_ids?: string[];
  /** 1-based evidence page for a PDF chart, keyed by source file ID. */
  source_preview_pages?: Record<string, number>;
  research_ready?: boolean;
  readiness_reason?: string;
  nav_confidence?: number | null;
  /** Evidence confidence band: high (>=85%), medium (60-85%), low (<60%). */
  nav_confidence_band?: "high" | "medium" | "low" | "unknown";
  nav_confidence_requires_review?: boolean;
  nav_quality?: {
    status: "passed" | "blocked";
    blocking: boolean;
    reasons: string[];
    computed_cumulative_return: number | null;
    computed_maximum_drawdown: number | null;
    largest_period_change: number | null;
    disclosed_cumulative_return: number | null;
    disclosed_maximum_drawdown: number | null;
  };
  machine_nav_review?: {
    status: "machine_reviewed" | "human_review_required";
    machine_reviewed: boolean;
    confidence: number | null;
    reasons: string[];
    scope: "可进入初步研究" | "需人工复核";
    method: string;
    reviewed_at: string;
  };
}

export interface KbReviewPageLocation {
  is_pdf: boolean;
  /** 1-based PDF page; null means no defensible performance page was found. */
  page_number: number | null;
  confidence: number;
  reason: string;
  selection_method: "existing_chart_evidence" | "pdf_text_rank" | "original_image" | "not_found";
}

export interface KbProductDetail extends KbProduct {
  close_date: string | null;
  confirmed_by: string | null;
  confirmed_at: string | null;
  notes: string | null;
  aliases: Array<{ id: string; alias: string; alias_type: string }>;
  nav_count: number;
  fact_count: number;
}

export interface KbNavPoint {
  id: string;
  observation_date: string;
  nav: number;
  acc_nav: number | null;
  frequency: string | null;
  source_file_id: string | null;
  review_status: ReviewStatus;
}

export interface KbNavCandidate {
  id: string;
  source_file_id: string;
  source_fragment_id: string | null;
  frequency: DataFrequency | null;
  confidence: number | null;
  status: "pending" | "published" | "discarded";
  point_count: number;
  start_date: string | null;
  end_date: string | null;
  points: Array<{ observation_date: string; nav: number }>;
}

export interface CtaPhaseAResponse {
  product_id: string;
  product_name: string;
  frequency: DataFrequency;
  data_contract: {
    confirmation_status: ConfirmationStatus;
    reviewed_nav_count: number;
    source: string;
    parsing_triggered: false;
  };
  performance_path: {
    observation_count: number;
    start_date: string;
    end_date: string;
    cagr: number;
    maximum_drawdown: number;
    calmar: number | null;
    expected_shortfall_5pct: number;
    drawdown_duration_max_periods: number;
    current_drawdown_duration_periods: number;
    recovery_completed: boolean;
  };
  attribution: {
    factor_exposure: Array<{ factor_name: string; display_name: string; factor_group: string; beta: number; hac_t_stat: number; hac_p_value: number; bootstrap_ci_low: number | null; bootstrap_ci_high: number | null }>;
    return_contribution: Array<{ factor_name: string; display_name: string; factor_group: string; contribution_pct_of_mean_return: number }>;
    euler_risk_contribution: Array<{ factor_name: string; display_name: string; factor_group: string; component_risk_contribution: number | null }>;
  };
  baseline: { r_squared: number; adj_r_squared: number; annualized_alpha_candidate: number; out_of_sample: { r_squared?: number | null }; warnings: string[] };
  warnings: string[];
}

export interface CtaPhaseBRollingWindow {
  window: number;
  status: "available" | "insufficient";
  n_periods: number;
  paths: Array<{ date: string; alpha: number; betas: Record<string, number> }>;
  summary?: Record<string, {
    median_beta: number;
    first_beta: number;
    last_beta: number;
    beta_range: number;
    sign_consistency: number | null;
    sign_change_count: number;
    coefficient_variation: number | null;
  }>;
  alerts?: Array<{ factor_name: string; type: string; severity: string; message: string }>;
  warnings: string[];
}

export interface CtaPhaseBResponse {
  product_id: string;
  product_name: string;
  frequency: DataFrequency;
  data_contract: CtaPhaseAResponse["data_contract"];
  alignment: {
    factor_cache_profile: "baseline";
    factor_names: string[];
    aligned_observation_count: number;
    start_date: string;
    end_date: string;
  };
  rolling_beta: { windows: CtaPhaseBRollingWindow[] };
  kalman_beta: {
    method: string;
    factor_names: string[];
    parameters: { state_variance?: number; observation_variance?: number; smoother_used: false };
    long_run_style_beta: Record<string, number>;
    beta_range: Record<string, number>;
    dynamic_timing_contribution_total: number;
    innovation_root_mean_square: number;
    paths: Array<{
      date: string;
      alpha: number;
      alpha_standard_error: number;
      betas: Record<string, number>;
      beta_standard_errors: Record<string, number>;
      long_run_style_contribution: Record<string, number>;
      dynamic_timing_contribution_by_factor: Record<string, number>;
      dynamic_timing_contribution: number;
      innovation: number;
    }>;
    warnings: string[];
  };
  warnings: string[];
}

export interface CtaPhaseCResponse {
  product_id: string;
  product_name: string;
  frequency: DataFrequency;
  data_contract: CtaPhaseAResponse["data_contract"];
  alignment: CtaPhaseBResponse["alignment"];
  state_proxy: { source: string; market_proxy_factor: string; interpretation: string };
  regime_rules: {
    method: string;
    parameters: Record<string, number | boolean>;
    state_counts: Record<string, number>;
    states: string[];
    features: Array<Record<string, string | number | null>>;
    warnings: string[];
  };
  regime_attribution: {
    method: string;
    parameters: Record<string, number>;
    regimes: Array<{
      state: "normal" | "crisis" | "whipsaw";
      status: "available" | "insufficient";
      observation_count: number;
      conditional_return: number | null;
      positive_rate: number | null;
      expected_shortfall_5pct: number | null;
      conditional_alpha: number | null;
      conditional_alpha_annualized: number | null;
      crisis_alpha_candidate: number | null;
      conditional_beta: Record<string, number> | null;
      conditional_correlation: number | null;
      residual_volatility: number | null;
      warnings: string[];
    }>;
    warnings: string[];
  };
  warnings: string[];
}

export interface CtaPhaseDResponse {
  product_id: string;
  product_name: string;
  frequency: DataFrequency;
  data_contract: CtaPhaseAResponse["data_contract"];
  alignment: CtaPhaseBResponse["alignment"];
  nonlinear_increment: {
    method: string;
    frequency: DataFrequency;
    factor_names: string[];
    status: "available" | "insufficient";
    parameters: {
      train_window: number;
      test_window: number;
      max_segments: number;
      min_segments: number;
      min_r2_uplift: number;
      min_positive_fraction: number;
      training_scheme: string;
      test_scheme: string;
      uses_future_data: false;
      linear_model: string;
      nonlinear_model: Record<string, number | string>;
    };
    segments: Array<{
      segment: number;
      train_start_date: string;
      train_end_date: string;
      test_start_date: string;
      test_end_date: string;
      train_observations: number;
      test_observations: number;
      linear: { observations: number; oos_r2: number | null; correlation: number | null; rmse: number | null; mean_residual: number | null };
      nonlinear: { observations: number; oos_r2: number | null; correlation: number | null; rmse: number | null; mean_residual: number | null };
      r2_delta: number | null;
      correlation_delta: number | null;
      rmse_delta: number | null;
      predictions_available: boolean;
    }>;
    summary: {
      segment_count: number;
      evaluated_delta_count: number;
      mean_r2_delta: number | null;
      median_r2_delta: number | null;
      positive_delta_fraction: number | null;
      stable_improvement: boolean;
      conclusion: string;
      rule: Record<string, number | boolean>;
    };
    sensitivity: {
      window_sensitivity: Array<Record<string, number | string | boolean | null>>;
      threshold_sensitivity: Array<Record<string, number | boolean | null>>;
      model_parameter_sensitivity: Array<Record<string, number | string | boolean | null>>;
      selection_policy: { base_case_is_pre_registered: boolean; selected_from_sensitivity: boolean; sensitivity_is_diagnostic_only: boolean };
      multiple_testing: {
        candidate_scenarios_count: number;
        correction: string;
        raw_one_sided_sign_test_p_value: number | null;
        bonferroni_adjusted_p_value: number | null;
        dsr_applicable: false;
        dsr_note: string;
      };
    };
    warnings: string[];
  };
  warnings: string[];
}

export type CtaAttributionPhase = "phase-a" | "phase-b" | "phase-c" | "phase-d";
export type CtaAttributionResult = CtaPhaseAResponse | CtaPhaseBResponse | CtaPhaseCResponse | CtaPhaseDResponse;

export interface CtaAttributionSnapshotSummary {
  snapshot_id: string;
  created_at: string | null;
  snapshot_type: "cta_dynamic_attribution";
  phase: CtaAttributionPhase;
  model_version: string;
  product_id: string;
  product_name: string;
  as_of_date: string;
  nav_fingerprint: string;
  factor_data_version: string;
}

export interface CtaAttributionSnapshotResponse extends CtaAttributionSnapshotSummary {
  frequency: DataFrequency;
  parameters: Record<string, unknown>;
  factor_data_provenance: Record<string, unknown>;
  results: CtaAttributionResult;
  warnings: string[];
}

export async function kbUploadFile(file: File): Promise<{ id: string; file_hash: string; size_bytes: number; version: number; parsing_status: ParsingStatus }> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/kb/files/upload`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, "文件上传失败"));
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

export async function kbListFiles(parsingStatus?: ParsingStatus): Promise<KbFile[]> {
  const query = parsingStatus ? `?parsing_status=${parsingStatus}` : "";
  const response = await fetch(`${API_BASE_URL}/kb/files${query}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取文件列表失败"));
  return response.json() as Promise<KbFile[]>;
}

export async function kbListProducts(params?: {
  confirmation_status?: ConfirmationStatus;
  strategy?: string;
  search?: string;
}): Promise<KbProduct[]> {
  const query = new URLSearchParams();
  if (params?.confirmation_status) query.set("confirmation_status", params.confirmation_status);
  if (params?.strategy) query.set("strategy", params.strategy);
  if (params?.search) query.set("search", params.search);
  const response = await fetch(`${API_BASE_URL}/kb/products?${query.toString()}`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取产品列表失败"));
  return response.json() as Promise<KbProduct[]>;
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

export interface KbProductUpdate {
  standard_name?: string;
  manager_name?: string | null;
  strategy?: string | null;
  inception_date?: string | null;
  close_date?: string | null;
  nav_frequency?: string | null;
  status?: string;
  notes?: string | null;
}

export async function kbUpdateProduct(productId: string, fields: KbProductUpdate): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "更新产品失败"));
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

export async function kbListNavCandidates(productId: string): Promise<KbNavCandidate[]> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}/nav-candidates`);
  if (!response.ok) throw new Error(await errorMessage(response, "获取候选净值失败"));
  return response.json() as Promise<KbNavCandidate[]>;
}

/** Read-only CTA Phase-A evaluation. The server accepts only confirmed, reviewed NAV. */
export async function getCtaPhaseA(productId: string): Promise<CtaPhaseAResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-a`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 已审核净值归因失败"));
  return response.json() as Promise<CtaPhaseAResponse>;
}

/** Read-only CTA Phase-B dynamic beta evaluation. */
export async function getCtaPhaseB(productId: string): Promise<CtaPhaseBResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-b`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 动态 Beta 归因失败"));
  return response.json() as Promise<CtaPhaseBResponse>;
}

/** Read-only CTA Phase-C observable-regime attribution. */
export async function getCtaPhaseC(productId: string): Promise<CtaPhaseCResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-c`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 状态归因失败"));
  return response.json() as Promise<CtaPhaseCResponse>;
}

/** Read-only Phase-D causal OOS linear/nonlinear increment evaluation. */
export async function getCtaPhaseD(productId: string): Promise<CtaPhaseDResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/products/${productId}/phase-d`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 非线性增量归因失败"));
  return response.json() as Promise<CtaPhaseDResponse>;
}

export async function createCtaAttributionSnapshot(productId: string, phase: CtaAttributionPhase): Promise<{
  snapshot_id: string;
  created_at: string | null;
  snapshot_type: "cta_dynamic_attribution";
  phase: CtaAttributionPhase;
  model_version: string;
  product_id: string;
  as_of_date: string;
  nav_fingerprint: string;
  factor_data_version: string;
  result: CtaAttributionResult;
  idempotent: boolean;
}> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ product_id: productId, phase }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存 CTA 归因快照失败"));
  return response.json();
}

export async function listCtaAttributionSnapshots(productId: string, limit = 20): Promise<CtaAttributionSnapshotSummary[]> {
  const query = new URLSearchParams({ product_id: productId, limit: String(limit) });
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots?${query.toString()}`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA 归因快照失败"));
  return response.json();
}

export async function getLatestCtaPhaseDEvidence(
  productIds: string[],
  asOfDate?: string,
): Promise<Record<string, CtaRankingAttributionEvidencePayload>> {
  const query = new URLSearchParams();
  if (productIds.length > 0) query.set("product_ids", productIds.join(","));
  if (asOfDate) query.set("as_of_date", asOfDate);
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots/latest-phase-d?${query.toString()}`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取 Phase D 冻结证据失败"));
  return response.json() as Promise<Record<string, CtaRankingAttributionEvidencePayload>>;
}

export async function getCtaAttributionSnapshot(snapshotId: string): Promise<CtaAttributionSnapshotResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-attribution/snapshots/${snapshotId}`);
  if (!response.ok) throw new Error(await errorMessage(response, "读取 CTA 归因快照详情失败"));
  return response.json();
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

/** Replace traced candidate values with a user-reviewed calibrated NAV series. */
export async function kbReplaceNavSeries(
  productId: string,
  points: NavPoint[],
  options?: { frequency?: string; sourceFileId?: string },
): Promise<{ product_id: string; saved: number; review_status: "reviewed" }> {
  const response = await fetch(`${API_BASE_URL}/kb/products/${productId}/nav`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: points.map((point) => ({ observation_date: point.observation_date, nav: point.net_asset_value })),
      frequency: options?.frequency,
      source_file_id: options?.sourceFileId,
    }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存校准净值失败"));
  return response.json();
}

// ---------------------------------------------------------------------------
// Agent conversation (P2)
// ---------------------------------------------------------------------------

export interface ChatCitation {
  file_id: string | null;
  filename: string | null;
  page_number: number | null;
  fragment_id: string | null;
  fragment_type: string | null;
  snippet: string | null;
  confidence: number | null;
}

export interface ChatToolCall {
  name: string;
  status: string;
  duration_ms: number | null;
  summary: string | null;
}

export interface ChatAllocationDraft {
  goal: string;
  hard_constraints: string[];
  soft_preferences: string[];
  allocations: Array<{ product_id: string; product_name: string; weight: number; score: number; rationale: string }>;
  exclusions: Array<{ product_id: string; product_name: string; reasons: string[] }>;
  risk_warnings: string[];
  due_diligence_gaps: string[];
  snapshot_id: string | null;
}

export interface ChatMessage {
  role: string;
  content: string;
  citations: ChatCitation[];
  tool_calls: ChatToolCall[];
  products_referenced: Array<{ id: string; name: string; category: string }>;
  data_context: Array<{
    product_id: string;
    name: string;
    nav_count: number;
    start_date: string | null;
    end_date: string | null;
    frequency: string | null;
    source_files: string[];
    fact_count: number;
    review_status: string;
  }>;
  method_provenance?: Record<string, { method?: string; model?: string | null; detail?: string }>;
  allocation_draft?: ChatAllocationDraft | null;
}

export async function kbChat(request: {
  query: string;
  product_ids?: string[];
  session_id?: string;
}): Promise<ChatMessage> {
  const response = await fetch(`${API_BASE_URL}/kb/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "Agent 对话失败"));
  return response.json() as Promise<ChatMessage>;
}

// ---------------------------------------------------------------------------
// Investment committee & outer-loop tracking (P4)
// ---------------------------------------------------------------------------

export type DecisionStatus =
  | "draft"
  | "pending_review"
  | "approved"
  | "vetoed"
  | "superseded";

export interface KbAllocation {
  product_id: string;
  name: string;
  weight: number;
  score: number;
}

export interface KbDecision {
  id: string;
  decision_type: string;
  title: string | null;
  status: DecisionStatus;
  reviewer: string | null;
  veto_reason: string | null;
  created_at: string | null;
}

export interface KbTrackingRecord {
  id: string;
  decision_id: string;
  review_date: string | null;
  periods_elapsed: number;
  expected_return: number | null;
  actual_return: number | null;
  return_deviation: number | null;
  actual_max_drawdown: number | null;
  per_product: Record<string, Record<string, unknown>> | null;
  status: "normal" | "breach" | "insufficient_data";
  breach_reasons: string[] | null;
  triggered_task_id: string | null;
  created_at: string | null;
}

export interface KbReviewTask {
  id: string;
  decision_id: string | null;
  tracking_id: string | null;
  task_type: string;
  title: string;
  description: string | null;
  status: "open" | "in_progress" | "resolved" | "dismissed";
  priority: string;
  assignee: string | null;
  proposal: string | null;
  resolution: string | null;
  resolved_by: string | null;
  resolved_at: string | null;
  created_at: string | null;
}

export interface KbDecisionDetail extends KbDecision {
  content: {
    allocations?: KbAllocation[];
    excluded_count?: number;
    constraints?: Record<string, unknown>;
    expected_metrics?: Record<string, Record<string, number | null>>;
    decision_date?: string;
    ic_comment?: string;
    method_provenance?: Record<string, { method?: string; model?: string | null; detail?: string }>;
  } | null;
  reviewed_at: string | null;
  adjustment: Record<string, unknown> | null;
  snapshot: { id: string; label: string | null; content: Record<string, unknown>; created_at: string | null } | null;
  run: {
    id: string;
    user_query: string;
    phase: string;
    tools_used: string[] | null;
    citations: ChatCitation[] | null;
    answer: string | null;
    created_at: string | null;
  } | null;
  tracking: KbTrackingRecord[];
  review_tasks: KbReviewTask[];
}

export interface KbInvestmentMemo {
  decision_id: string;
  title: string | null;
  snapshot_id: string | null;
  markdown: string;
  generated_at: string;
}

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

// ---------------------------------------------------------------------------
// CODEX CTA ranking
// ---------------------------------------------------------------------------

export interface CtaRankingAttributionEvidencePayload {
  source: string;
  snapshot_id: string;
  model_version: string;
  nav_value_signature: string;
  as_of_date: string;
  observation_count: number;
  status: string;
  evaluated_segments: number;
  minimum_segments: number;
  stable_improvement: boolean;
  sensitivity_stable: boolean;
  selected_from_sensitivity: boolean;
}

export interface CtaRankingRequestPayload {
  products: Array<{
    product_id: string;
    product_name: string;
    nav_points: NavPoint[];
    frequency: DataFrequency;
    strategy?: string | null;
  }>;
  market_series?: Array<{
    series_id: string;
    series_name: string;
    points: Array<{ observation_date: string; return_value: number }>;
  }>;
  as_of_date?: string;
  annual_risk_free_rate?: number;
  model_version?: string;
  attribution_evidence?: Record<string, CtaRankingAttributionEvidencePayload>;
}

export interface CtaRankingDimensionScore {
  dimension: string;
  label: string;
  weight: number;
  raw_score: number;
  adjusted_score: number;
  metric_scores: Record<string, number>;
  metric_values: Record<string, number | null>;
  warnings: string[];
}

export interface CtaRankingProductResult {
  product_id: string;
  product_name: string;
  rank: number | null;
  score: number | null;
  percentile: number | null;
  eligible: boolean;
  status: string;
  nav_start_date: string | null;
  nav_end_date: string | null;
  observation_count: number;
  dimension_scores: Record<string, number>;
  dimensions: CtaRankingDimensionScore[];
  metrics: Record<string, number | null>;
  warnings: string[];
}

export interface CtaRankingResponse {
  model_version: string;
  ranking_only: boolean;
  as_of_date: string;
  market_data_end_date: string | null;
  universe_size: number;
  eligible_count: number;
  dimension_weights: Record<string, number>;
  rankings: CtaRankingProductResult[];
  warnings: string[];
  nav_fingerprint: string;
  attribution_evidence_fingerprint: string;
  method_provenance?: Record<string, unknown>;
}

export interface CtaRankingSnapshotSummary {
  snapshot_id: string;
  label: string | null;
  model_version: string;
  as_of_date: string;
  created_at: string | null;
  nav_fingerprint: string;
  attribution_evidence_fingerprint: string;
  universe_size: number;
  eligible_count: number;
}

export interface CtaRankingSnapshotResponse {
  snapshot_id: string;
  created_at: string | null;
  ranking: CtaRankingResponse;
}

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
