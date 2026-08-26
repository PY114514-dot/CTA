/** Shared type and interface definitions for the research backend API. */

// ---------------------------------------------------------------------------
// Common / shared types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// NAV analysis
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Product identity / strategy
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Multi-product report
// ---------------------------------------------------------------------------

export interface ReportDisclosedMetrics {
  product_id: string;
  product_name: string | null;
  strategy: string | null;
  start_date: string;
  end_date: string;
  cumulative_return: number;
  annualized_return: number;
  maximum_drawdown: number;
  sharpe_ratio?: number | null;
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
  binding_confidence?: number | null;
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

// ---------------------------------------------------------------------------
// OCR
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// External factors
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// FOF library / workbench
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// AI Strategy Attribution (report generation)
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
    relative_mean_return: number;
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
    statistical_stability_label?: string;
    reliability_label?: string;
    carry_data_available?: boolean;
  };
  evidence: string[];
  warnings: string[];
  method_provenance: Record<string, unknown>;
  disclaimer: string;
}

export interface LlmConfigResponse {
  enabled: boolean;
  provider: string;
  model: string;
  api_base_configured: boolean;
  thinking_enabled: boolean;
  max_tokens: number;
}

export interface ModelConnectionTest {
  ok: boolean;
  model: string;
  latency_ms: number;
  detail: string;
}

// ---------------------------------------------------------------------------
// Factor library — quantitative L1 attribution
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

// ---------------------------------------------------------------------------
// Factor library management
// ---------------------------------------------------------------------------

export interface FactorMetaItem {
  name: string;
  display_name: string;
  category: string;
  frequency: "daily" | "weekly" | "monthly";
  description: string;
  params: Record<string, number>;
  cached: boolean;
}

/** Rich detail for one factor (signal rule, LaTeX formula, derivation, code). */
export interface FactorDetail {
  name: string;
  display_name: string;
  category: string;
  frequency: "daily" | "weekly" | "monthly";
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
  frequency: "daily" | "weekly" | "monthly";
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

export interface RiskOverlayMeta {
  profile: "baseline" | "vol_target" | "drawdown_control";
  display_name: string;
  description: string;
  params: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Sector breakdown
// ---------------------------------------------------------------------------

export interface SectorContribution {
  sector: string;
  display_sector: string;
  contribution: number;
  direction: "盈利" | "亏损" | "持平";
}

export interface FactorSectorBreakdown {
  factor_name: string;
  display_name: string;
  period_return: number;
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

// ---------------------------------------------------------------------------
// Chart extraction (CV pixel tracing + VLM)
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
  /** True when the result is a candidate and must be manually calibrated. */
  review_required?: boolean;
  /** Quality-gate identifiers explaining why automatic adoption is blocked. */
  review_reasons?: string[];
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
     next_action?: "review_candidate" | "confirm_product_binding" | "recover_chart_calibration" | "none";
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
  reviewed_nav_count?: number;
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
    layout_product_name?: string | null;
    color_hex: string | null;
    bbox: { left_ratio?: number; top_ratio?: number; right_ratio?: number; bottom_ratio?: number } | null;
    page_number?: number | null;
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
  strategy_disclosure?: StrategyDisclosure | null;
  inception_date: string | null;
  close_date: string | null;
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
  research_workflow?: {
    stage: "rejected" | "needs_identity" | "needs_nav" | "needs_nav_source" | "needs_review" | "needs_confirmation" | "needs_quality_review" | "research_ready";
    label: string;
    next_action: "edit_identity" | "calibrate_nav" | "confirm_identity" | null;
    blocking_reasons: string[];
    completed_steps: Record<string, boolean>;
  };
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
    disclosed_annualized_return: number | null;
    disclosed_maximum_drawdown: number | null;
    disclosed_sharpe_ratio: number | null;
  };
  machine_nav_review?: {
    status: "machine_reviewed" | "human_review_required" | "direct_source";
    machine_reviewed: boolean;
    confidence: number | null;
    reasons: string[];
    scope: "可进入初步研究" | "可进入研究" | "需人工复核";
    method: string;
    reviewed_at: string | null;
  };
}

export interface StrategyDisclosure {
  arbitrage_type: "跨期" | "跨品种" | "跨市场" | "期限结构" | "混合" | null;
  markets_or_sectors: string[];
  holding_period: "日内" | "数日" | "周度" | null;
  directional_exposure: "低" | "中" | "高" | null;
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

export interface KbProductUpdate {
  standard_name?: string;
  manager_name?: string | null;
  strategy?: string | null;
  inception_date?: string | null;
  close_date?: string | null;
  nav_frequency?: string | null;
  status?: string;
  notes?: string | null;
  strategy_disclosure?: StrategyDisclosure | null;
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
  draft_id: string;
  allocations: Array<{ product_id: string; product_name: string; weight: number; score: number; rationale: string }>;
  exclusions: Array<{ product_id: string; product_name: string; reasons: string[] }>;
  risk_warnings: string[];
  portfolio: { nav: Array<{ date: string; nav: number }>; metrics: { cumulative_return: number; annualized_return: number; annualized_volatility: number; sharpe_ratio: number | null; maximum_drawdown: number } } | null;
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
  name?: string;
  product_name?: string;
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
  allocation_count?: number;
  portfolio_metrics?: {
    annualized_return: number;
    annualized_volatility: number;
    sharpe_ratio: number | null;
    maximum_drawdown: number;
  } | null;
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
    deterministic_score?: { total: number; components: Record<string, number>; rules?: string };
    workflow?: { stage: string; tracking_enabled: boolean };
    llm_interpretation?: string;
    portfolio?: {
      nav: Array<{ date: string; nav: number }>;
      metrics: { annualized_return: number; annualized_volatility: number; sharpe_ratio: number | null; maximum_drawdown: number };
    };
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

// ---------------------------------------------------------------------------
// CTA dynamic attribution (phases A–D)
// ---------------------------------------------------------------------------

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
    return_contribution: Array<{ factor_name: string; display_name: string; factor_group: string; mean_return_contribution: number }>;
    euler_risk_contribution: Array<{ factor_name: string; display_name: string; factor_group: string; component_risk_contribution: number | null }>;
    return_reconciliation: { mean_product_return: number | null; mean_factor_explained_return: number | null; mean_intercept_return: number | null; mean_residual_return: number | null };
  };
  factor_bundle: {
    bundle_version: string;
    model_version: string;
    factors: Array<{
      name: string;
      label: string;
      status: "available" | "not_covered";
      source: string;
      calculation: string;
    }>;
  };
  baseline: { r_squared: number; adj_r_squared: number; annualized_alpha_candidate: number; out_of_sample: { r_squared?: number | null; positive_segment_fraction?: number | null; segments?: Array<{ segment: number; test_start_date: string; test_end_date: string; observations: number; r_squared: number | null; directional_accuracy: number }> }; validation: { status: "applicable" | "observe_only" | "not_applicable"; reason: string; segments_passed: number; segments_required: number }; warnings: string[] };
  candidate_factor_gate: {
    candidate_name: string;
    status: "admitted" | "candidate" | "insufficient";
    admitted: boolean;
    segments: Array<{
      segment: number;
      train_end_date: string;
      test_start_date: string;
      test_end_date: string;
      baseline_oos_r2: number;
      candidate_oos_r2: number;
      incremental_oos_r2: number;
    }>;
    summary: { evaluated_segments?: number; mean_incremental_oos_r2?: number | null; median_incremental_oos_r2?: number | null; positive_increment_fraction?: number | null; mean_candidate_oos_r2?: number | null };
    rule: string;
    conclusion: string;
  } | null;
  model_applicability: ModelApplicability;
  warnings: string[];
}

export interface ModelApplicability {
  product_id: string;
  product_name: string;
  reviewed_nav_count: number;
  product_pool: "commodity_cta" | "equity_quant" | "options_volatility" | "mixed_or_unconfirmed";
  classification_source: "confirmed_strategy" | "initial_name_label" | "insufficient_evidence";
  status: "applicable" | "observe_only" | "ready_for_baseline" | "not_covered" | "insufficient_data";
  factor_contract: string | null;
  factor_coverage: {
    label: string;
    status: "available" | "not_covered";
    source: string | null;
    data_version: string | null;
    factors: Array<{ name: string; label: string; status: "available" | "not_covered"; observations: number }>;
  } | null;
  reason: string;
  next_step: string;
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
  scenario_checks: Array<{
    key: string;
    title: string;
    description: string;
    status: "available" | "insufficient";
    stable: boolean;
    conclusion: string;
    evaluated_segments: number;
    positive_fraction: number | null;
    mean_r2_delta: number | null;
    details: CtaPhaseDResponse["nonlinear_increment"] | null;
  }>;
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
      standardization?: string;
      baseline_model?: string;
      augmented_model?: Record<string, number | string>;
      interaction_name?: string;
      linear_model?: string;
      nonlinear_model?: Record<string, number | string>;
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
      interaction_coefficient?: number | null;
      predictions_available: boolean;
    }>;
    summary: {
      segment_count: number;
      evaluated_delta_count: number;
      mean_r2_delta: number | null;
      median_r2_delta: number | null;
      positive_delta_fraction: number | null;
      stable_improvement: boolean;
      interaction_coefficient_mean?: number | null;
      interaction_coefficient_median?: number | null;
      interaction_coefficient_sign_consistency?: number | null;
      rejection_reason?: string | null;
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
  evidence_package: {
    purpose: "reviewable_attribution_evidence";
    data_lineage: {
      reviewed_observation_count: number;
      date_range: { start: string; end: string };
      source_linked_observation_count: number;
      source_unlinked_observation_count: number;
      source_files: Array<{ file_id: string; filename: string; file_hash: string; version: number; report_period: string | null }>;
    };
    claims: Array<{
      claim: string;
      confidence: "low" | "medium" | "high";
      supporting_evidence: string[];
      counter_evidence: string[];
    }>;
    llm_handoff: { images_included: false; image_policy: string; forbidden_use: string[] };
  };
  results: CtaAttributionResult;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// CTA ranking
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
    factor_exposures?: Record<string, number>;
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
  raw_score: number | null;
  adjusted_score: number | null;
  metric_count: number;
  covered_metric_count: number;
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

export interface CtaPortfolioPositionPayload {
  product_id: string;
  weight: number;
}

export interface CtaProductScoreRequestPayload {
  products: Array<{
    product_id: string;
    product_name: string;
    nav_points: NavPoint[];
    frequency: DataFrequency;
    strategy?: string | null;
    factor_exposures?: Record<string, number>;
  }>;
  as_of_date?: string;
  annual_risk_free_rate?: number;
  market_series?: Array<{
    series_id: string;
    series_name: string;
    points: Array<{ observation_date: string; return_value: number }>;
  }>;
  current_portfolio?: CtaPortfolioPositionPayload[];
  candidate_product_id?: string;
  candidate_weight?: number;
}

export interface CtaConfidenceScore {
  score: number;
  band: "high" | "medium" | "low";
  components: Record<string, number | null>;
  weights: Record<string, number>;
  method: string;
  warnings: string[];
}

export interface CtaQualityHistoryPoint {
  as_of_date: string;
  quality_score: number | null;
  quality_status: string;
  dimension_scores: Record<string, number>;
  score_delta: number | null;
  change_reasons: Array<{ dimension: string; delta: number | null; reason: string }>;
  warnings: string[];
}

export interface CtaAttributionQuality {
  score: number;
  band: "high" | "medium" | "low";
  components: Record<string, number | null>;
  weights: Record<string, number>;
  method: string;
  warnings: string[];
}

export interface CtaRegimeSplit {
  method: string;
  regime_factor: string;
  window: number;
  trending_share: number;
  trending: { observations: number; r_squared: number; annualized_alpha: number };
  choppy: { observations: number; r_squared: number; annualized_alpha: number };
  augmented: { r_squared: number | null; r_squared_gain: number | null };
}

export interface CtaAttributionDetail {
  status: "available" | "unavailable";
  inference_method?: string;
  start_date?: string;
  end_date?: string;
  n_observations?: number;
  r_squared?: number;
  adj_r_squared?: number;
  annualized_alpha?: number;
  alpha_t_stat?: number;
  alpha_p_value?: number;
  alpha_bootstrap_ci?: { low: number | null; high: number | null };
  factors?: Array<{
    name: string;
    display_name: string;
    beta: number;
    t_stat: number;
    p_value: number;
    significant: boolean;
    factor_group: string;
    contribution_pct: number;
    mean_return_contribution: number;
  }>;
  factor_group_mean_return_contributions?: Record<string, number>;
  return_reconciliation?: { mean_product_return: number | null; mean_factor_explained_return: number | null; mean_intercept_return: number | null; mean_residual_return: number | null };
  out_of_sample?: Record<string, unknown>;
  regime?: CtaRegimeSplit | null;
  attribution_quality?: CtaAttributionQuality | null;
  warnings: string[];
}

export interface CtaDrawdownPathPoint {
  date: string;
  drawdown: number;
}

export interface CtaReturnAnalysis {
  status: "available" | "insufficient";
  method: string;
  observation_count: number;
  start_date?: string;
  end_date?: string;
  metrics: {
    cumulative_return: number | null;
    annualized_return: number | null;
    annualized_volatility: number | null;
    positive_period_ratio: number | null;
    gain_loss_asymmetry: number | null;
    best_period_return: number | null;
    worst_period_return: number | null;
    rolling_13_period_return: number | null;
    rolling_26_period_return: number | null;
    rolling_52_period_return: number | null;
    maximum_drawdown: number | null;
    current_drawdown: number | null;
    longest_drawdown_duration_periods: number | null;
    average_recovery_periods: number | null;
    drawdown_episode_count: number;
    recovery_completed: boolean;
  };
  calendar_year_returns: Record<string, number>;
  drawdown_path: CtaDrawdownPathPoint[];
  warnings: string[];
}

export interface CtaFactorRiskExposure {
  name: string;
  display_name: string;
  beta: number;
  t_stat: number;
  significant: boolean;
  risk_contribution: number | null;
  risk_contribution_pct: number | null;
}

export interface CtaRiskExposure {
  status: "available" | "insufficient";
  factor_exposure_status: "available" | "unavailable";
  method: string;
  statistical: {
    annualized_volatility: number | null;
    downside_deviation: number | null;
    var_95: number | null;
    var_99: number | null;
    cvar_95: number | null;
    skewness: number | null;
    excess_kurtosis: number | null;
    worst_period_return: number | null;
    maximum_drawdown: number | null;
  };
  factor_exposures: CtaFactorRiskExposure[];
  factor_group_risk_contributions: Record<string, number>;
  concentration: { hhi: number | null; normalized_hhi: number | null };
  systematic_variance_share: number | null;
  unexplained_variance_share: number | null;
  residual_annual_volatility: number | null;
  warnings: string[];
}

export interface CtaScoreComponentScore {
  score: number | null;
  band: "high" | "medium" | "low" | null;
  status: string;
  components: Record<string, number | null>;
  weights: Record<string, number>;
  method: string;
  warnings: string[];
}

export interface CtaProductScoreSummary {
  quality_score: number | null;
  quality_status: string;
  confidence_score: number;
  confidence_band: "high" | "medium" | "low";
  attribution_quality_score: number | null;
  attribution_quality_band: "high" | "medium" | "low" | null;
  return_score: number | null;
  return_band: "high" | "medium" | "low" | null;
  risk_score: number | null;
  risk_band: "high" | "medium" | "low" | null;
  headline: {
    annualized_return: number | null;
    cumulative_return: number | null;
    maximum_drawdown: number | null;
    annualized_volatility: number | null;
    var_95: number | null;
    r_squared: number | null;
    oos_r_squared: number | null;
    annualized_alpha: number | null;
    alpha_t_stat: number | null;
    trending_regime_r_squared: number | null;
    choppy_regime_r_squared: number | null;
  };
}

export interface CtaProductScoreItem {
  product_id: string;
  product_name: string;
  summary: CtaProductScoreSummary;
  detail: {
    quality_explanation: {
      model_version: string;
      dimension_weights: Record<string, number>;
      dimensions: CtaRankingDimensionScore[];
      warnings: string[];
    };
    confidence: CtaConfidenceScore;
    returns_analysis: CtaReturnAnalysis;
    risk_exposure: CtaRiskExposure;
    return_score: CtaScoreComponentScore;
    risk_score: CtaScoreComponentScore;
    attribution: CtaAttributionDetail;
    quality_history: CtaQualityHistoryPoint[];
  };
}

export interface CtaAllocationScore {
  status: "available" | "not_available" | "insufficient";
  score: number | null;
  reason?: string;
  candidate_product_id?: string;
  candidate_weight?: number;
  current_portfolio?: CtaPortfolioPositionPayload[];
  metrics: Record<string, number | null>;
  components: Record<string, number>;
  component_weights?: Record<string, number>;
  warnings: string[];
}

export interface CtaProductScoreResponse {
  model_version: string;
  quality_model_version: string;
  as_of_date: string;
  nav_fingerprint?: string;
  universe_size?: number;
  products: CtaProductScoreItem[];
  allocation: CtaAllocationScore;
  factor_bundle: { bundle_version: string; factors: Array<Record<string, unknown>> };
  warnings: string[];
}

export interface CtaProductScoreListItem {
  product_id: string;
  product_name: string;
  summary: CtaProductScoreSummary;
}

export interface CtaProductScoreListResponse {
  model_version: string;
  quality_model_version: string;
  as_of_date: string;
  nav_fingerprint?: string;
  universe_size?: number;
  products: CtaProductScoreListItem[];
  warnings: string[];
}

export interface CtaProductScoreSnapshot {
  snapshot_id: string;
  created_at: string | null;
  report: CtaProductScoreListResponse;
}

export interface CtaFamaReadinessRequestPayload {
  products: Array<{
    product_id: string;
    frequency: DataFrequency;
    nav_points: NavPoint[];
    source_group: string;
    exit_date?: string | null;
    point_in_time_verified: boolean;
  }>;
  exit_history_complete: boolean;
  survivorship_audit_passed: boolean;
  backfill_audit_passed: boolean;
  same_source_deduplicated: boolean;
  oos_state_segments: number;
}

export interface CtaFamaReadinessResponse {
  status: "ready_for_research" | "blocked";
  production_ready: false;
  policy: "research_design_only";
  common_frequency: DataFrequency | null;
  gates: Record<string, {
    status: "passed" | "failed";
    observed: Record<string, unknown>;
    required: Record<string, unknown>;
    reason: string;
  }>;
  warnings: string[];
}

export interface CtaFamaAuditState {
  global: {
    exit_history_complete: boolean;
    survivorship_audit_passed: boolean;
    backfill_audit_passed: boolean;
    same_source_deduplicated: boolean;
    oos_state_segments: number;
    reviewed_by: string | null;
    updated_at: string | null;
  };
  products: Array<{
    product_id: string;
    source_group: string | null;
    point_in_time_verified: boolean;
    reviewed_by: string | null;
    updated_at: string | null;
  }>;
}

// ---------------------------------------------------------------------------
// Analysis History
// ---------------------------------------------------------------------------

export interface HistoryRecordPayload {
  /** Stable client ID so later delete requests target the same backend row. */
  id: string;
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
  source_text?: string;
  strategy_profile?: Record<string, unknown>;
  ai_report?: Record<string, unknown>;
}

export interface HistoryRecordResponse extends HistoryRecordPayload {
  saved_at: string;
}
