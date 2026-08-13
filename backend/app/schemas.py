"""Pydantic contracts shared by HTTP routes and business services.

Field names use stable English API identifiers. Display labels belong to the
frontend, avoiding coupling API evolution to a single locale.
"""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DataFrequency(StrEnum):
    """Supported disclosure frequencies for a product NAV series."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class StrategyConfirmation(StrEnum):
    """Optional user-confirmed strategy scope for the full analysis report."""

    AUTO = "auto"
    COMMODITY_CTA = "commodity_cta"
    EQUITY_CTA = "equity_cta"
    MIXED = "mixed"


class NetAssetValuePoint(BaseModel):
    """One dated, positive unit-NAV observation."""

    observation_date: date
    net_asset_value: float = Field(gt=0)


class NavAnalysisRequest(BaseModel):
    """Input required to calculate product performance metrics."""

    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    annual_risk_free_rate: float = Field(default=0.015, ge=-1, le=1)

    @model_validator(mode="after")
    def validate_chronological_dates(self) -> "NavAnalysisRequest":
        """Require strictly increasing dates to prevent misleading returns."""
        observation_dates = [point.observation_date for point in self.nav_points]
        if observation_dates != sorted(observation_dates):
            raise ValueError("nav_points must be ordered by observation_date")
        if len(observation_dates) != len(set(observation_dates)):
            raise ValueError("nav_points must not contain duplicate dates")
        return self


class PerformanceMetrics(BaseModel):
    """Metrics calculated under the selected frequency and risk-free assumptions."""

    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float | None
    maximum_drawdown: float
    calmar_ratio: float | None


class NavAnalysisResponse(BaseModel):
    """Performance result plus period returns for frontend visualization."""

    metrics: PerformanceMetrics
    periodic_returns: list[float]
    annualization_factor: int


class ChartCurvePoint(BaseModel):
    """A candidate curve point in normalized coordinates of the source image."""

    x_ratio: float = Field(ge=0, le=1)
    y_ratio: float = Field(ge=0, le=1)


class NavImageDigitizationResponse(BaseModel):
    """Candidate NAV points extracted from a chart image for user review."""

    nav_points: list[NetAssetValuePoint]
    confidence: float = Field(ge=0, le=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    ocr_text: str | None = None
    warnings: list[str] = Field(default_factory=list)
    detected_line: str = "product"
    extraction_frequency: DataFrequency = DataFrequency.WEEKLY
    # Carries the CV trace back to the browser so the user can compare it with
    # the original line.  It is evidence only, not a reviewed NAV series.
    candidate_curve: list[ChartCurvePoint] = Field(default_factory=list)


class FactorReturnSeries(BaseModel):
    """Aligned factor return history supplied to the initial attribution model."""

    factor_name: str = Field(min_length=1, max_length=80)
    periodic_returns: list[float] = Field(min_length=2)


class StrategyFingerprintRequest(BaseModel):
    """Input for an explainable MVP factor-exposure assessment.

    All series must have an equal length and frequency. Market-data ingestion
    will populate factor series automatically in a later development module.
    """

    product_periodic_returns: list[float] = Field(min_length=8)
    factor_return_series: list[FactorReturnSeries] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_series_lengths(self) -> "StrategyFingerprintRequest":
        """Reject factor data that is not aligned with product returns."""
        product_length = len(self.product_periodic_returns)
        invalid_factors = [
            series.factor_name
            for series in self.factor_return_series
            if len(series.periodic_returns) != product_length
        ]
        if invalid_factors:
            raise ValueError(f"factor series length mismatch: {', '.join(invalid_factors)}")
        return self


class FactorExposure(BaseModel):
    """Correlation-based evidence; never a confirmed product position."""

    factor_name: str
    correlation: float
    exposure_strength: str


class StrategyFingerprintResponse(BaseModel):
    """Conservative, explainable initial CTA factor classification."""

    strategy_style: str
    confidence: float = Field(ge=0, le=1)
    factor_exposures: list[FactorExposure]
    disclaimer: str


class ProductImageRecognitionResponse(BaseModel):
    """OCR-derived product identity candidate; users must confirm it."""

    product_name_candidate: str | None
    confidence: float = Field(ge=0, le=1)
    ocr_text: str | None = None
    warnings: list[str] = Field(default_factory=list)


class StrategyEvidence(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]


class ProductStrategyProfileRequest(BaseModel):
    product_name: str = Field(min_length=1, max_length=120)
    source_text: str = Field(default="", max_length=5000)


class ProductStrategyProfileResponse(BaseModel):
    product_name: str
    strategy_hypothesis: StrategyEvidence
    factor_hypotheses: list[StrategyEvidence]
    futures_categories: list[StrategyEvidence]
    disclaimer: str


class ReportDisclosedMetrics(BaseModel):
    product_id: str
    product_name: str | None = None
    strategy: str | None = None
    start_date: date
    end_date: date
    cumulative_return: float
    annualized_return: float
    maximum_drawdown: float
    # Some factsheets publish recent-year return/annualized return but do not
    # publish maximum drawdown.  Keep the numeric field for backwards
    # compatibility while making the absence explicit to callers; a zero in
    # that case must never be treated as a disclosed zero drawdown.
    maximum_drawdown_disclosed: bool = True


class ReportProductIdentity(BaseModel):
    """Reviewable product identity evidence recovered from a report image."""

    product_name: str | None = None
    manager_name: str | None = None
    strategy: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    method: str = "OCR"


class ReportCurveCandidate(BaseModel):
    curve_index: int
    product_id: str | None = None
    left_ratio: float = Field(ge=0, le=1)
    top_ratio: float = Field(ge=0, le=1)
    right_ratio: float = Field(ge=0, le=1)
    bottom_ratio: float = Field(ge=0, le=1)
    color_hex: str | None = None
    legend_label: str | None = None
    layout_product_name: str | None = None
    binding_confidence: float = Field(default=0.0, ge=0, le=1)
    binding_evidence: list[str] = Field(default_factory=list)


class ReportProductCard(BaseModel):
    """A product name/metric block discovered on a multi-product page."""

    product_id: str
    product_name: str | None = None
    left_ratio: float = Field(ge=0, le=1)
    top_ratio: float = Field(ge=0, le=1)
    right_ratio: float = Field(ge=0, le=1)
    bottom_ratio: float = Field(ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class ReportLegendItem(BaseModel):
    label: str = ""
    color_hex: str | None = None
    left_ratio: float = Field(ge=0, le=1)
    top_ratio: float = Field(ge=0, le=1)
    right_ratio: float = Field(ge=0, le=1)
    bottom_ratio: float = Field(ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ReportCurveBinding(BaseModel):
    """Evidence-based association between one curve and one product row."""

    curve_index: int
    product_id: str | None = None
    status: str = "unmatched"
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class MultiProductReportResponse(BaseModel):
    disclosed_metrics: list[ReportDisclosedMetrics]
    product_curve_candidates: list[ReportCurveCandidate]
    product_cards: list[ReportProductCard] = Field(default_factory=list)
    legend_items: list[ReportLegendItem] = Field(default_factory=list)
    curve_bindings: list[ReportCurveBinding] = Field(default_factory=list)
    product_identity: ReportProductIdentity = Field(default_factory=ReportProductIdentity)
    vlm_layout_attempted: bool = False
    vlm_layout_used: bool = False
    # Persisted provenance for the ingestion audit.  "attempted" means an
    # HTTP call was made; configuration alone must never be presented as a
    # successful VLM run.
    vlm_audit: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class OcrRegionResponse(BaseModel):
    """Raw text recognized inside a user-selected image region."""

    text: str | None = None


class PaddleOcrPage(BaseModel):
    """Markdown recovered from one page by PaddleOCR's document parser."""

    page_number: int = Field(ge=1)
    markdown: str


class PaddleOcrDocumentResponse(BaseModel):
    """Reviewable document extraction result; source images remain external."""

    pages: list[PaddleOcrPage]
    model: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Market data endpoints
# ---------------------------------------------------------------------------


class SymbolInfo(BaseModel):
    """One tradeable symbol (index or futures variety) available for analysis."""

    symbol: str
    name: str
    type: str  # "index" | "futures"
    asset_class: str  # "equity" | "commodity" | "financial"
    sector: str | None = None
    exchange: str | None = None


class SymbolListResponse(BaseModel):
    """All available symbols grouped for frontend display."""

    symbols: list[SymbolInfo]
    sectors: list[str]
    sector_colors: dict[str, str]


class MarketDataStatusResponse(BaseModel):
    """Provider availability and loaded data summary."""

    api_available: bool
    api_provider: str = "akshare"
    uploaded_symbols: list[str] = Field(default_factory=list)


class CsvUploadResponse(BaseModel):
    """Result of uploading a CSV market-data file."""

    symbol: str
    rows_loaded: int = Field(ge=0)
    date_range: tuple[date, date] | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis pipeline endpoints
# ---------------------------------------------------------------------------


class StrategyClassifyRequest(BaseModel):
    """Input for strategy-type classification (Step 1 of the pipeline)."""

    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    data_source: str = Field(default="api", pattern="^(api|upload)$")
    # OCR/VLM text or a user-supplied strategy disclosure.  This is evidence
    # only: the classifier must keep it separate from NAV-statistical results.
    strategy_hint: str = Field(default="", max_length=5000)
    strategy_confirmation: StrategyConfirmation = StrategyConfirmation.AUTO

    @model_validator(mode="after")
    def validate_chronological(self) -> "StrategyClassifyRequest":
        dates = [p.observation_date for p in self.nav_points]
        if dates != sorted(dates):
            raise ValueError("nav_points must be ordered by observation_date")
        return self


class StrategyClassifyResponse(BaseModel):
    """Strategy-type classification result with evidence and confidence."""

    strategy_type: str  # commodity CTA | equity/index direction (CTA or quant) | mixed | insufficient_data
    confidence_pct: float = Field(ge=0, le=100)
    confidence_label: str  # "高" | "中" | "低"
    evidence: list[str] = Field(default_factory=list)
    correlations: dict[str, float] = Field(default_factory=dict)
    details: dict = Field(default_factory=dict)
    disclaimer: str = (
        "策略类型判断基于净值与基准指数的统计相关性，仅供参考，不构成投资建议。"
    )


class FactorAnalysisRequest(BaseModel):
    """Input for factor exposure analysis (Step 2 of the pipeline)."""

    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    strategy_type: str = Field(default="insufficient_data")
    data_source: str = Field(default="api", pattern="^(api|upload)$")
    rolling_window: int = Field(default=12, ge=6, le=60)

    @model_validator(mode="after")
    def validate_chronological(self) -> "FactorAnalysisRequest":
        dates = [p.observation_date for p in self.nav_points]
        if dates != sorted(dates):
            raise ValueError("nav_points must be ordered by observation_date")
        return self


class FactorExposureItem(BaseModel):
    """One factor's exposure estimate for the API response."""

    factor_name: str
    factor_label: str
    exposure_beta: float
    correlation: float
    t_statistic: float
    p_value: float
    confidence_pct: float = Field(ge=0, le=100)
    confidence_label: str
    out_of_sample_hit_rate: float | None = Field(default=None, ge=0, le=1)
    hit_rate_observations: int = Field(default=0, ge=0)
    hit_rate_p_value: float | None = Field(default=None, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    rolling_exposure: list[float] = Field(default_factory=list)


class FactorCollinearityDiagnosticsResponse(BaseModel):
    """Interpretability diagnostics for overlapping factor proxies."""

    factor_names: list[str] = Field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    covariance_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    condition_number: float | None = None
    max_abs_correlation: float | None = None
    high_correlation_pairs: list[str] = Field(default_factory=list)
    ridge_applied: bool = False
    ridge_alpha: float | None = None


class FactorRiskProfileResponse(BaseModel):
    """Model-based stress and variance diagnostics for factor proxies."""

    downside_betas: dict[str, float | None] = Field(default_factory=dict)
    tail_correlations: dict[str, float | None] = Field(default_factory=dict)
    regime_returns: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    variance_contributions_pct: dict[str, float | None] = Field(default_factory=dict)
    principal_components: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FactorAnalysisResponse(BaseModel):
    """Factor exposure analysis result."""

    factors: list[FactorExposureItem]
    r_squared: float
    adj_r_squared: float
    rolling_r_squared: list[float] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    collinearity: FactorCollinearityDiagnosticsResponse = Field(
        default_factory=FactorCollinearityDiagnosticsResponse
    )
    risk_profile: FactorRiskProfileResponse = Field(default_factory=FactorRiskProfileResponse)
    # Chart-ready aligned series.  Kept separate from the tabular factor
    # estimates so callers can render the product/factor comparison without
    # reimplementing the date alignment logic.
    charts_data: dict = Field(default_factory=dict)
    # Explicit provenance prevents the UI from presenting every result as
    # generic "AI" output when it was actually produced by CV, VLM, or OLS.
    method_provenance: dict = Field(default_factory=dict)
    disclaimer: str = (
        "因子暴露为统计估计结果，不代表产品实际持仓或策略配置，仅供参考。"
    )


class VarietyIdentifyRequest(BaseModel):
    """Input for futures variety identification (Step 3, commodity CTA only)."""

    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    data_source: str = Field(default="api", pattern="^(api|upload)$")
    top_n: int = Field(default=8, ge=3, le=20)
    rolling_window: int = Field(default=12, ge=6, le=60)

    @model_validator(mode="after")
    def validate_chronological(self) -> "VarietyIdentifyRequest":
        dates = [p.observation_date for p in self.nav_points]
        if dates != sorted(dates):
            raise ValueError("nav_points must be ordered by observation_date")
        return self


class VarietyCandidateItem(BaseModel):
    """One candidate futures variety in the API response."""

    symbol: str
    name: str
    sector: str
    probability_pct: float = Field(ge=0, le=100)
    correlation: float
    lasso_coefficient: float
    rolling_hit_rate: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class SectorExposureItem(BaseModel):
    """Sector-level aggregated exposure."""

    sector: str
    color: str
    total_probability: float
    top_variety: str
    variety_count: int


class VarietyIdentifyResponse(BaseModel):
    """Variety identification result."""

    top_varieties: list[VarietyCandidateItem]
    sector_exposure: list[SectorExposureItem]
    method_stability: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "品种推断为统计候选结果，不代表产品实际持仓，不可作为持仓披露使用。"
    )


# ---------------------------------------------------------------------------
# Report generation & LLM config
# ---------------------------------------------------------------------------


class ReportGenerateRequest(BaseModel):
    """Input for one-click full pipeline report generation."""

    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    data_source: str = Field(default="api", pattern="^(api|upload)$")
    rolling_window: int = Field(default=12, ge=6, le=60)
    top_n_varieties: int = Field(default=8, ge=3, le=20)
    # OCR/VLM text or a user-supplied strategy disclosure passed to the
    # classifier and report writer as non-confirmatory context.
    strategy_hint: str = Field(default="", max_length=5000)
    strategy_confirmation: StrategyConfirmation = StrategyConfirmation.AUTO
    # An explicit reviewer acknowledgement is required to continue past
    # detected NAV outliers.  Keep it on the report request (rather than the
    # classification-only request), because the report endpoint owns the
    # quality gate.
    quality_override_confirmed: bool = False

    @model_validator(mode="after")
    def validate_chronological(self) -> "ReportGenerateRequest":
        dates = [p.observation_date for p in self.nav_points]
        if dates != sorted(dates):
            raise ValueError("nav_points must be ordered by observation_date")
        return self


class ReportGenerateResponse(BaseModel):
    """Full pipeline report with all steps and generated summary."""

    strategy_summary: str
    structured: dict = Field(default_factory=dict)
    charts_data: dict = Field(default_factory=dict)
    disclaimer: str = ""
    generated_at: str = ""
    engine: str = "template"
    warnings: list[str] = Field(default_factory=list)
    method_provenance: dict = Field(default_factory=dict)


class DeepAttributionRequest(BaseModel):
    """Input for the evidence-first nonlinear CTA attribution module."""

    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    strategy_type: str = Field(default="commodity_cta", pattern="^(commodity_cta|equity_quant|mixed)$")
    data_source: str = Field(default="api", pattern="^(api|upload)$")
    top_n: int = Field(default=8, ge=3, le=20)
    bootstrap_samples: int = Field(default=80, ge=20, le=300)

    @model_validator(mode="after")
    def validate_chronological(self) -> "DeepAttributionRequest":
        dates = [point.observation_date for point in self.nav_points]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("nav_points must be strictly ordered without duplicate dates")
        return self


class DeepAttributionResponse(BaseModel):
    """Deep CTA attribution result with explicit statistical limitations."""

    asset_class: dict = Field(default_factory=dict)
    sector_exposures: list[dict] = Field(default_factory=list)
    strategy_fingerprints: list[dict] = Field(default_factory=list)
    state_analysis: list[dict] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    method_provenance: dict = Field(default_factory=dict)
    disclaimer: str = ""


class LlmConfigResponse(BaseModel):
    """Current LLM configuration status (never exposes the API key)."""

    enabled: bool
    provider: str = ""  # e.g. "deepseek", "doubao", "openai"
    model: str = ""
    api_base_configured: bool = False


class LlmConfigUpdateRequest(BaseModel):
    """Update LLM configuration at runtime."""

    api_base: str = Field(default="")
    api_key: str = Field(default="")
    model: str = Field(default="deepseek-v4-flash")


# ---------------------------------------------------------------------------
# Factor attribution
# ---------------------------------------------------------------------------


class FactorAttributionNavPoint(BaseModel):
    """A single dated NAV observation for factor attribution."""

    date: date
    nav: float = Field(gt=0, description="Unit net asset value (positive)")


class FactorAttributionRequest(BaseModel):
    """Input for the L1 quantitative factor attribution endpoint."""

    nav_points: list[FactorAttributionNavPoint] = Field(
        min_length=10,
        description="At least 10 dated NAV observations",
    )
    frequency: DataFrequency = DataFrequency.DAILY
    rolling_window: int | None = Field(
        default=None,
        ge=20,
        description="If set, run rolling regression with this window size",
    )
    factor_names: list[str] | None = Field(
        default=None,
        description="Subset of factors to include (None = all cached)",
    )
    hac_max_lags: int | None = Field(
        default=None,
        ge=0,
        le=60,
        description="Newey-West HAC lag length; omitted uses a frequency-based default",
    )
    bootstrap_reps: int = Field(
        default=1000,
        ge=0,
        le=5000,
        description="Stationary-bootstrap repetitions; zero disables bootstrap",
    )
    bootstrap_block_length: int | None = Field(
        default=None,
        ge=1,
        le=120,
        description="Expected stationary-bootstrap block length",
    )
    random_seed: int = Field(default=20260811, ge=0, le=2**32 - 1)
    oos_train_window: int | None = Field(
        default=None,
        ge=20,
        le=5000,
        description="Initial expanding-window training length for out-of-sample validation",
    )


# ---------------------------------------------------------------------------
# CODEX CTA ranking
# ---------------------------------------------------------------------------


class CtaRankingMarketPoint(BaseModel):
    """One dated periodic return for a market proxy used by the ranker."""

    observation_date: date
    return_value: float = Field(ge=-1)


class CtaRankingMarketSeries(BaseModel):
    """A reviewable market-return proxy aligned to product periodic returns."""

    series_id: str = Field(min_length=1, max_length=80)
    series_name: str = Field(min_length=1, max_length=120)
    points: list[CtaRankingMarketPoint] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_chronological_dates(self) -> "CtaRankingMarketSeries":
        dates = [point.observation_date for point in self.points]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("market series dates must be strictly ordered without duplicates")
        return self


class CtaRankingProductInput(BaseModel):
    """One confirmed product NAV series supplied to the read-only ranker."""

    product_id: str = Field(min_length=1, max_length=80)
    product_name: str = Field(min_length=1, max_length=160)
    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    strategy: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def validate_chronological_dates(self) -> "CtaRankingProductInput":
        dates = [point.observation_date for point in self.nav_points]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("product NAV dates must be strictly ordered without duplicates")
        return self


class CtaRankingAttributionEvidence(BaseModel):
    """Frozen Phase-D evidence that may inform ranking robustness only.

    The evidence is deliberately a compact audit reference rather than the
    full nonlinear model output.  A caller must supply a previously frozen
    result; the ranking service never runs Phase D or treats model fit as a
    return contribution.
    """

    source: Literal["immutable_phase_d_snapshot"]
    snapshot_id: str = Field(min_length=1, max_length=80)
    model_version: str = Field(min_length=1, max_length=80)
    nav_value_signature: str = Field(min_length=1, max_length=128)
    as_of_date: date
    observation_count: int = Field(ge=0)
    status: str = Field(min_length=1, max_length=32)
    evaluated_segments: int = Field(ge=0)
    minimum_segments: int = Field(ge=1)
    stable_improvement: bool
    sensitivity_stable: bool
    selected_from_sensitivity: bool


class CtaRankingRequest(BaseModel):
    """Input for one reproducible CODEX CTA cross-sectional ranking run."""

    products: list[CtaRankingProductInput] = Field(min_length=1, max_length=200)
    market_series: list[CtaRankingMarketSeries] = Field(default_factory=list, max_length=20)
    as_of_date: date | None = None
    annual_risk_free_rate: float = Field(default=0.015, ge=-1, le=1)
    model_version: str = Field(default="codex-cta-score-v1.1", min_length=1, max_length=64)
    attribution_evidence: dict[str, CtaRankingAttributionEvidence] = Field(
        default_factory=dict,
        max_length=200,
    )

    @model_validator(mode="after")
    def validate_product_universe(self) -> "CtaRankingRequest":
        product_ids = [product.product_id for product in self.products]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("product_id values must be unique")
        frequencies = {product.frequency for product in self.products}
        if len(frequencies) > 1:
            raise ValueError("CODEX ranking requires one common product frequency per run")
        series_ids = [series.series_id for series in self.market_series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("market series IDs must be unique")
        unknown_evidence_products = set(self.attribution_evidence) - set(product_ids)
        if unknown_evidence_products:
            raise ValueError("attribution_evidence contains product IDs outside the ranking universe")
        return self


class CtaRankingDimensionScore(BaseModel):
    """Explainable score for one CODEX dimension."""

    dimension: str
    label: str
    weight: float = Field(ge=0, le=100)
    raw_score: float = Field(ge=0, le=100)
    adjusted_score: float = Field(ge=0, le=100)
    metric_scores: dict[str, float] = Field(default_factory=dict)
    metric_values: dict[str, float | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CtaRankingProductResult(BaseModel):
    """One product's rank, score decomposition and audit warnings."""

    product_id: str
    product_name: str
    rank: int | None = None
    score: float | None = Field(default=None, ge=0, le=100)
    percentile: float | None = Field(default=None, ge=0, le=100)
    eligible: bool
    status: str
    nav_start_date: date | None = None
    nav_end_date: date | None = None
    observation_count: int = Field(ge=0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    dimensions: list[CtaRankingDimensionScore] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CtaRankingResponse(BaseModel):
    """Immutable-style result of a read-only CODEX CTA ranking run."""

    model_version: str
    ranking_only: bool = True
    as_of_date: date
    market_data_end_date: date | None = None
    universe_size: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    dimension_weights: dict[str, float]
    rankings: list[CtaRankingProductResult]
    warnings: list[str] = Field(default_factory=list)
    nav_fingerprint: str
    attribution_evidence_fingerprint: str = ""
    method_provenance: dict[str, object] = Field(default_factory=dict)


class CtaRankingSnapshotSummary(BaseModel):
    """Historical weekly ranking metadata shown in the ranking history."""

    snapshot_id: str
    label: str | None = None
    model_version: str
    as_of_date: date
    created_at: str | None = None
    nav_fingerprint: str
    attribution_evidence_fingerprint: str = ""
    universe_size: int = Field(ge=0)
    eligible_count: int = Field(ge=0)


class CtaRankingSnapshotResponse(BaseModel):
    """A persisted immutable CODEX ranking snapshot."""

    snapshot_id: str
    created_at: str | None = None
    ranking: CtaRankingResponse


# ---------------------------------------------------------------------------
# FOF recommendation agent
# ---------------------------------------------------------------------------


class FofRiskProfile(StrEnum):
    """Risk budget used by the explainable FOF scoring policy."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GROWTH = "growth"


class FofFundInput(BaseModel):
    """A reviewable fund record supplied to the FOF recommendation agent."""

    fund_id: str = Field(min_length=1, max_length=80)
    fund_name: str = Field(min_length=1, max_length=120)
    nav_points: list[NetAssetValuePoint] = Field(min_length=2)
    frequency: DataFrequency
    strategy: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def validate_chronological_dates(self) -> "FofFundInput":
        dates = [point.observation_date for point in self.nav_points]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("fund nav_points must be strictly ordered without duplicate dates")
        return self


class FofRecommendationRequest(BaseModel):
    """Input for an auditable, data-supplied FOF candidate recommendation."""

    funds: list[FofFundInput] = Field(min_length=2, max_length=50)
    risk_profile: FofRiskProfile = FofRiskProfile.BALANCED
    max_recommendations: int = Field(default=5, ge=1, le=10)
    max_single_fund_weight: float = Field(default=0.35, gt=0, le=0.6)
    annual_risk_free_rate: float = Field(default=0.015, ge=-1, le=1)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_fund_ids(self) -> "FofRecommendationRequest":
        fund_ids = [fund.fund_id for fund in self.funds]
        if len(fund_ids) != len(set(fund_ids)):
            raise ValueError("fund_id values must be unique")
        return self


class FofCandidateEvaluation(BaseModel):
    fund_id: str
    fund_name: str
    eligible: bool
    score: float = Field(ge=0, le=100)
    score_breakdown: dict[str, float]
    metrics: PerformanceMetrics
    warnings: list[str] = Field(default_factory=list)


class FofAllocationItem(BaseModel):
    fund_id: str
    fund_name: str
    weight: float = Field(gt=0, le=1)
    rationale: str


class FofToolCall(BaseModel):
    name: str
    status: str
    duration_ms: float = Field(ge=0)
    summary: str


class FofReflection(BaseModel):
    passed: bool
    checks: list[str]
    warnings: list[str] = Field(default_factory=list)


class FofAgentEvent(BaseModel):
    """One state-machine event emitted by the independent FOF agent runtime."""

    phase: str
    action: str
    status: str
    iteration: int = Field(ge=0)
    detail: str


class FofRecommendationResponse(BaseModel):
    session_id: str
    evaluations: list[FofCandidateEvaluation]
    recommendations: list[FofAllocationItem]
    tool_trace: list[FofToolCall]
    agent_trace: list[FofAgentEvent] = Field(default_factory=list)
    plan_revisions: int = Field(default=0, ge=0)
    reflection: FofReflection
    memory_context: list[str] = Field(default_factory=list)
    method_provenance: dict = Field(default_factory=lambda: {
        "agent": {
            "method": "规则状态机 + 本地统计工具",
            "detail": "风险偏好、筛选、评分与权重均由可审计规则执行；当前不是 LLM 自主决策",
        },
    })
    disclaimer: str = (
        "本结果基于调用方提供的历史净值和预设规则生成，未接入实时数据，"
        "不构成投资建议、募集推介或适当性结论。请结合尽调、流动性、费用和合规信息复核。"
    )
