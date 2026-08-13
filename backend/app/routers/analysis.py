"""Analysis pipeline: strategy classification, factor analysis, variety identification, report generation."""

from time import perf_counter
from hashlib import sha256

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import persist_local_environment
from app.dependencies import akshare_provider, csv_provider, get_report_config, set_report_config
from app.services.external_factor_library import latest_snapshot_metadata
from app.schemas import (
    StrategyClassifyRequest,
    StrategyClassifyResponse,
    FactorAnalysisRequest,
    FactorAnalysisResponse,
    FactorExposureItem,
    FactorCollinearityDiagnosticsResponse,
    FactorRiskProfileResponse,
    VarietyIdentifyRequest,
    VarietyIdentifyResponse,
    VarietyCandidateItem,
    SectorExposureItem,
    ReportGenerateRequest,
    ReportGenerateResponse,
    DeepAttributionRequest,
    DeepAttributionResponse,
    LlmConfigResponse,
    LlmConfigUpdateRequest,
)
from app.services.analysis import (
    classify_strategy,
    analyze_factors,
    identify_varieties,
    identify_equity_references,
    generate_report,
    ReportConfig,
)
from app.services.analysis.pdf_export import render_markdown_pdf
from app.services.analysis._pipeline_utils import assess_nav_quality, select_provider, compute_periodic_returns
from app.services.analysis.deep_attribution import deep_attribution, as_dict as deep_attribution_as_dict

router = APIRouter(tags=["AI归因分析"])


class ReportPdfRequest(BaseModel):
    product_name: str = "产品"
    markdown: str


@router.post("/api/analysis/deep-attribution", response_model=DeepAttributionResponse)
def run_deep_attribution(request: DeepAttributionRequest) -> DeepAttributionResponse:
    """Run nonlinear/state-aware CTA attribution independently of ranking."""
    periodic_returns, return_dates = compute_periodic_returns(request.nav_points)
    provider = select_provider(request.data_source, akshare_provider, csv_provider)
    try:
        result = deep_attribution(
            product_returns=periodic_returns,
            product_dates=return_dates,
            frequency=request.frequency.value,
            strategy_type=request.strategy_type,
            provider=provider,
            top_n=request.top_n,
            bootstrap_samples=request.bootstrap_samples,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"深度归因出错: {error}") from error
    return DeepAttributionResponse(**deep_attribution_as_dict(result))


@router.post("/api/analysis/report-pdf")
def export_analysis_report_pdf(request: ReportPdfRequest) -> Response:
    """Return a real PDF download instead of opening a browser print popup."""
    if not request.markdown.strip():
        raise HTTPException(status_code=400, detail="报告内容为空，无法导出 PDF")
    pdf_bytes = render_markdown_pdf(f"{request.product_name} · 净值分析报告", request.markdown)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=analysis-report.pdf"},
    )


@router.post("/api/analysis/config/test")
def test_analysis_config() -> dict[str, object]:
    """Make a minimal real request without exposing the configured secret."""
    config = get_report_config()
    if not config.enabled:
        raise HTTPException(400, "请先保存 API Base、API Key 与模型名称。")

    started = perf_counter()
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"{config.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 4,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:300] or error.response.reason_phrase
        raise HTTPException(error.response.status_code, f"服务返回错误：{detail}") from error
    except httpx.HTTPError as error:
        raise HTTPException(502, f"无法连接模型服务：{error}") from error
    return {
        "ok": True,
        "model": config.model,
        "latency_ms": round((perf_counter() - started) * 1000),
        "detail": f"已收到模型响应：{str(content).strip()[:80] or '（空响应）'}",
    }


@router.post("/api/analysis/classify", response_model=StrategyClassifyResponse)
def run_strategy_classification(request: StrategyClassifyRequest) -> StrategyClassifyResponse:
    """Step 1: Classify strategy type (commodity CTA vs equity quant)."""
    periodic_returns, return_dates = compute_periodic_returns(request.nav_points)
    provider = select_provider(request.data_source, akshare_provider, csv_provider)

    try:
        result = classify_strategy(
            product_returns=periodic_returns,
            product_dates=return_dates,
            frequency=request.frequency.value,
            provider=provider,
            strategy_hint=request.strategy_hint,
            strategy_confirmation=request.strategy_confirmation.value,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"分类过程出错: {error}") from error

    return StrategyClassifyResponse(
        strategy_type=result.strategy_type,
        confidence_pct=result.confidence_pct,
        confidence_label=result.confidence_label,
        evidence=result.evidence,
        correlations=result.correlations,
        details=result.details,
    )


@router.post("/api/analysis/factors", response_model=FactorAnalysisResponse)
def run_factor_analysis(request: FactorAnalysisRequest) -> FactorAnalysisResponse:
    """Step 2: Analyze factor exposures (Beta, Momentum, Mean Reversion, etc.)."""
    periodic_returns, return_dates = compute_periodic_returns(request.nav_points)
    provider = select_provider(request.data_source, akshare_provider, csv_provider)

    try:
        result = analyze_factors(
            product_returns=periodic_returns,
            product_dates=return_dates,
            frequency=request.frequency.value,
            strategy_type=request.strategy_type,
            provider=provider,
            rolling_window=request.rolling_window,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"因子分析出错: {error}") from error

    factors = [
        FactorExposureItem(
            factor_name=f.factor_name,
            factor_label=f.factor_label,
            exposure_beta=f.exposure_beta,
            correlation=f.correlation,
            t_statistic=f.t_statistic,
            p_value=f.p_value,
            confidence_pct=f.confidence_pct,
            confidence_label=f.confidence_label,
            out_of_sample_hit_rate=f.out_of_sample_hit_rate,
            hit_rate_observations=f.hit_rate_observations,
            hit_rate_p_value=f.hit_rate_p_value,
            evidence=f.evidence,
            rolling_exposure=f.rolling_exposure,
        )
        for f in result.factors
    ]

    return FactorAnalysisResponse(
        factors=factors,
        r_squared=result.r_squared,
        adj_r_squared=result.adj_r_squared,
        rolling_r_squared=result.rolling_r_squared,
        evidence=result.evidence,
        warnings=result.warnings,
        collinearity=FactorCollinearityDiagnosticsResponse(
            factor_names=result.collinearity.factor_names,
            correlation_matrix=result.collinearity.correlation_matrix,
            covariance_matrix=result.collinearity.covariance_matrix,
            condition_number=result.collinearity.condition_number,
            max_abs_correlation=result.collinearity.max_abs_correlation,
            high_correlation_pairs=result.collinearity.high_correlation_pairs,
            ridge_applied=result.collinearity.ridge_applied,
            ridge_alpha=result.collinearity.ridge_alpha,
        ),
        risk_profile=FactorRiskProfileResponse(
            downside_betas=result.risk_profile.downside_betas,
            tail_correlations=result.risk_profile.tail_correlations,
            regime_returns=result.risk_profile.regime_returns,
            variance_contributions_pct=result.risk_profile.variance_contributions_pct,
            principal_components=result.risk_profile.principal_components,
            warnings=result.risk_profile.warnings,
        ),
        charts_data={
            "factor_explainability": {
                "series": result.explainability_series,
                "factor_labels": {f.factor_name: f.factor_label for f in result.factors},
                "description": "产品累计净值与回归拟合曲线；累计贡献为 beta × 因子收益的滚动累加",
            }
            if result.explainability_series else {},
        },
        method_provenance={
            "factor_analysis": {
                "method": "OLS 多因子回归 + 共线性诊断 + 滚动窗口（高共线时岭回归稳定化）+ 样本外方向一致率",
                "data_frequency": request.frequency.value,
                "strategy_scope": request.strategy_type,
                "detail": "统计代理因子，不代表实际持仓或交易胜率",
            },
        },
    )


@router.post("/api/analysis/varieties", response_model=VarietyIdentifyResponse)
def run_variety_identification(request: VarietyIdentifyRequest) -> VarietyIdentifyResponse:
    """Step 3: Identify likely futures varieties (commodity CTA only)."""
    periodic_returns, return_dates = compute_periodic_returns(request.nav_points)
    provider = select_provider(request.data_source, akshare_provider, csv_provider)

    try:
        result = identify_varieties(
            product_returns=periodic_returns,
            product_dates=return_dates,
            frequency=request.frequency.value,
            provider=provider,
            top_n=request.top_n,
            rolling_window=request.rolling_window,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"品种推断出错: {error}") from error

    top_varieties = [
        VarietyCandidateItem(
            symbol=v.symbol,
            name=v.name,
            sector=v.sector,
            probability_pct=v.probability_pct,
            correlation=v.correlation,
            lasso_coefficient=v.lasso_coefficient,
            rolling_hit_rate=v.rolling_hit_rate,
            evidence=v.evidence,
        )
        for v in result.top_varieties
    ]

    sector_exposure = [
        SectorExposureItem(
            sector=s.sector,
            color=s.color,
            total_probability=s.total_probability,
            top_variety=s.top_variety,
            variety_count=s.variety_count,
        )
        for s in result.sector_exposure
    ]

    return VarietyIdentifyResponse(
        top_varieties=top_varieties,
        sector_exposure=sector_exposure,
        method_stability=result.method_stability,
        evidence=result.evidence,
        warnings=result.warnings,
    )


@router.post("/api/analysis/report", response_model=ReportGenerateResponse)
def run_full_analysis_report(request: ReportGenerateRequest) -> ReportGenerateResponse:
    """One-click full pipeline: classify -> factors -> varieties -> report."""
    periodic_returns, return_dates = compute_periodic_returns(request.nav_points)
    provider = select_provider(request.data_source, akshare_provider, csv_provider)
    nav_values = [float(point.net_asset_value) for point in request.nav_points]
    nav_base = nav_values[0] if nav_values else 1.0
    running_peak = nav_base
    maximum_drawdown = 0.0
    for value in nav_values:
        running_peak = max(running_peak, value)
        maximum_drawdown = min(maximum_drawdown, value / running_peak - 1)
    elapsed_days = max((request.nav_points[-1].observation_date - request.nav_points[0].observation_date).days, 1)
    cumulative_return = nav_values[-1] / nav_base - 1 if nav_base else 0.0
    annualized_return = (1 + cumulative_return) ** (365.25 / elapsed_days) - 1 if cumulative_return > -1 else None
    periods_per_year = {"daily": 252, "weekly": 52, "monthly": 12}[request.frequency.value]
    annualized_volatility = float(periodic_returns.std(ddof=1) * periods_per_year ** 0.5) if len(periodic_returns) > 1 else None
    nav_profile = {
        "point_count": len(request.nav_points),
        "start_date": request.nav_points[0].observation_date.isoformat(),
        "end_date": request.nav_points[-1].observation_date.isoformat(),
        "frequency": request.frequency.value,
        "cumulative_return": round(cumulative_return, 6),
        "annualized_return": round(annualized_return, 6) if annualized_return is not None else None,
        "annualized_volatility": round(annualized_volatility, 6) if annualized_volatility is not None else None,
        "maximum_drawdown": round(maximum_drawdown, 6),
    }

    warnings = assess_nav_quality(request.nav_points)
    nav_fingerprint = sha256(
        "|".join(f"{point.observation_date.isoformat()}:{float(point.net_asset_value):.8f}" for point in request.nav_points).encode()
    ).hexdigest()[:12]
    external_factor_context = latest_snapshot_metadata(
        product_start=min(point.observation_date for point in request.nav_points),
        product_end=max(point.observation_date for point in request.nav_points),
    )
    warnings.extend(external_factor_context.get("warnings", []))
    nav_quality_blocked = any(
        "非数值或非正值" in warning or "异常跳变" in warning
        for warning in warnings
    )
    manual_quality_override = nav_quality_blocked and request.quality_override_confirmed
    if nav_quality_blocked and not manual_quality_override:
        return ReportGenerateResponse(
            strategy_summary=(
                "本次仅生成数据复核结论：净值序列存在需要人工确认的异常，"
                "已停止策略分类、因子归因和品种推断，避免把图像识别误差表述为研究结论。"
            ),
            structured={"nav_profile": nav_profile},
            charts_data={
                "product_nav_curve": [
                    {
                        "date": point.observation_date.isoformat(),
                        "nav": round(float(point.net_asset_value), 8),
                        "normalized_nav": round(float(point.net_asset_value) / nav_base, 8) if nav_base else None,
                    }
                    for point in request.nav_points
                ],
                "chart_notes": {
                    "frequency": request.frequency.value,
                    "factor_alignment": "数据质量未通过，未运行因子对齐或回归",
                },
            },
            disclaimer="当前为数据复核报告，不构成策略分类、因子归因或投资建议。",
            generated_at="",
            engine="quality-gate",
            warnings=warnings,
            method_provenance={
                "nav_quality": {
                    "method": "本地净值异常检查",
                    "detail": "检测到异常跳变或无效净值；必须人工复核后重新生成正式报告。",
                },
                "report_version": {
                    "method": "可追溯研究快照",
                    "nav_fingerprint": nav_fingerprint,
                    "point_count": len(request.nav_points),
                    "frequency": request.frequency.value,
                    "detail": "此版本被数据质量门禁拦截，仅保留复核证据。",
                },
            },
        )

    if manual_quality_override:
        warnings.append("人工复核已确认异常净值点：本次继续研究；异常跳变仍保留在报告和审计记录中。")

    # Step 1: Classification
    try:
        classification = classify_strategy(
            periodic_returns,
            return_dates,
            request.frequency.value,
            provider,
            strategy_hint=request.strategy_hint,
            strategy_confirmation=request.strategy_confirmation.value,
        )
    except Exception as exc:
        classification = None
        warnings.append(f"策略分类失败: {exc}")

    # This reference is intentionally computed for every product window,
    # independently of whether we can classify the strategy.  It gives the
    # user a same-period AKShare market comparison without pretending that a
    # low-correlation product has a confirmed CTA factor exposure.
    market_reference_context = {
        "available": bool(classification and classification.correlations),
        "source": "AKShare 同期市场行情",
        "start_date": nav_profile["start_date"],
        "end_date": nav_profile["end_date"],
        "frequency": request.frequency.value,
        "references": classification.correlations if classification else {},
        "data_role": "same_period_market_reference",
        "note": "按产品净值起止日计算，仅作市场背景对比，不代表产品持仓或正式因子归因。",
    }
    if not market_reference_context["available"]:
        warnings.append("未获取到可对齐的 AKShare 同期市场参考，已跳过市场对比。")

    # Step 2: Factor analysis
    strategy_type = classification.strategy_type if classification else "insufficient_data"
    factors = None
    disclosure_hint = (classification.details if classification else {}).get("disclosure_hint", {})
    disclosure_conflict = bool(disclosure_hint.get("conflict"))
    user_confirmation = (classification.details if classification else {}).get("user_confirmation", {})
    user_confirmed = bool(user_confirmation.get("confirmed"))
    user_disclosure_conflict = bool(user_confirmation.get("disclosure_conflict"))
    # ``mixed`` is supported by the factor builder as a deliberately broad
    # statistical proxy set.  It must not be treated as a confirmed holdings
    # model, but skipping it entirely made mixed reports lose every factor.
    route_factor_model = strategy_type in ("commodity_cta", "equity_quant", "mixed") and (
        not disclosure_conflict or user_confirmed
    )
    if route_factor_model:
        try:
            factors = analyze_factors(
                periodic_returns, return_dates, request.frequency.value,
                strategy_type, provider, request.rolling_window,
            )
        except Exception as exc:
            warnings.append(f"因子分析失败: {exc}")
    else:
        if disclosure_conflict and not user_confirmed:
            warnings.append("管理人披露与净值统计分类存在冲突，已跳过单一资产因子模型；请先人工确认策略范围。")
        elif user_confirmed and strategy_type == "mixed":
            warnings.append("混合/多资产统计因子模型未能运行；请检查外部行情是否可按实际日期对齐。")
        else:
            warnings.append("策略类型尚未确认，已跳过因子与品种推断；请补充策略披露或上传可对齐的外部因子数据。")
    if disclosure_conflict and user_confirmed:
        warnings.append("已按用户确认的策略范围运行模型；披露与净值统计冲突仍保留在报告中。")
    if user_disclosure_conflict:
        warnings.append("用户确认范围与资料披露候选不完全一致；请在报告中核对策略边界。")

    # Step 3: Variety identification (commodity only)
    varieties = None
    if strategy_type == "commodity_cta" and (not disclosure_conflict or user_confirmed):
        try:
            varieties = identify_varieties(
                periodic_returns, return_dates, request.frequency.value,
                provider, request.top_n_varieties, request.rolling_window,
            )
        except Exception as exc:
            warnings.append(f"品种推断失败: {exc}")
    elif strategy_type == "equity_quant" and (not disclosure_conflict or user_confirmed):
        try:
            varieties = identify_equity_references(
                periodic_returns, return_dates, request.frequency.value,
                provider, request.top_n_varieties, request.rolling_window,
            )
        except Exception as exc:
            warnings.append(f"股指 CTA 市场线索推断失败: {exc}")
    elif strategy_type == "mixed":
        warnings.append("策略为混合型，未自动套用商品期货品种库；请先确认商品与股指部分的研究范围。")

    # Step 4: Report generation
    report = generate_report(
        classification,
        factors,
        varieties,
        get_report_config(),
        strategy_hint=request.strategy_hint,
        external_factor_context=external_factor_context,
        market_reference_context=market_reference_context,
    )

    # Deep attribution is an independent evidence layer.  It is intentionally
    # not used to alter CODEX ranking or the existing OLS factor scores.
    deep_result = None
    if strategy_type in ("commodity_cta", "equity_quant", "mixed") and (not disclosure_conflict or user_confirmed):
        try:
            deep_result = deep_attribution(
                product_returns=periodic_returns,
                product_dates=return_dates,
                frequency=request.frequency.value,
                strategy_type=strategy_type,
                provider=provider,
                top_n=request.top_n_varieties,
            )
            report.structured["deep_attribution"] = deep_attribution_as_dict(deep_result)
            if deep_result.asset_class.get("most_likely_label"):
                supported = [
                    item["label"]
                    for item in deep_result.strategy_fingerprints
                    if item.get("status") == "supported"
                ]
                report.strategy_summary = (
                    f"{report.strategy_summary}\n"
                    f"【深度归因】资产大类候选为{deep_result.asset_class['most_likely_label']}"
                    f"（候选概率 {deep_result.asset_class.get('confidence_pct', 0):.1f}%）；"
                    f"相对稳定的行为指纹：{'、'.join(supported[:3]) if supported else '暂未发现稳定指纹'}。"
                    "以上为公开代理的统计候选，不是实际持仓确认。"
                )
        except Exception as exc:
            warnings.append(f"深度归因失败: {exc}")

    if factors and factors.warnings:
        warnings.extend(factors.warnings)
    if varieties and varieties.warnings:
        warnings.extend(varieties.warnings)
    if deep_result and deep_result.warnings:
        warnings.extend(deep_result.warnings)

    # Keep the original product curve alongside the aligned regression curve.
    # This lets the reviewer see whether a seemingly good factor fit is only
    # an artefact of the warm-up/intersection window.
    report.structured["nav_profile"] = nav_profile
    report.charts_data["product_nav_curve"] = [
        {
            "date": point.observation_date.isoformat(),
            "nav": round(float(point.net_asset_value), 8),
            "normalized_nav": round(float(point.net_asset_value) / nav_base, 8) if nav_base else None,
        }
        for point in request.nav_points
    ]
    report.charts_data["chart_notes"] = {
        "frequency": request.frequency.value,
        "factor_alignment": "按实际日期/周期间交集对齐，未使用末 N 条位置截断",
    }

    return ReportGenerateResponse(
        strategy_summary=report.strategy_summary,
        structured=report.structured,
        charts_data=report.charts_data,
        disclaimer=report.disclaimer,
        generated_at=report.generated_at,
        engine=report.engine,
        warnings=warnings,
        method_provenance={
            **report.method_provenance,
            "nav_quality": {
                "method": "本地异常检查 + 人工复核确认" if manual_quality_override else "本地异常检查",
                "detail": "检测到异常跳变，已由用户确认后继续运行；结论仍应结合原始材料复核。" if manual_quality_override else "未检测到阻断性的无效净值或异常跳变。",
                "manual_override_confirmed": manual_quality_override,
            },
            "nav_extraction": {
                "method": "用户上传净值表 / CV 曲线追踪 / VLM 结构辅助（由上游导入方式决定）",
                "data_source": request.data_source,
                "frequency": request.frequency.value,
                "detail": "报告只使用已传入的净值点；请在低置信度或待复核状态下先人工校验",
            },
            "report_version": {
                "method": "可追溯研究快照",
                "nav_fingerprint": nav_fingerprint,
                "point_count": len(request.nav_points),
                "frequency": request.frequency.value,
                "detail": "净值被重新校准后，应以新的指纹重新生成报告；旧报告仅作历史记录。",
            },
            "deep_attribution": (
                deep_result.method_provenance
                if deep_result
                else {
                    "method": "未运行",
                    "detail": "策略范围未确认、数据质量门禁或资产类别冲突导致深度归因未运行。",
                }
            ),
        },
    )


@router.get("/api/analysis/config", response_model=LlmConfigResponse)
def get_analysis_config() -> LlmConfigResponse:
    """Get current LLM configuration status (API key is never exposed)."""
    config = get_report_config()
    provider = ""
    if config.api_base:
        if "deepseek" in config.api_base:
            provider = "deepseek"
        elif "volces" in config.api_base or "doubao" in config.api_base:
            provider = "doubao"
        elif "dashscope" in config.api_base:
            provider = "qwen"
        elif "openai" in config.api_base:
            provider = "openai"
        else:
            provider = "custom"

    return LlmConfigResponse(
        enabled=config.enabled,
        provider=provider,
        model=config.model,
        api_base_configured=bool(config.api_base),
    )


@router.put("/api/analysis/config", response_model=LlmConfigResponse)
def update_analysis_config(request: LlmConfigUpdateRequest) -> LlmConfigResponse:
    """Update LLM configuration at runtime."""
    new_config = ReportConfig(
        api_base=request.api_base,
        api_key=request.api_key,
        model=request.model,
        enabled=bool(request.api_base and request.api_key),
    )
    set_report_config(new_config)
    persist_local_environment({
        "LLM_API_BASE": request.api_base,
        "LLM_API_KEY": request.api_key,
        "LLM_MODEL": request.model,
    })
    return get_analysis_config()
