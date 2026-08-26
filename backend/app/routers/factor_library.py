"""Factor library endpoints: listing, building, performance, attribution, validation, drift."""

import threading
import uuid
from datetime import date as dt_date, datetime

from fastapi import APIRouter, HTTPException
from app.dependencies import akshare_provider, csv_provider
from app.schemas import (
    FactorAttributionRequest,
    SectorBreakdownRequest,
    FactorValidateRequest,
    FactorAlphaRequest,
    FactorDriftRequest,
)
from app.services import factor_library

router = APIRouter(tags=["因子库"])


# ---------------------------------------------------------------------------
# Build job store (encapsulates the in-memory job dict + lock)
# ---------------------------------------------------------------------------


class BuildJobStore:
    """Thread-safe in-memory store for async factor-build jobs.

    Keeps at most *max_history* completed/errored jobs.
    """

    def __init__(self, max_history: int = 10) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def has_running(self) -> bool:
        with self._lock:
            return any(j["status"] == "running" for j in self._jobs.values())

    def create(self, params: dict) -> dict:
        job_id = uuid.uuid4().hex[:8]
        job: dict = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "logs": [],
            "result": None,
            "error": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "params": params,
        }
        with self._lock:
            # Evict oldest when over capacity
            if len(self._jobs) >= self._max_history:
                oldest_keys = sorted(self._jobs, key=lambda k: self._jobs[k]["started_at"])
                for k in oldest_keys[: len(self._jobs) - self._max_history + 1]:
                    del self._jobs[k]
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)


_build_store = BuildJobStore()


# ---------------------------------------------------------------------------
# Factor listing & detail
# ---------------------------------------------------------------------------


@router.get("/api/factor-library/factors")
def list_factors() -> dict:
    """List all available factors with metadata and cache status."""
    return {
        "factors": factor_library.get_all_factors(),
        "core_varieties": factor_library.registry.get_core_symbols(),
        "sector_coverage": factor_library.registry.get_sector_coverage(),
    }


@router.get("/api/factor-library/factors/{factor_name}/detail")
def factor_detail(factor_name: str) -> dict:
    """Rich detail for one factor: signal rule, LaTeX formula, derivation, params and compute() source."""
    detail = factor_library.get_factor_detail(factor_name)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"未知因子 '{factor_name}'")
    return detail


# ---------------------------------------------------------------------------
# Factor building (sync + async)
# ---------------------------------------------------------------------------


@router.post("/api/factor-library/build")
def build_factor_library(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    factors: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Trigger factor computation for the given date range (synchronous)."""
    try:
        start_date = dt_date.fromisoformat(start)
        end_date = dt_date.fromisoformat(end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"日期格式错误: {exc}") from exc

    factor_names = [f.strip() for f in factors.split(",")] if factors else None
    providers = [csv_provider, akshare_provider]

    try:
        result = factor_library.build_factors(
            providers=providers,
            start=start_date,
            end=end_date,
            factor_names=factor_names,
            use_cache=use_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"因子构建失败: {exc}") from exc

    return result


@router.post("/api/factor-library/build-async")
def build_factor_library_async(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    factors: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Start factor library build in a background thread.

    Returns a job_id immediately; poll GET /api/factor-library/build-job/{job_id}.
    """
    if _build_store.has_running():
        raise HTTPException(status_code=409, detail="已有构建任务正在运行，请等待完成后再试")

    try:
        start_date = dt_date.fromisoformat(start)
        end_date = dt_date.fromisoformat(end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"日期格式错误: {exc}") from exc

    factor_names = [f.strip() for f in factors.split(",")] if factors else None

    job = _build_store.create({
        "start": start, "end": end, "factors": factor_names, "use_cache": use_cache,
    })

    def _run() -> None:
        def on_progress(msg: str, pct: float) -> None:
            job["logs"].append({"time": datetime.now().strftime("%H:%M:%S"), "message": msg})
            job["progress"] = round(pct, 1)

        try:
            result = factor_library.build_factors(
                providers=[csv_provider, akshare_provider],
                start=start_date,
                end=end_date,
                factor_names=factor_names,
                use_cache=use_cache,
                on_progress=on_progress,
            )
            job["result"] = result
            job["status"] = "done"
            job["progress"] = 100
        except Exception as exc:  # noqa: BLE001
            job["error"] = str(exc)
            job["status"] = "error"
            job["logs"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": f"构建异常终止: {exc}",
            })

    threading.Thread(target=_run, daemon=True, name=f"factor-build-{job['job_id']}").start()
    return {"job_id": job["job_id"]}


@router.get("/api/factor-library/build-job/{job_id}")
def get_build_job(job_id: str) -> dict:
    """Poll the status of an async build job (progress, logs, result)."""
    job = _build_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="构建任务不存在")
    return job


# ---------------------------------------------------------------------------
# Performance & series
# ---------------------------------------------------------------------------


@router.get("/api/factor-library/performance")
def factor_performance(risk_profile: str = "baseline") -> dict:
    """Performance statistics for cached factors."""
    try:
        performance = factor_library.compute_factor_performance(risk_profile)
        from app.services.factor_library.risk_overlay import overlay_metadata
        return {"performance": performance, "risk_overlay": overlay_metadata(risk_profile)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/factor-library/series/{factor_name}")
def get_factor_series(
    factor_name: str,
    start: str | None = None,
    end: str | None = None,
    format: str = "returns",
    risk_profile: str = "baseline",
) -> dict:
    """Get a cached factor return series (or NAV)."""
    start_date = dt_date.fromisoformat(start) if start else None
    end_date = dt_date.fromisoformat(end) if end else None

    try:
        if format == "nav":
            series = factor_library.get_factor_nav(factor_name, start_date, end_date, risk_profile)
        else:
            series = factor_library.get_factor_series(factor_name, start_date, end_date, risk_profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if series is None or series.empty:
        raise HTTPException(
            status_code=404,
            detail=f"因子 '{factor_name}' 尚未构建或无数据。请先调用 POST /api/factor-library/build",
        )

    return {
        "factor_name": factor_name,
        "format": format,
        "risk_profile": risk_profile,
        "rows": len(series),
        "start": series.index[0].strftime("%Y-%m-%d"),
        "end": series.index[-1].strftime("%Y-%m-%d"),
        "data": [
            {"date": dt.strftime("%Y-%m-%d"), "value": round(float(val), 8)}
            for dt, val in series.items()
        ],
    }


@router.get("/api/factor-library/status")
def factor_library_status() -> dict:
    """Return cache status and data provider availability."""
    cached_factors = factor_library.cache.list_cached_factors()
    return {
        "cached_factors": cached_factors,
        "providers": {
            "akshare_available": akshare_provider.is_available(),
            "csv_symbols": csv_provider.loaded_symbols(),
        },
        "core_varieties": factor_library.registry.get_core_symbols(),
    }


# ---------------------------------------------------------------------------
# Export (CSV / Markdown)
# ---------------------------------------------------------------------------


@router.get("/api/factor-library/export")
def export_factor_library(format: str = "csv") -> dict:
    """Export the factor catalog and performance as CSV or Markdown text.

    Query params:
        format: "csv" (default) or "markdown".

    Returns JSON with ``filename``, ``content_type`` and ``content`` so the
    frontend can trigger a client-side download.
    """
    from app.services.factor_library.export import export_csv, export_markdown

    if format == "markdown":
        return {
            "filename": "factor_library.md",
            "content_type": "text/markdown; charset=utf-8",
            "content": export_markdown(),
        }
    # Default: CSV
    return {
        "filename": "factor_library.csv",
        "content_type": "text/csv; charset=utf-8",
        "content": export_csv(),
    }


# ---------------------------------------------------------------------------
# Sector breakdown
# ---------------------------------------------------------------------------


@router.post("/api/factor-library/sector-breakdown")
def sector_breakdown(request: SectorBreakdownRequest) -> dict:
    """Sector-level profit/loss attribution for factors over a date range."""
    from app.services.factor_library.sector_breakdown import compute_all_sector_breakdowns

    try:
        start = dt_date.fromisoformat(request.start)
        end = dt_date.fromisoformat(request.end)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid date format; use YYYY-MM-DD")

    results = compute_all_sector_breakdowns(start, end, request.factor_names)

    return {"start": request.start, "end": request.end, "factors": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Factor attribution (L1 regression)
# ---------------------------------------------------------------------------


@router.post("/api/factor-library/attribute")
def run_factor_attribution(request: FactorAttributionRequest) -> dict:
    """Quantitative L1 factor attribution with HAC inference and block bootstrap."""
    import numpy as np
    from app.services.factor_library.factor_regression import run_factor_regression

    dates = [p.date for p in request.nav_points]
    navs = np.array([p.nav for p in request.nav_points], dtype=float)

    periodic_returns = np.diff(navs) / navs[:-1]
    return_dates = dates[1:]

    try:
        result = run_factor_regression(
            product_returns=periodic_returns,
            product_dates=return_dates,
            frequency=request.frequency.value,
            factor_names=request.factor_names,
            rolling_window=request.rolling_window,
            hac_max_lags=request.hac_max_lags,
            bootstrap_reps=request.bootstrap_reps,
            bootstrap_block_length=request.bootstrap_block_length,
            random_seed=request.random_seed,
            oos_train_window=request.oos_train_window,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"因子回归失败: {exc}") from exc

    return {
        "intercept": result.intercept,
        "annualized_alpha": result.annualized_alpha,
        "alpha_t_stat": result.alpha_t_stat,
        "alpha_p_value": result.alpha_p_value,
        "annualized_alpha_bootstrap_ci_low": result.annualized_alpha_bootstrap_ci_low,
        "annualized_alpha_bootstrap_ci_high": result.annualized_alpha_bootstrap_ci_high,
        "r_squared": result.r_squared,
        "adj_r_squared": result.adj_r_squared,
        "f_statistic": result.f_statistic,
        "f_p_value": result.f_p_value,
        "n_observations": result.n_observations,
        "factors": [
            {
                "name": f.name,
                "display_name": f.display_name,
                "beta": f.beta,
                "std_error": f.std_error,
                "t_stat": f.t_stat,
                "p_value": f.p_value,
                "contribution_pct": f.contribution_pct,
                "significant": f.significant,
                "factor_group": f.factor_group,
                "ordinary_std_error": f.ordinary_std_error,
                "bootstrap_ci_low": f.bootstrap_ci_low,
                "bootstrap_ci_high": f.bootstrap_ci_high,
            }
            for f in result.factors
        ],
        "residual": {
            "annual_vol": result.residual_annual_vol,
            "skewness": result.residual_skew,
            "kurtosis": result.residual_kurtosis,
        },
        "rolling_r_squared": [
            {"date": s.date, "r_squared": s.r_squared, "betas": s.betas}
            for s in result.rolling_r_squared
        ],
        "lasso_selected": result.lasso_selected,
        "frequency": result.frequency,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "warnings": result.warnings,
        "inference_method": result.inference_method,
        "hac_max_lags": result.hac_max_lags,
        "diagnostics": result.diagnostics,
        "bootstrap": result.bootstrap,
        "factor_groups": result.factor_groups,
        "factor_group_contributions": result.factor_group_contributions,
        "factor_risk_contributions": result.factor_risk_contributions,
        "joint_hac": result.joint_hac,
        "collinearity": result.collinearity,
        "out_of_sample": result.out_of_sample,
    }


# ---------------------------------------------------------------------------
# Validation, alpha, drift (P4)
# ---------------------------------------------------------------------------


@router.post("/api/factor-library/validate")
def validate_l1_vs_l4(request: FactorValidateRequest) -> dict:
    """Compare L1 regression results against L4 weekly report ground truth."""
    from app.services.factor_library.validation import validate_against_report

    regression_result = request.regression_result
    report_snapshot = request.report_snapshot
    period_returns = request.factor_period_returns
    period_start = request.period_start
    period_end = request.period_end

    if period_returns is None and (period_start or period_end):
        if not period_start or not period_end:
            raise HTTPException(status_code=422, detail="自动计算同期因子收益需要同时提供 period_start 和 period_end")
        try:
            start_date = dt_date.fromisoformat(period_start)
            end_date = dt_date.fromisoformat(period_end)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"报告期日期格式错误: {exc}") from exc
        period_returns = {}
        for item in regression_result.get("factors", []):
            name = item.get("name")
            if not name:
                continue
            series = factor_library.get_factor_series(name, start_date, end_date, risk_profile="baseline")
            if series is not None and not series.empty:
                period_returns[name] = float((1 + series).prod() - 1)
    result = validate_against_report(regression_result, report_snapshot, period_returns)

    return {
        "product_name": result.product_name,
        "report_date": result.report_date,
        "r_squared": result.r_squared,
        "r_gap_interpretation": result.r_gap_interpretation,
        "direction_agreement_pct": result.direction_agreement_pct,
        "comparison_period": {"start": period_start, "end": period_end} if period_returns is not None else None,
        "factor_comparisons": [
            {
                "factor_name": fc.factor_name,
                "display_name": fc.display_name,
                "l1_beta": fc.l1_beta,
                "l1_t_stat": fc.l1_t_stat,
                "l1_factor_period_return": fc.l1_factor_period_return,
                "l1_predicted_contribution": fc.l1_predicted_contribution,
                "l4_contribution": fc.l4_contribution,
                "direction_match": fc.direction_match,
                "magnitude_ratio": fc.magnitude_ratio,
            }
            for fc in result.factor_comparisons
        ],
        "summary": result.summary,
        "warnings": result.warnings,
    }


@router.post("/api/factor-library/alpha")
def measure_alpha_endpoint(request: FactorAlphaRequest) -> dict:
    """Compute detailed alpha metrics from a regression result."""
    from app.services.factor_library.validation import measure_alpha

    metrics = measure_alpha(request.regression_result)

    return {
        "annualized_alpha": metrics.annualized_alpha,
        "alpha_t_stat": metrics.alpha_t_stat,
        "alpha_significant": metrics.alpha_significant,
        "information_ratio": metrics.information_ratio,
        "residual_annual_vol": metrics.residual_annual_vol,
        "residual_skewness": metrics.residual_skew,
        "residual_kurtosis": metrics.residual_kurtosis,
        "anomaly_count": metrics.anomaly_count,
        "anomaly_dates": metrics.anomaly_dates,
        "interpretation": metrics.interpretation,
    }


@router.post("/api/factor-library/drift")
def detect_drift_endpoint(request: FactorDriftRequest) -> dict:
    """Detect style drift from rolling regression snapshots."""
    from app.services.factor_library.validation import detect_drift

    report = detect_drift(request.rolling_snapshots, threshold_pct=request.threshold_pct)

    return {
        "n_periods": report.n_periods,
        "date_range": report.date_range,
        "r_squared_trend": report.r_squared_trend,
        "style_shift_detected": report.style_shift_detected,
        "alerts": [
            {
                "factor_name": a.factor_name,
                "display_name": a.display_name,
                "beta_early": a.beta_early,
                "beta_late": a.beta_late,
                "change": a.change,
                "change_pct": a.change_pct,
                "severity": a.severity,
            }
            for a in report.alerts
        ],
        "summary": report.summary,
    }
