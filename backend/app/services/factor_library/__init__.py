"""CTA Factor Library — public interface.

Provides:
- Factor registry (all available factors with metadata)
- build_factors(): orchestrate data fetch + factor computation + caching
- get_factor_series(): load cached factor returns
- Factor classes for direct use
"""

import inspect
import logging
import textwrap
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from app.services.factor_library.base import FactorBase, FactorMeta
from app.services.factor_library.base import turnover_and_costs
from app.services.factor_library.trend import TrendFactor
from app.services.factor_library.volume_price import VolumePriceCorrFactor
from app.services.factor_library.cross_section import CrossSectionMomentumFactor
from app.services.factor_library.basis import BasisCarryFactor
from app.services.factor_library.profit import ProfitMarginFactor
from app.services.factor_library.short_trend import ShortTermTrendFactor
from app.services.factor_library.skewness import SkewnessFactor
from app.services.factor_library.mean_reversion import MeanReversionFactor
from app.services.factor_library.fundamental import WarehouseReceiptFactor, InventoryFactor
from app.services.factor_library.registry import get_core_symbols, get_sector_coverage
from app.services.factor_library.data_fetcher import fetch_panels, get_data_summary
from app.services.factor_library import cache
from app.services.factor_library.risk_overlay import (
    BASELINE_PROFILE,
    SUPPORTED_PROFILES,
    apply_risk_overlay,
    overlay_metadata,
)
from app.services.market_data.provider import MarketDataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

# Instantiate all factors (P1 price-volume + P3 fundamental)
_FACTORS: dict[str, FactorBase] = {}


def _register(factor: FactorBase) -> None:
    _FACTORS[factor.meta.name] = factor


_register(TrendFactor())
_register(VolumePriceCorrFactor())
_register(CrossSectionMomentumFactor())
_register(BasisCarryFactor())
_register(ProfitMarginFactor())
_register(ShortTermTrendFactor())
_register(SkewnessFactor())
_register(MeanReversionFactor())
_register(WarehouseReceiptFactor())
_register(InventoryFactor())


def get_all_factors() -> list[dict]:
    """Return metadata for all registered factors."""
    return [
        {
            "name": f.meta.name,
            "display_name": f.meta.display_name,
            "category": f.meta.category,
            "description": f.meta.description,
            "params": f.meta.params,
            "cached": cache.get_cache_info(f.meta.name) is not None,
        }
        for f in _FACTORS.values()
    ]


def get_factor(name: str) -> FactorBase | None:
    """Get a factor instance by name."""
    return _FACTORS.get(name)


def get_factor_detail(name: str) -> dict | None:
    """Return rich detail for a single factor, for the "公式与代码" dialog.

    Includes the curated signal rule / LaTeX formula / derivation from the
    factor meta plus the *actual* ``compute()`` source code, extracted at
    runtime via ``inspect`` so it can never drift from the implementation.
    Returns None for an unknown factor name.
    """
    factor = _FACTORS.get(name)
    if factor is None:
        return None
    try:
        code = textwrap.dedent(inspect.getsource(factor.compute))
    except (OSError, TypeError):
        code = ""
    return {
        "name": factor.meta.name,
        "display_name": factor.meta.display_name,
        "category": factor.meta.category,
        "description": factor.meta.description,
        "signal_rule": factor.meta.signal_rule,
        "formula": factor.meta.formula,
        "derivation": factor.meta.derivation,
        "params": factor.meta.params,
        "code": code,
    }


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------

def build_factors(
    providers: list[MarketDataProvider],
    start: date,
    end: date,
    factor_names: list[str] | None = None,
    use_cache: bool = True,
    on_progress: Callable[[str, float], None] | None = None,
) -> dict:
    """Build factor return series for the given date range.

    Parameters
    ----------
    providers : list[MarketDataProvider]
        Data providers in priority order.
    start, end : date
        Date range for factor computation.
    factor_names : list[str] | None
        Which factors to build (None = all).
    use_cache : bool
        If True, skip factors that are already cached for this range.
    on_progress : Callable[[str, float], None] | None
        Optional callback receiving (message, percent 0-100) for live logs.

    Returns
    -------
    dict with keys:
        - "results": {factor_name: {"rows": int, "start": str, "end": str, "status": str}}
        - "data_summary": summary of fetched market data
        - "errors": {factor_name: error_message} for any failures
    """
    def _emit(msg: str, pct: float) -> None:
        logger.info("[build %.0f%%] %s", pct, msg)
        if on_progress is not None:
            try:
                on_progress(msg, pct)
            except Exception:  # noqa: BLE001 - logging must never break the build
                pass

    targets = factor_names or list(_FACTORS.keys())
    results = {}
    errors = {}

    _emit(f"开始构建 {len(targets)} 个因子 ({start.isoformat()} ~ {end.isoformat()})", 2)

    # Check which factors need building
    to_build = []
    for name in targets:
        if name not in _FACTORS:
            errors[name] = f"Unknown factor: {name}"
            _emit(f"未知因子 '{name}'，已跳过", 2)
            continue
        # Factors waiting on user-supplied input data are skipped gracefully
        note = _FACTORS[name].input_note()
        if note is not None:
            results[name] = {
                "rows": 0,
                "start": None,
                "end": None,
                "status": "skipped",
                "note": note,
            }
            _emit(f"因子 '{name}' 跳过：{note}", 3)
            continue
        if use_cache:
            cached = cache.load_factor_returns(name, start, end)
            if cached is not None and len(cached) > 0:
                results[name] = {
                    "rows": len(cached),
                    "start": cached.index[0].strftime("%Y-%m-%d"),
                    "end": cached.index[-1].strftime("%Y-%m-%d"),
                    "status": "cached",
                }
                _emit(f"因子 '{name}' 已有缓存 ({len(cached)} 行)，跳过", 5)
                continue
        to_build.append(name)

    # Fetch market data if needed
    panels = {}
    data_summary = {}
    if to_build:
        _emit(f"正在获取 {len(get_core_symbols())} 个核心品种行情数据...", 8)
        panels = fetch_panels(providers, start, end)
        data_summary = get_data_summary(panels)

        if not panels:
            _emit("未获取到任何行情数据，构建终止（区间可能过短，需覆盖约 300 个交易日以上）", 100)
            for name in to_build:
                errors[name] = (
                    "No market data: 每个品种需至少 300 个交易日数据，"
                    "请扩大构建区间（建议起始日不晚于 1.5 年前）"
                )
            return {"results": results, "data_summary": data_summary, "errors": errors}

        _emit(f"行情数据就绪: {len(panels)} 个品种可用", 15)

        # Compute each factor
        n = len(to_build)
        for idx, name in enumerate(to_build):
            factor = _FACTORS[name]
            pct_base = 15 + idx * (80.0 / n)
            try:
                _emit(f"正在计算因子 '{name}' ({idx + 1}/{n})...", pct_base)
                logger.info("Computing factor '%s'...", name)
                factor_ret = factor.compute(panels)

                if factor_ret.empty or len(factor_ret) < 20:
                    errors[name] = f"Insufficient output ({len(factor_ret)} rows)"
                    _emit(f"因子 '{name}' 输出不足 ({len(factor_ret)} 行)，已跳过", pct_base + 80.0 / n)
                    continue

                # Cache the result
                cache.save_factor_returns(
                    name, factor_ret, start, end,
                    params=factor.meta.params,
                    data_version=data_summary,
                )
                # Preserve the exact series used by this build.  The active
                # cache above is intentionally retained for existing callers;
                # these immutable artifacts are the reproducibility record.
                cache.save_factor_artifact(name, "baseline_return", factor_ret, start, end, metadata={"params": factor.meta.params})
                for profile in ("vol_target", "drawdown_control"):
                    overlaid = apply_risk_overlay(factor_ret, profile)
                    cache.save_factor_artifact(name, "risk_overlay_return", overlaid, start, end, metadata={"profile": profile, "overlay": overlay_metadata(profile)})
                raw_signal = factor.compute_raw_signal(panels)
                if raw_signal is not None:
                    raw_series = raw_signal.mean(axis=1) if isinstance(raw_signal, pd.DataFrame) else raw_signal
                    cache.save_factor_artifact(name, "raw_signal", raw_series, start, end, metadata={"params": factor.meta.params})
                weights = factor.compute_weights(panels)
                if weights is not None and not weights.empty:
                    turnover, costs = turnover_and_costs(weights, factor_ret, cost_bps=5.0)
                    net_after_cost = factor_ret.sub(costs, fill_value=0.0)
                    cache.save_factor_artifact(name, "turnover", turnover, start, end, metadata={"cost_bps": 5.0, "total_turnover": float(turnover.sum())})
                    cache.save_factor_artifact(name, "net_return_after_cost", net_after_cost, start, end, metadata={"cost_bps": 5.0, "total_cost": float(costs.sum())})

                # Cache per-variety contribution matrix (best-effort)
                try:
                    contrib = factor.compute_contributions(panels)
                    if contrib is not None and not contrib.empty:
                        cache.save_factor_contributions(name, contrib, start, end)
                except (NotImplementedError, Exception) as contrib_exc:
                    logger.debug("Contributions unavailable for '%s': %s", name, contrib_exc)

                results[name] = {
                    "rows": len(factor_ret),
                    "start": factor_ret.index[0].strftime("%Y-%m-%d"),
                    "end": factor_ret.index[-1].strftime("%Y-%m-%d"),
                    "status": "built",
                }
                logger.info("Factor '%s' built: %d rows", name, len(factor_ret))
                _emit(f"因子 '{name}' 构建完成 ({len(factor_ret)} 行)", pct_base + 80.0 / n)

            except Exception as exc:
                logger.exception("Factor '%s' computation failed", name)
                errors[name] = str(exc)
                _emit(f"因子 '{name}' 计算失败: {exc}", pct_base + 80.0 / n)
    else:
        _emit("所有因子均已有缓存，无需重新计算", 95)

    built = sum(1 for r in results.values() if r["status"] == "built")
    _emit(f"构建完成: 新建 {built} 个, 缓存命中 {len(results) - built} 个, 失败 {len(errors)} 个", 100)
    return {"results": results, "data_summary": data_summary, "errors": errors}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def get_factor_series(
    factor_name: str,
    start: date | None = None,
    end: date | None = None,
    risk_profile: str = BASELINE_PROFILE,
) -> pd.Series | None:
    """Load a cached factor return series.

    Returns None if the factor has not been built yet.
    """
    rets = cache.load_factor_returns(factor_name, start, end)
    if rets is None:
        return None
    return apply_risk_overlay(rets, risk_profile)


def get_factor_nav(
    factor_name: str,
    start: date | None = None,
    end: date | None = None,
    risk_profile: str = BASELINE_PROFILE,
) -> pd.Series | None:
    """Load cached factor returns and convert to cumulative NAV (base=1)."""
    rets = get_factor_series(factor_name, start, end, risk_profile)
    if rets is None or rets.empty:
        return None
    nav = (1 + rets).cumprod()
    nav.name = f"{factor_name}_nav"
    return nav


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def compute_factor_performance(risk_profile: str = BASELINE_PROFILE) -> list[dict]:
    """Compute performance statistics for all cached factors.

    Returns a list of dicts (one per cached factor) with annualized return,
    volatility, Sharpe (rf=0), max drawdown, Calmar ratio and win rate.
    """
    if risk_profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported risk profile: {risk_profile}")
    perf: list[dict] = []
    for name, factor in _FACTORS.items():
        baseline_rets = cache.load_factor_returns(name)
        rets = apply_risk_overlay(baseline_rets, risk_profile) if baseline_rets is not None else None
        if rets is None or rets.empty or len(rets) < 20:
            continue

        rets = rets.astype(float)
        nav = (1 + rets).cumprod()

        ann_ret = float(rets.mean() * 252)
        ann_vol = float(rets.std(ddof=1) * (252 ** 0.5))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

        drawdown = nav / nav.cummax() - 1.0
        max_dd = float(drawdown.min())
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
        win_rate = float((rets > 0).mean())
        positive = rets[rets > 0]
        negative = rets[rets < 0]
        avg_win = float(positive.mean()) if not positive.empty else 0.0
        avg_loss = float(negative.mean()) if not negative.empty else 0.0
        payoff_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-12 else 0.0
        monthly = (1 + rets).resample("ME").prod() - 1

        info = cache.get_cache_info(name) or {}
        perf.append({
            "name": name,
            "display_name": factor.meta.display_name,
            "category": factor.meta.category,
            "annualized_return": round(ann_ret, 4),
            "annualized_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
            "calmar": round(calmar, 2),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "payoff_ratio": round(payoff_ratio, 3),
            "worst_month": round(float(monthly.min()), 4) if not monthly.empty else 0.0,
            "risk_profile": risk_profile,
            "risk_overlay": overlay_metadata(risk_profile),
            "rows": len(rets),
            "start": rets.index[0].strftime("%Y-%m-%d"),
            "end": rets.index[-1].strftime("%Y-%m-%d"),
            "built_at": info.get("built_at", ""),
        })
    return perf
