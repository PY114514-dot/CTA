"""Step 3: Futures variety identification (commodity CTA only).

Identifies which futures varieties a product is most likely trading by:
1. Per-variety correlation scan
2. LASSO / ElasticNet sparse regression (non-zero coefficients = candidate varieties)
3. Sector-level aggregation
4. Rolling-window stability check

Output is a probability-ranked list of candidate varieties with evidence.
This is a statistical inference, NOT a reconstruction of actual holdings.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from app.services.market_data.provider import MarketDataProvider
from app.services.market_data.index_registry import FUTURES_VARIETIES, SECTOR_COLORS
from app.services.analysis._stats import safe_pearson

logger = logging.getLogger(__name__)

_MIN_OBS = 20  # minimum observations for regression
_MIN_VARIETY_DATA = 15  # minimum data points for a variety to be included


@dataclass
class VarietyCandidate:
    """A candidate futures variety with probability and evidence."""

    symbol: str
    name: str
    sector: str
    probability_pct: float  # 0-100 composite probability
    correlation: float
    lasso_coefficient: float
    rolling_hit_rate: float  # fraction of windows where this variety enters Top-K
    evidence: list[str] = field(default_factory=list)


@dataclass
class SectorExposure:
    """Aggregated exposure at the sector level."""

    sector: str
    color: str
    total_probability: float  # sum of variety probabilities in this sector
    top_variety: str  # highest-probability variety in this sector
    variety_count: int  # number of varieties with non-trivial probability


@dataclass
class VarietyIdentificationResult:
    """Complete variety identification output."""

    top_varieties: list[VarietyCandidate]
    sector_exposure: list[SectorExposure]
    method_stability: float  # 0-1, how stable the Top-K is across windows
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def identify_varieties(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    provider: MarketDataProvider,
    top_n: int = 8,
    rolling_window: int = 12,
) -> VarietyIdentificationResult:
    """Identify likely futures varieties traded by the product.

    Args:
        product_returns: Periodic returns array.
        product_dates: Corresponding dates.
        frequency: "daily" | "weekly" | "monthly".
        provider: Market data provider.
        top_n: Number of top varieties to return.
        rolling_window: Window size for stability check.

    Returns:
        VarietyIdentificationResult with ranked varieties and sector exposure.
    """
    n = len(product_returns)
    if n < _MIN_OBS:
        return VarietyIdentificationResult(
            top_varieties=[],
            sector_exposure=[],
            method_stability=0.0,
            warnings=[f"样本量 {n} 期不足 {_MIN_OBS} 期，无法进行品种推断"],
        )

    start = min(product_dates)
    end = max(product_dates)
    # Buffer start for alignment
    buffer = {"daily": timedelta(days=5), "weekly": timedelta(weeks=2), "monthly": timedelta(days=35)}
    buffered_start = start - buffer.get(frequency, timedelta(days=5))

    # Fetch all available variety returns
    variety_data, aligned_product = _fetch_variety_returns(
        product_returns, product_dates, frequency, provider, buffered_start, end
    )

    if not variety_data:
        return VarietyIdentificationResult(
            top_varieties=[],
            sector_exposure=[],
            method_stability=0.0,
            warnings=["无法获取任何期货品种行情数据，请检查网络或上传 CSV 数据"],
        )

    # Align product returns with variety matrix
    min_len = min(len(aligned_product), min(len(v) for v in variety_data.values()))
    if min_len < _MIN_OBS:
        return VarietyIdentificationResult(
            top_varieties=[], sector_exposure=[], method_stability=0.0,
            warnings=["外部品种行情与产品净值没有足够的实际日期交集，已停止品种推断"],
        )
    aligned_product = aligned_product[-min_len:]

    # Build feature matrix
    symbols = list(variety_data.keys())
    X = np.column_stack([variety_data[s][-min_len:] for s in symbols])

    if X.shape[0] < _MIN_OBS or X.shape[1] == 0:
        return VarietyIdentificationResult(
            top_varieties=[],
            sector_exposure=[],
            method_stability=0.0,
            warnings=["对齐后数据不足，无法进行品种回归"],
        )

    # Step 1: Per-variety correlation scan
    correlations = _correlation_scan(aligned_product, X, symbols)

    # Step 2: LASSO sparse regression
    lasso_coefs = _lasso_regression(aligned_product, X, symbols)

    # Step 3: Rolling stability
    hit_rates, stability = _rolling_stability(
        aligned_product, X, symbols, rolling_window, top_n
    )

    # Step 4: Composite scoring
    candidates = _composite_scoring(
        symbols, correlations, lasso_coefs, hit_rates, top_n
    )

    # Step 5: Sector aggregation
    sectors = _aggregate_sectors(candidates)

    # Generate evidence
    evidence = _generate_evidence(candidates, sectors, stability, len(variety_data))
    warnings = []
    if n < 60:
        warnings.append(f"样本仅 {n} 期，品种级推断置信度受限，建议至少 60 期以上")
    if stability < 0.5:
        warnings.append(f"结论跨窗口稳定性较低（{stability:.0%}），品种排名可能随时间变化")

    return VarietyIdentificationResult(
        top_varieties=candidates[:top_n],
        sector_exposure=sectors,
        method_stability=round(stability, 3),
        evidence=evidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_variety_returns(
    product_returns: np.ndarray,
    product_dates: list[date],
    frequency: str,
    provider: MarketDataProvider,
    start: date,
    end: date,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fetch varieties and retain only actual product-period intersections."""
    result: dict[str, np.ndarray] = {}
    product_by_period = pd.Series(product_returns, index=pd.to_datetime(product_dates), dtype=float)
    if frequency == "weekly":
        product_by_period = product_by_period.groupby(product_by_period.index.to_period("W-SUN")).last()
    elif frequency == "monthly":
        product_by_period = product_by_period.groupby(product_by_period.index.to_period("M")).last()
    else:
        product_by_period.index = product_by_period.index.normalize()
    series_by_symbol: dict[str, pd.Series] = {}

    for symbol in FUTURES_VARIETIES:
        try:
            returns = provider.get_returns(symbol, start, end, is_index=False)
            if returns.empty or len(returns) < _MIN_VARIETY_DATA:
                continue

            market = returns.copy()
            market.index = pd.to_datetime(market.index)
            if frequency == "weekly":
                market = (1.0 + market).resample("W-SUN").prod() - 1.0
                market.index = market.index.to_period("W-SUN")
            elif frequency == "monthly":
                market = (1.0 + market).resample("ME").prod() - 1.0
                market.index = market.index.to_period("M")
            else:
                market.index = market.index.normalize()
            series_by_symbol[symbol] = market
        except Exception as exc:
            logger.debug("Skipping variety %s: %s", symbol, exc)

    if not series_by_symbol:
        return {}, np.asarray([], dtype=float)
    frame = pd.DataFrame({symbol: series.reindex(product_by_period.index) for symbol, series in series_by_symbol.items()})
    frame["__product__"] = product_by_period
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < _MIN_OBS:
        return {}, np.asarray([], dtype=float)
    for symbol in series_by_symbol:
        result[symbol] = frame[symbol].to_numpy(dtype=float)
    return result, frame["__product__"].to_numpy(dtype=float)


def _resample_to_frequency(returns: pd.Series, frequency: str, target_len: int) -> np.ndarray | None:
    """Resample daily returns to match product frequency."""
    if returns.empty:
        return None

    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    if frequency == "weekly":
        resampled = (1 + returns).resample("W").prod() - 1
    elif frequency == "monthly":
        resampled = (1 + returns).resample("ME").prod() - 1
    else:
        resampled = returns

    resampled = resampled.dropna()
    values = resampled.values

    if len(values) < target_len:
        return None
    return values[-target_len:]


# ---------------------------------------------------------------------------
# Analysis methods
# ---------------------------------------------------------------------------


def _correlation_scan(
    y: np.ndarray, X: np.ndarray, symbols: list[str]
) -> dict[str, float]:
    """Compute Pearson correlation between product and each variety."""
    correlations = {}
    for i, symbol in enumerate(symbols):
        col = X[:, i]
        correlations[symbol] = round(safe_pearson(y, col), 4)
    return correlations


def _lasso_regression(
    y: np.ndarray, X: np.ndarray, symbols: list[str]
) -> dict[str, float]:
    """Run LASSO with cross-validated alpha to select varieties.

    Returns normalized coefficients (non-zero = selected by LASSO).
    """
    # Standardize features for fair comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        # LassoCV with cross-validation
        lasso = LassoCV(cv=5, max_iter=5000, n_alphas=50, random_state=42)
        lasso.fit(X_scaled, y)
        coefs = lasso.coef_
    except Exception:
        # Fallback to ElasticNet if LassoCV fails
        try:
            enet = ElasticNetCV(cv=5, max_iter=5000, l1_ratio=0.8, random_state=42)
            enet.fit(X_scaled, y)
            coefs = enet.coef_
        except Exception:
            return {s: 0.0 for s in symbols}

    return {symbol: round(float(coefs[i]), 5) for i, symbol in enumerate(symbols)}


def _rolling_stability(
    y: np.ndarray, X: np.ndarray, symbols: list[str],
    window: int, top_k: int,
) -> tuple[dict[str, float], float]:
    """Check how stable the Top-K varieties are across rolling windows.

    Returns:
        hit_rates: fraction of windows each variety appears in Top-K
        stability: overall stability metric (average Jaccard similarity between consecutive windows)
    """
    n = len(y)
    effective_window = max(window, _MIN_OBS)

    if n < effective_window * 2:
        return {s: 0.0 for s in symbols}, 0.0

    step = max(1, effective_window // 3)
    hit_counts = {s: 0 for s in symbols}
    total_windows = 0
    prev_top_set: set[str] | None = None
    jaccard_scores: list[float] = []

    for i in range(effective_window, n + 1, step):
        y_win = y[i - effective_window:i]
        X_win = X[i - effective_window:i]

        # Quick LASSO on this window
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_win)
            lasso = LassoCV(cv=3, max_iter=2000, n_alphas=20, random_state=42)
            lasso.fit(X_scaled, y_win)
            coefs = np.abs(lasso.coef_)
        except Exception:
            # Fallback: use correlation ranking for this window
            coefs = np.array([
                abs(safe_pearson(y_win, X_win[:, j]))
                for j in range(X_win.shape[1])
            ])

        # Top-K by coefficient magnitude
        top_indices = np.argsort(coefs)[-top_k:]
        top_set = {symbols[idx] for idx in top_indices if coefs[idx] > 0}

        for s in top_set:
            hit_counts[s] = hit_counts.get(s, 0) + 1

        # Jaccard similarity with previous window
        if prev_top_set is not None and (prev_top_set or top_set):
            intersection = len(prev_top_set & top_set)
            union = len(prev_top_set | top_set)
            jaccard_scores.append(intersection / union if union > 0 else 0.0)

        prev_top_set = top_set
        total_windows += 1

    if total_windows == 0:
        return {s: 0.0 for s in symbols}, 0.0

    hit_rates = {s: round(hit_counts[s] / total_windows, 3) for s in symbols}
    stability = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

    return hit_rates, stability


def _composite_scoring(
    symbols: list[str],
    correlations: dict[str, float],
    lasso_coefs: dict[str, float],
    hit_rates: dict[str, float],
    top_n: int,
) -> list[VarietyCandidate]:
    """Combine correlation, LASSO, and stability into a composite probability score."""
    candidates = []

    for symbol in symbols:
        info = FUTURES_VARIETIES.get(symbol)
        if info is None:
            continue

        corr = correlations.get(symbol, 0.0)
        lasso_coef = lasso_coefs.get(symbol, 0.0)
        hit_rate = hit_rates.get(symbol, 0.0)

        # Composite score: 40% |correlation| + 35% |lasso_coef| (normalized) + 25% hit_rate
        # Normalize lasso coef relative to max
        max_lasso = max(abs(v) for v in lasso_coefs.values()) if lasso_coefs else 1.0
        norm_lasso = abs(lasso_coef) / max_lasso if max_lasso > 0 else 0.0

        composite = 0.40 * abs(corr) + 0.35 * norm_lasso + 0.25 * hit_rate
        probability_pct = min(composite * 100, 100.0)

        # Evidence
        evidence = []
        if abs(corr) >= 0.3:
            evidence.append(f"与产品收益相关系数 {corr:.3f}")
        if abs(lasso_coef) > 0:
            evidence.append(f"LASSO 回归系数 {lasso_coef:.4f}（非零，被稀疏模型选中）")
        else:
            evidence.append("LASSO 回归系数为零（未被稀疏模型选中）")
        if hit_rate > 0:
            evidence.append(f"在 {hit_rate:.0%} 的滚动窗口中进入 Top-{top_n}")

        candidates.append(VarietyCandidate(
            symbol=symbol,
            name=info.name,
            sector=info.sector,
            probability_pct=round(probability_pct, 1),
            correlation=corr,
            lasso_coefficient=lasso_coef,
            rolling_hit_rate=hit_rate,
            evidence=evidence,
        ))

    # Sort by composite probability descending
    candidates.sort(key=lambda c: c.probability_pct, reverse=True)
    return candidates


def _aggregate_sectors(candidates: list[VarietyCandidate]) -> list[SectorExposure]:
    """Aggregate variety-level probabilities into sector-level exposure."""
    sector_map: dict[str, list[VarietyCandidate]] = {}
    for c in candidates:
        if c.probability_pct > 5:  # only count non-trivial probabilities
            sector_map.setdefault(c.sector, []).append(c)

    sectors = []
    for sector, varieties in sector_map.items():
        total_prob = sum(v.probability_pct for v in varieties)
        top_variety = max(varieties, key=lambda v: v.probability_pct)
        sectors.append(SectorExposure(
            sector=sector,
            color=SECTOR_COLORS.get(sector, "#999999"),
            total_probability=round(total_prob, 1),
            top_variety=top_variety.name,
            variety_count=len(varieties),
        ))

    sectors.sort(key=lambda s: s.total_probability, reverse=True)
    return sectors


def _generate_evidence(
    candidates: list[VarietyCandidate],
    sectors: list[SectorExposure],
    stability: float,
    total_varieties: int,
) -> list[str]:
    """Generate summary evidence for the identification result."""
    evidence = [f"共扫描 {total_varieties} 个活跃品种"]

    if candidates:
        top3 = candidates[:3]
        names = "、".join(f"{c.name}({c.probability_pct}%)" for c in top3)
        evidence.append(f"概率最高的三个品种：{names}")

    if sectors:
        top_sector = sectors[0]
        evidence.append(f"最集中的板块：{top_sector.sector}（累计概率 {top_sector.total_probability}%）")

    evidence.append(f"结论跨窗口稳定性：{stability:.0%}")

    return evidence
