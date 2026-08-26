"""Factor library base class and common utilities.

Every factor in the library inherits from FactorBase and implements the
`compute` method, which takes a panel of per-variety OHLCV data and returns
a single daily factor-return series (the "factor NAV" in return form).

Common quantitative helpers (volatility targeting, cross-sectional
standardization, portfolio construction) live here so individual factor
modules stay focused on signal logic.

Formula notation convention
---------------------------
All factor ``meta.formula`` fields use a compact shared vocabulary so that
each formula only needs to define the *signal* (1-3 lines); the portfolio
construction pipeline is implied by a trailing arrow annotation.

Standard operators (defined once, used everywhere):

    z(x)        cross-sectional z-score (per date, across varieties)
    LS(q%)      long top q%, short bottom q%, equal-weight, gross exposure = 1
    VT(σ*)      volatility-target at σ* annualized, leverage cap 3×

Shorthand:

    P_t         close price at time t
    r_t         daily return  (P_t / P_{t-1} - 1)
    R_n         n-day cumulative return  P_t / P_{t-n} - 1
    MA_n        n-day simple moving average of P
    σ_n         annualized n-day realized vol = std(r, n) × √252
    ρ_n(·,·)   n-day rolling Pearson correlation

Example reading:  ``x_t = ρ₄₀(ΔP, ΔV) · sign(R₂₀)  ⇒  z(x) → LS(20%)``
means "the signal is the 40-day volume-price correlation times the sign of
20-day momentum; cross-sectionally z-score it, then long the top quintile
and short the bottom quintile."
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Factor metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorMeta:
    """Static descriptor for a factor."""

    name: str            # machine key, e.g. "trend_120_250"
    display_name: str    # Chinese display, e.g. "长期规则(趋势)"
    category: str        # "量价" | "基本面"
    description: str     # one-line explanation
    params: dict = field(default_factory=dict)  # default parameters
    # Rich detail shown in the factor "公式与代码" dialog (all optional):
    signal_rule: str = ""   # plain-language "when does it trade"
    formula: str = ""       # LaTeX math (rendered by KaTeX on the frontend)
    derivation: str = ""    # economic rationale / derivation prose
    frequency: str = "daily"  # "daily" | "weekly" | "monthly"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class FactorBase(ABC):
    """Interface that all factor implementations must satisfy."""

    meta: FactorMeta

    @abstractmethod
    def compute(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        """Compute factor returns from a panel of variety OHLCV data.

        Parameters
        ----------
        panels : dict[symbol, DataFrame]
            Each DataFrame has columns [date, open, high, low, close, volume]
            sorted by date ascending.  Only varieties with sufficient data
            should be included; the caller filters beforehand.

        Returns
        -------
        pd.Series
            Factor returns indexed by date (datetime), named after the factor.
            Most factors are daily; see ``meta.frequency`` for exceptions.
        """

    def compute_nav(self, panels: dict[str, pd.DataFrame]) -> pd.Series:
        """Convenience: compute cumulative NAV (base=1.0) from returns."""
        rets = self.compute(panels)
        nav = (1 + rets).cumprod()
        nav.name = f"{self.meta.name}_nav"
        return nav

    def requires_market_panels(self) -> bool:
        """Whether this factor needs the daily futures OHLCV panels to build."""
        return True

    def input_note(self) -> str | None:
        """Return a note when the factor cannot run due to missing external input.

        Factors that depend on user-supplied data (e.g. warehouse receipts,
        inventory CSVs) override this to explain what is missing, so the build
        orchestrator can skip them gracefully instead of logging a failure.
        Returns None when the factor is ready to compute.
        """
        return None

    def compute_raw_signal(self, panels: dict[str, pd.DataFrame]) -> pd.Series | pd.DataFrame | None:
        """Return the pre-portfolio signal when the factor exposes one.

        ``None`` is explicit: it tells the build audit that this factor has
        not yet implemented a raw-signal artifact, instead of silently
        pretending its baseline return is the raw signal.
        """
        return None

    def compute_weights(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """Return target weights for transaction-cost accounting when available."""
        return None

    def compute_contributions(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return the daily per-variety contribution to the factor return.

        Output is a DataFrame indexed by date with one column per variety;
        each cell is that variety's contribution to the factor's daily return,
        so ``compute_contributions(panels).sum(axis=1)`` reconciles with
        ``compute(panels)`` (up to the no-position masking).  Used to
        decompose a factor's return by sector (板块盈亏 attribution).

        Factors that do not yet support attribution raise NotImplementedError.
        """
        raise NotImplementedError(
            f"factor '{self.meta.name}' does not support contribution attribution"
        )


# ---------------------------------------------------------------------------
# Common quantitative helpers
# ---------------------------------------------------------------------------

def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling realized volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def vol_target_weights(
    returns: pd.Series,
    target_vol: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Compute a scalar leverage series that targets *target_vol* annualized.

    Uses trailing *lookback*-day realized vol.  Caps leverage at
    *max_leverage* to avoid extreme positions during low-vol regimes.
    """
    ann_vol = realized_vol(returns, lookback)
    leverage = target_vol / ann_vol.replace(0, np.nan)
    leverage = leverage.clip(upper=max_leverage).fillna(0.0)
    return leverage


def cross_sectional_zscore(signals: pd.DataFrame) -> pd.DataFrame:
    """Standardize signals cross-sectionally (per date, across varieties).

    Input: DataFrame with DatetimeIndex and one column per variety.
    Output: same shape, each row z-scored (mean=0, std=1).
    """
    row_mean = signals.mean(axis=1)
    row_std = signals.std(axis=1).replace(0, np.nan)
    return signals.sub(row_mean, axis=0).div(row_std, axis=0)


def long_short_portfolio(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Compute portfolio return from a weight matrix and return matrix.

    Both inputs share the same DatetimeIndex and column set (varieties).
    Weights are applied to next-day returns (shift to avoid look-ahead).
    """
    # Align columns
    common_cols = weights.columns.intersection(returns.columns)
    w = weights[common_cols]
    r = returns[common_cols]
    # Weights determined at t, applied to return at t+1
    port_ret = (w.shift(1) * r).sum(axis=1)
    # Warm-up days have all-zero weights and would otherwise contribute a
    # 0.0 return that survives dropna(), polluting win-rate ((ret > 0).mean())
    # and diluting the mean used for annualized return.  Mark no-position
    # days as NaN so they are truly excluded from the factor return series.
    has_position = w.shift(1).abs().sum(axis=1) > 1e-12
    port_ret = port_ret.where(has_position)
    return port_ret.dropna()


def contributions_from_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Per-variety daily contribution matrix (date × variety).

    Cell (t, v) = w_{v, t-1} * r_{v, t}: the contribution of variety ``v`` to
    the factor's return on day ``t``.  Row-summing reproduces the long-short
    portfolio return (``long_short_portfolio``) before the no-position masking,
    so this matrix is the natural basis for sector-level attribution.
    """
    common_cols = weights.columns.intersection(returns.columns)
    return weights[common_cols].shift(1) * returns[common_cols]


def turnover_and_costs(weights: pd.DataFrame, gross_returns: pd.Series, cost_bps: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """One-way turnover and same-date trading cost for lagged target weights."""
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    turnover = weights.fillna(0.0).diff().abs().sum(axis=1).reindex(gross_returns.index).fillna(0.0)
    return turnover, turnover * (cost_bps / 10_000.0)


def equal_weight_long_short(
    signal: pd.DataFrame,
    quantile: float = 0.2,
) -> pd.DataFrame:
    """Construct equal-weight long-short weights from a signal matrix.

    Long the top *quantile* fraction, short the bottom *quantile* fraction.
    Weights are normalized so gross exposure = 1 (i.e., 0.5 long + 0.5 short).
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for dt in signal.index:
        row = signal.loc[dt].dropna()
        if len(row) < 4:
            continue
        n_long = max(1, int(len(row) * quantile))
        sorted_row = row.sort_values(ascending=False)
        longs = sorted_row.index[:n_long]
        shorts = sorted_row.index[-n_long:]
        weights.loc[dt, longs] = 1.0 / (2 * n_long)
        weights.loc[dt, shorts] = -1.0 / (2 * n_long)

    return weights


def build_close_panel(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract close prices into a unified panel (date x variety)."""
    closes = {}
    for symbol, df in panels.items():
        if df.empty or "close" not in df.columns:
            continue
        s = df.set_index("date")["close"]
        s.index = pd.to_datetime(s.index)
        closes[symbol] = s
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index()


def build_volume_panel(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract volume into a unified panel (date x variety)."""
    vols = {}
    for symbol, df in panels.items():
        if df.empty or "volume" not in df.columns:
            continue
        s = df.set_index("date")["volume"]
        s.index = pd.to_datetime(s.index)
        vols[symbol] = s
    if not vols:
        return pd.DataFrame()
    return pd.DataFrame(vols).sort_index()
