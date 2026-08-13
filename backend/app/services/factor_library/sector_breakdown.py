"""Sector-level profit/loss attribution for the factor library.

Given a date range, loads cached per-variety contribution matrices, aggregates
them by sector (板块), and produces both structured data and report-style prose
similar to the Guotai Junan Futures weekly format:

    "本周上涨 X%；在黑色、有色板块有所盈利，在能化板块呈现亏损。"
"""

import logging
from datetime import date

import pandas as pd

from app.services.factor_library import cache
from app.services.factor_library.registry import get_core_varieties

logger = logging.getLogger(__name__)

# Display names for sectors (registry uses short forms)
SECTOR_DISPLAY: dict[str, str] = {
    "黑色": "黑色",
    "有色": "有色金属",
    "贵金属": "贵金属",
    "能化": "能源化工",
    "农产品": "农产品",
    "金融": "金融",
}


def _symbol_to_sector() -> dict[str, str]:
    """Build symbol -> sector mapping from the core variety registry."""
    return {sym: info.sector for sym, info in get_core_varieties().items()}


def compute_sector_breakdown(
    factor_name: str,
    start: date,
    end: date,
) -> dict | None:
    """Compute sector-level P&L attribution for a single factor.

    Returns None if contribution data is unavailable.

    Result dict:
        factor_name, display_name, period_return (cumulative %),
        sectors: [{sector, display_sector, contribution (cumulative %), direction}],
        prose: str
    """
    from app.services.factor_library import get_factor

    contrib = cache.load_factor_contributions(factor_name, start, end)
    if contrib is None or contrib.empty:
        return None

    factor = get_factor(factor_name)
    display_name = factor.meta.display_name if factor else factor_name

    # Map columns (symbols) to sectors
    sym_sector = _symbol_to_sector()

    # Aggregate by sector: sum contributions per sector per day, then cumulate
    sector_daily: dict[str, pd.Series] = {}
    unmapped_cols = []
    for col in contrib.columns:
        sector = sym_sector.get(col)
        if sector is None:
            unmapped_cols.append(col)
            continue
        if sector not in sector_daily:
            sector_daily[sector] = pd.Series(0.0, index=contrib.index)
        sector_daily[sector] = sector_daily[sector] + contrib[col].fillna(0.0)

    if unmapped_cols:
        logger.debug("Unmapped symbols in contributions for '%s': %s", factor_name, unmapped_cols)

    # Cumulative return per sector over the period
    sector_cum: dict[str, float] = {}
    for sector, daily in sector_daily.items():
        sector_cum[sector] = float(daily.sum())

    # Total factor return over the period
    total_return = sum(sector_cum.values())

    # Build sector list sorted by contribution descending
    sectors = []
    for sector, cum in sorted(sector_cum.items(), key=lambda x: x[1], reverse=True):
        sectors.append({
            "sector": sector,
            "display_sector": SECTOR_DISPLAY.get(sector, sector),
            "contribution": round(cum * 100, 4),  # percentage
            "direction": "盈利" if cum > 0 else ("亏损" if cum < 0 else "持平"),
        })

    # Generate prose
    prose = _generate_prose(display_name, total_return, sectors, start, end)

    return {
        "factor_name": factor_name,
        "display_name": display_name,
        "period_return": round(total_return * 100, 4),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sectors": sectors,
        "prose": prose,
    }


def compute_all_sector_breakdowns(
    start: date,
    end: date,
    factor_names: list[str] | None = None,
) -> list[dict]:
    """Compute sector breakdowns for all (or selected) factors.

    Skips factors without cached contribution data.
    """
    from app.services.factor_library import get_all_factors

    all_factors = get_all_factors()
    targets = factor_names or [f["name"] for f in all_factors]

    results = []
    for name in targets:
        breakdown = compute_sector_breakdown(name, start, end)
        if breakdown is not None:
            results.append(breakdown)

    return results


def _generate_prose(
    display_name: str,
    total_return: float,
    sectors: list[dict],
    start: date,
    end: date,
) -> str:
    """Generate report-style prose for a factor's sector attribution.

    Example output:
        "趋势因子区间累计上涨 1.23%；在黑色、有色金属板块有所盈利，
         在能源化工板块呈现亏损。"
    """
    pct = total_return * 100
    if pct > 0:
        direction_word = "上涨"
    elif pct < 0:
        direction_word = "下跌"
    else:
        direction_word = "持平"

    parts = [f"{display_name}因子区间累计{direction_word} {abs(pct):.2f}%"]

    profitable = [s["display_sector"] for s in sectors if s["contribution"] > 0.001]
    losing = [s["display_sector"] for s in sectors if s["contribution"] < -0.001]

    if profitable:
        parts.append(f"在{'、'.join(profitable)}板块有所盈利")
    if losing:
        parts.append(f"在{'、'.join(losing)}板块呈现亏损")

    if not profitable and not losing:
        parts.append("各板块贡献均不显著")

    return "；".join(parts) + "。"
