"""Import verifiable SHFE near/far contract observations for CTA Carry."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.services.market_data.akshare_provider import AKShareProvider


# SHFE publishes historical contract expiry lists reliably.  Other exchanges
# are intentionally excluded until their contract-list endpoints are equally
# reproducible.
SHFE_CARRY_SYMBOLS = ("rb", "cu", "al", "zn", "au", "ag", "fu", "bu")
_MIN_NEAR_DAYS = 7
_MIN_TERM_GAP_DAYS = 45


def import_shfe_carry(
    start: date,
    end: date,
    *,
    provider: AKShareProvider | None = None,
    symbols: tuple[str, ...] = SHFE_CARRY_SYMBOLS,
) -> list[dict]:
    """Build weekly Carry inputs from exchange expiry data and named contracts.

    A row is emitted only where both contracts, their exchange-published expiry
    dates, and next-week prices for the exact same contracts are available.
    """
    if start >= end:
        raise ValueError("结束日期必须晚于开始日期")
    provider = provider or AKShareProvider()
    price_books: dict[date, pd.DataFrame] = {}
    rows: list[dict] = []
    observation_days = pd.date_range(start, end, freq="W-FRI").date

    for observation_date in observation_days:
        contracts = _contracts_for_date(provider, observation_date)
        if contracts.empty:
            continue
        for symbol in symbols:
            pair = _select_contract_pair(contracts, symbol, observation_date)
            if pair is None:
                continue
            near, far = pair
            price_date, prices = _prices_for_date(provider, price_books, observation_date)
            # Products in this factor model report weekly NAV.  Holding only
            # one trading day after a weekly signal mismatches that horizon.
            next_price_date, next_prices = _prices_for_date(provider, price_books, price_date + timedelta(days=7))
            near_price = _contract_price(prices, near["contract"])
            far_price = _contract_price(prices, far["contract"])
            next_price = _contract_price(next_prices, near["contract"])
            far_next_price = _contract_price(next_prices, far["contract"])
            if near_price is None or far_price is None or next_price is None or far_next_price is None:
                continue
            rows.append({
                # Timestamp the return when it is realised, rather than at the
                # earlier signal date, so factor regression cannot see ahead.
                "observation_date": next_price_date,
                "symbol": symbol,
                "near_contract": near["contract"],
                "far_contract": far["contract"],
                "near_price": near_price,
                "far_price": far_price,
                "near_next_price": next_price,
                "far_next_price": far_next_price,
                "near_days_to_expiry": (near["expiry"] - next_price_date).days,
                "far_days_to_expiry": (far["expiry"] - next_price_date).days,
                "holding_return": next_price / near_price - 1.0,
                "available_date": next_price_date,
            })
    return rows


def assert_carry_coverage(rows: list[dict], start: date, end: date) -> None:
    """Reject partial downloads instead of silently replacing the factor."""
    realised_dates = {row["observation_date"] for row in rows}
    if not realised_dates:
        raise ValueError("AkShare 未取得有效的 Carry 观测")
    if min(realised_dates) > start + timedelta(days=14) or max(realised_dates) < end - timedelta(days=14):
        raise ValueError("AkShare 返回不完整；未覆盖请求区间，未写入 Carry 数据")
    incomplete = [day for day in realised_dates if sum(row["observation_date"] == day for row in rows) < 4]
    if incomplete:
        raise ValueError("AkShare 返回的部分日期不足 4 个品种，未写入 Carry 数据")


def _contracts_for_date(provider: AKShareProvider, on_date: date) -> pd.DataFrame:
    """Use the latest published SHFE list at or immediately before the date."""
    for offset in range(4):
        result = provider.get_shfe_contracts(on_date - timedelta(days=offset))
        if not result.empty:
            return result
    return pd.DataFrame(columns=["contract", "symbol", "expiry"])


def _select_contract_pair(contracts: pd.DataFrame, symbol: str, on_date: date) -> tuple[dict, dict] | None:
    candidates = contracts[contracts["symbol"] == symbol].copy()
    candidates = candidates[candidates["expiry"] > on_date + timedelta(days=_MIN_NEAR_DAYS)]
    if len(candidates) < 2:
        return None
    near = candidates.iloc[0].to_dict()
    far_candidates = candidates[candidates["expiry"] >= near["expiry"] + timedelta(days=_MIN_TERM_GAP_DAYS)]
    if far_candidates.empty:
        return None
    return near, far_candidates.iloc[0].to_dict()


def _prices_for_date(
    provider: AKShareProvider,
    cache: dict[date, pd.DataFrame],
    on_date: date,
) -> tuple[date, pd.DataFrame]:
    for offset in range(4):
        candidate = on_date + timedelta(days=offset)
        if candidate not in cache:
            cache[candidate] = provider.get_shfe_daily_prices(candidate)
        if not cache[candidate].empty:
            return candidate, cache[candidate]
    return on_date, pd.DataFrame(columns=["contract", "close"])


def _contract_price(prices: pd.DataFrame, contract: str) -> float | None:
    values = prices.loc[prices["contract"] == contract, "close"]
    return float(values.iloc[0]) if not values.empty else None
