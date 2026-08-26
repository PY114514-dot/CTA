"""Versioned CTA-BARRA factor contract and real term-structure carry data.

The bundle is the single read boundary for the CTA attribution models.  It
keeps factor identity/provenance separate from a model's feature engineering
and, importantly, does not rename a price-return proxy as Carry.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.config import DATA_DIRECTORY
from app.services.factor_library import cache


CTA_FACTOR_BUNDLE_VERSION = "cta_factor_bundle_v1"
CTA_FACTOR_BUNDLE_MODEL_VERSION = "cta-barra-v1.0"
CTA_FACTOR_NAMES = (
    "trend",
    "short_term_trend_20",
    "cross_section_mom",
    "term_structure_carry",
    "volatility_state",
    "mean_reversion_5d",
)
COMMODITY_ARBITRAGE_FACTOR_NAMES = CTA_FACTOR_NAMES + ("calendar_spread_momentum",)
CTA_FACTOR_DIRECTORY = DATA_DIRECTORY / "cta_factors"
REAL_CARRY_PATH = CTA_FACTOR_DIRECTORY / "term_structure_carry.csv"
REAL_CARRY_MANIFEST_PATH = CTA_FACTOR_DIRECTORY / "term_structure_carry_manifest.json"

REAL_CARRY_COLUMNS = (
    "observation_date",
    "symbol",
    "near_contract",
    "far_contract",
    "near_price",
    "far_price",
    "near_next_price",
    "far_next_price",
    "near_days_to_expiry",
    "far_days_to_expiry",
    "holding_return",
    "available_date",
)

_FACTOR_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "trend",
        "label": "中长期趋势",
        "group": "strategy_style",
        "source": "cached_factor_library",
        "availability": "cache_manifest",
        "continuous_contract_rule": "provider_main_continuous",
        "frequency": "daily_then_compound",
        "calculation": "existing_versioned_factor_library_series",
    },
    {
        "name": "short_term_trend_20",
        "label": "短期动量",
        "group": "strategy_style",
        "source": "cached_factor_library",
        "availability": "cache_manifest",
        "continuous_contract_rule": "provider_main_continuous",
        "frequency": "daily_then_compound",
        "calculation": "existing_versioned_factor_library_series",
    },
    {
        "name": "cross_section_mom",
        "label": "截面动量",
        "group": "strategy_style",
        "source": "cached_factor_library",
        "availability": "cache_manifest",
        "continuous_contract_rule": "provider_main_continuous",
        "frequency": "daily_then_compound",
        "calculation": "existing_versioned_factor_library_series",
    },
    {
        "name": "mean_reversion_5d",
        "label": "短期反转",
        "group": "strategy_style",
        "source": "cached_factor_library",
        "availability": "cache_manifest",
        "continuous_contract_rule": "provider_main_continuous",
        "frequency": "daily_then_compound",
        "calculation": "existing_versioned_factor_library_series",
    },
    {
        "name": "term_structure_carry",
        "label": "真实 Carry / 期限结构",
        "group": "strategy_style",
        "source": "reviewed_term_structure_input",
        "availability": "term_structure_carry.csv",
        "continuous_contract_rule": "input_must_declare_near_far_contracts",
        "frequency": "daily_then_compound",
        "calculation": "annualized_near_far_slope_with_next_week_holding_return",
    },
    {
        "name": "calendar_spread_momentum",
        "label": "跨期价差动量",
        "group": "strategy_style",
        "source": "reviewed_term_structure_input",
        "availability": "term_structure_carry.csv",
        "continuous_contract_rule": "input_must_declare_near_far_contracts",
        "frequency": "weekly",
        "calculation": "four_week_curve_slope_change_long_short_next_week_calendar_spread_return",
    },
    {
        "name": "volatility_state",
        "label": "波动率状态",
        "group": "market_state",
        "source": "derived_from_versioned_factor_returns",
        "availability": "derived_at_model_cutoff",
        "continuous_contract_rule": "inherits_input_factor_contract",
        "frequency": "daily_then_compound",
        "calculation": "lagged_rolling_volatility_of_registered_momentum_factor",
    },
)


class CtaFactorContractError(ValueError):
    """Raised when a factor input cannot satisfy the versioned contract."""


def _as_date(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as error:
        raise CtaFactorContractError(f"{field} 必须是 ISO 日期") from error


def _normalise_carry_row(row: Any) -> dict[str, Any]:
    values = row.model_dump() if hasattr(row, "model_dump") else dict(row)
    missing = [column for column in REAL_CARRY_COLUMNS if column not in values]
    if missing:
        raise CtaFactorContractError(f"真实 Carry 缺少字段：{', '.join(missing)}")
    observation_date = _as_date(values["observation_date"], "observation_date")
    available_date = _as_date(values["available_date"], "available_date")
    if available_date > observation_date:
        raise CtaFactorContractError("available_date 不能晚于 observation_date")
    try:
        near_price = float(values["near_price"])
        far_price = float(values["far_price"])
        near_next_price = float(values["near_next_price"])
        far_next_price = float(values["far_next_price"])
        near_days = float(values["near_days_to_expiry"])
        far_days = float(values["far_days_to_expiry"])
        holding_return = float(values["holding_return"])
    except (TypeError, ValueError) as error:
        raise CtaFactorContractError("真实 Carry 数值字段格式无效") from error
    if not all(math.isfinite(value) for value in (near_price, far_price, near_next_price, far_next_price, near_days, far_days, holding_return)):
        raise CtaFactorContractError("真实 Carry 数值字段必须有限")
    if min(near_price, far_price, near_next_price, far_next_price) <= 0:
        raise CtaFactorContractError("近月和远月价格必须为正")
    if far_days <= near_days:
        raise CtaFactorContractError("far_days_to_expiry 必须大于 near_days_to_expiry")
    if holding_return < -1:
        raise CtaFactorContractError("holding_return 不能低于 -100%")
    symbol = str(values["symbol"]).strip()
    if not symbol:
        raise CtaFactorContractError("symbol 不能为空")
    near_contract = str(values["near_contract"]).strip()
    far_contract = str(values["far_contract"]).strip()
    if not near_contract or not far_contract:
        raise CtaFactorContractError("近远月合约代码不能为空")
    return {
        "observation_date": observation_date,
        "symbol": symbol,
        "near_contract": near_contract,
        "far_contract": far_contract,
        "near_price": near_price,
        "far_price": far_price,
        "near_next_price": near_next_price,
        "far_next_price": far_next_price,
        "near_days_to_expiry": near_days,
        "far_days_to_expiry": far_days,
        "holding_return": holding_return,
        "available_date": available_date,
    }


def validate_carry_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Validate and sort point-in-time near/far contract observations."""
    normalised = [_normalise_carry_row(row) for row in rows]
    if not normalised:
        raise CtaFactorContractError("真实 Carry 至少需要一条观测")
    keys = [(row["observation_date"], row["symbol"]) for row in normalised]
    if len(keys) != len(set(keys)):
        raise CtaFactorContractError("同一日期和品种只能有一条真实 Carry 观测")
    return sorted(normalised, key=lambda row: (row["observation_date"], row["symbol"]))


def compute_real_carry_returns(
    rows: Iterable[Any],
    *,
    quantile: float = 0.2,
    minimum_symbols: int = 4,
) -> pd.Series:
    """Build a long-short factor from observed near/far term structures.

    Positive carry is defined as backwardation: ``-log(far / near)`` scaled
    by the expiry-day gap.  The signal at date *t* is multiplied by the
    supplied next-period holding return for the same variety.  No product
    NAV or peer cross-sectional median is used.
    """
    if not 0 < quantile <= 0.5:
        raise CtaFactorContractError("quantile 必须在 (0, 0.5] 内")
    if minimum_symbols < 2:
        raise CtaFactorContractError("minimum_symbols 至少为 2")
    validated = validate_carry_rows(rows)
    by_date: dict[date, list[tuple[float, float]]] = defaultdict(list)
    for row in validated:
        slope = -math.log(row["far_price"] / row["near_price"]) * 365.0 / (
            row["far_days_to_expiry"] - row["near_days_to_expiry"]
        )
        by_date[row["observation_date"]].append((slope, row["holding_return"]))

    output: dict[date, float] = {}
    for observation_date, values in by_date.items():
        if len(values) < minimum_symbols:
            continue
        ordered = sorted(values, key=lambda item: item[0])
        leg_size = max(1, int(math.floor(len(ordered) * quantile)))
        short_leg = [item[1] for item in ordered[:leg_size]]
        long_leg = [item[1] for item in ordered[-leg_size:]]
        output[observation_date] = float(sum(long_leg) / len(long_leg) - sum(short_leg) / len(short_leg))
    if not output:
        return pd.Series(dtype=float, name="term_structure_carry")
    result = pd.Series(output, dtype=float)
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result.name = "term_structure_carry"
    return result


def compute_calendar_spread_momentum(
    rows: Iterable[Any],
    *,
    lookback_periods: int = 4,
    quantile: float = 0.2,
    minimum_symbols: int = 4,
) -> pd.Series:
    """Long-short calendar-spread return after a four-week curve-slope change.

    The signal is known before the following weekly holding period.  The payoff uses
    the same named far and near contracts, so it does not infer a spread from
    continuous contracts or undisclosed positions.
    """
    if lookback_periods < 1:
        raise CtaFactorContractError("lookback_periods 必须至少为 1")
    validated = validate_carry_rows(rows)
    history: dict[str, list[float]] = defaultdict(list)
    by_date: dict[date, list[tuple[float, float]]] = defaultdict(list)
    for row in validated:
        slope = math.log(row["far_price"] / row["near_price"])
        prior = history[row["symbol"]]
        if len(prior) >= lookback_periods:
            signal = slope - prior[-lookback_periods]
            spread_return = row["far_next_price"] / row["far_price"] - row["near_next_price"] / row["near_price"]
            by_date[row["observation_date"]].append((signal, spread_return))
        prior.append(slope)
    output: dict[date, float] = {}
    for observation_date, values in by_date.items():
        if len(values) < minimum_symbols:
            continue
        ordered = sorted(values, key=lambda item: item[0])
        leg_size = max(1, int(math.floor(len(ordered) * quantile)))
        output[observation_date] = float(
            sum(item[1] for item in ordered[-leg_size:]) / leg_size
            - sum(item[1] for item in ordered[:leg_size]) / leg_size
        )
    result = pd.Series(output, dtype=float, name="calendar_spread_momentum")
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def save_real_carry_rows(
    rows: Iterable[Any],
    *,
    source: str,
    data_version: str,
    continuous_contract_rule: str,
) -> dict[str, Any]:
    """Persist validated raw Carry inputs and an immutable manifest record."""
    validated = validate_carry_rows(rows)
    CTA_FACTOR_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with REAL_CARRY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REAL_CARRY_COLUMNS))
        writer.writeheader()
        for row in validated:
            writer.writerow({key: row[key].isoformat() if isinstance(row[key], date) else row[key] for key in REAL_CARRY_COLUMNS})
    carry_series = compute_real_carry_returns(validated)
    calendar_series = compute_calendar_spread_momentum(validated)
    content = {
        "bundle_version": CTA_FACTOR_BUNDLE_VERSION,
        "factor_name": "term_structure_carry",
        "source": source,
        "data_version": data_version,
        "continuous_contract_rule": continuous_contract_rule,
        "available_date": max(row["available_date"] for row in validated).isoformat(),
        "observation_start": min(row["observation_date"] for row in validated).isoformat(),
        "observation_end": max(row["observation_date"] for row in validated).isoformat(),
        "rows": len(validated),
        "factor_rows": {"term_structure_carry": len(carry_series), "calendar_spread_momentum": len(calendar_series)},
        "parameters": {"quantile": 0.2, "minimum_symbols": 4, "calendar_spread_lookback_periods": 4},
    }
    content["version_id"] = (
        f"{CTA_FACTOR_BUNDLE_VERSION}:{data_version}:"
        f"{content['observation_start']}:{content['observation_end']}:{len(validated)}"
    )
    REAL_CARRY_MANIFEST_PATH.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return content


def load_real_carry_rows(*, as_of_date: date | None = None) -> list[dict[str, Any]]:
    """Load only point-in-time Carry inputs available by the requested cutoff."""
    if not REAL_CARRY_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with REAL_CARRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            normalised = _normalise_carry_row(row)
            if as_of_date is None or (
                normalised["observation_date"] <= as_of_date
                and normalised["available_date"] <= as_of_date
            ):
                rows.append(normalised)
    return validate_carry_rows(rows) if rows else []


def load_real_carry_factor_returns(*, as_of_date: date | None = None) -> pd.Series:
    rows = load_real_carry_rows(as_of_date=as_of_date)
    return compute_real_carry_returns(rows) if rows else pd.Series(dtype=float, name="term_structure_carry")


def load_calendar_spread_momentum(*, as_of_date: date | None = None) -> pd.Series:
    rows = load_real_carry_rows(as_of_date=as_of_date)
    return compute_calendar_spread_momentum(rows) if rows else pd.Series(dtype=float, name="calendar_spread_momentum")


def _cached_factor_provenance(name: str, as_of_date: date | None) -> dict[str, Any]:
    entry = cache.get_cache_info(name)
    if entry is None:
        return {"name": name, "status": "not_covered", "source": "cached_factor_library"}
    end = _as_date(entry.get("end"), "cache.end") if entry.get("end") else None
    return {
        "name": name,
        "status": "available",
        "source": "cached_factor_library",
        "available_date": entry.get("start"),
        "calculation_end_date": min(end, as_of_date).isoformat() if end and as_of_date else end.isoformat() if end else None,
        "parameters": entry.get("params", {}),
        "data_version": entry.get("data_version", {}),
        "return_version": entry.get("active_return_version"),
        "continuous_contract_rule": "provider_main_continuous",
        "frequency": "daily_then_compound",
    }


def get_cta_factor_bundle(*, as_of_date: date | None = None) -> dict[str, Any]:
    """Return the current bundle manifest without fabricating unavailable data."""
    factors: list[dict[str, Any]] = []
    for definition in _FACTOR_DEFINITIONS:
        item = dict(definition)
        if definition["name"] in {"term_structure_carry", "calendar_spread_momentum"}:
            manifest = json.loads(REAL_CARRY_MANIFEST_PATH.read_text(encoding="utf-8")) if REAL_CARRY_MANIFEST_PATH.is_file() else {}
            series = get_factor_series(definition["name"], as_of_date=as_of_date)
            item.update({
                    "status": "available" if series is not None and not series.empty else "not_covered",
                "available_date": manifest.get("available_date"),
                "data_version": manifest.get("data_version"),
                "return_version": manifest.get("version_id"),
                "parameters": manifest.get("parameters", {"quantile": 0.2, "minimum_symbols": 4}),
                "cutoff_date": as_of_date.isoformat() if as_of_date else manifest.get("observation_end"),
            })
        elif definition["name"] == "volatility_state":
            trend = _cached_factor_provenance("trend", as_of_date)
            item.update({"status": "available" if trend["status"] == "available" else "not_covered", "derived_from": "trend", "cutoff_date": as_of_date.isoformat() if as_of_date else None})
        else:
            item.update(_cached_factor_provenance(definition["name"], as_of_date))
        factors.append(item)
    return {
        "bundle_version": CTA_FACTOR_BUNDLE_VERSION,
        "model_version": CTA_FACTOR_BUNDLE_MODEL_VERSION,
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "factors": factors,
        "alignment_contract": {
            "date_rule": "economic_period_intersection",
            "lag_rule": "factor_value_must_be_available_on_or_before_observation_date",
            "missing_data": "drop_incomplete_period_and_report_warning",
            "product_cross_section_fallback": False,
        },
    }


def get_factor_series(name: str, *, as_of_date: date | None = None) -> pd.Series | None:
    """Load one factor from the bundle with a hard information cutoff."""
    if name == "term_structure_carry":
        series = load_real_carry_factor_returns(as_of_date=as_of_date)
        return series if not series.empty else None
    if name == "calendar_spread_momentum":
        series = load_calendar_spread_momentum(as_of_date=as_of_date)
        return series if not series.empty else None
    if name == "volatility_state":
        base = get_factor_series("trend", as_of_date=as_of_date)
        if base is None or base.empty:
            return None
        # Lag before calculating state so the current factor return cannot
        # influence the volatility input for the same observation.
        result = base.shift(1).rolling(20, min_periods=20).std()
        result.name = name
        return result.dropna()
    series = __import__("app.services.factor_library", fromlist=["get_factor_series"]).get_factor_series(name, risk_profile="baseline")
    if series is None or series.empty:
        return None
    if as_of_date is not None:
        series = series[series.index <= pd.Timestamp(as_of_date)]
    return series
