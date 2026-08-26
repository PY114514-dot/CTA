"""Versioned local factor inputs for equity and option attribution.

Inputs are declared factor series, not inferred holdings.  Keeping them local
and point-in-time dated makes every later regression reproducible.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.config import ATTRIBUTION_FACTOR_CONTRACT_DIRECTORY


CONTRACTS: dict[str, dict[str, Any]] = {
    "equity_quant": {
        "version": "equity_quant_v1",
        "label": "股票/量化基础因子",
        "factors": (
            ("equity_market", "股票市场收益"),
            ("equity_size_spread", "规模风格收益"),
            ("equity_value_spread", "价值风格收益"),
            ("equity_quality_spread", "质量风格收益"),
            ("equity_momentum", "股票动量收益"),
            ("equity_volatility", "股票波动率变化"),
        ),
    },
    "options_volatility": {
        "version": "options_volatility_v1",
        "label": "期权/波动率基础因子",
        "factors": (
            ("option_underlying_market", "标的市场收益"),
            ("option_realized_volatility_change", "实现波动率变化"),
            ("option_implied_volatility_change", "隐含波动率变化"),
            ("option_iv_term_structure_change", "隐含波动率期限结构变化"),
            ("option_iv_skew_change", "隐含波动率偏度变化"),
        ),
    },
}


class MultiAssetFactorContractError(ValueError):
    pass


def _paths(pool: str) -> tuple[Path, Path]:
    if pool not in CONTRACTS:
        raise MultiAssetFactorContractError("仅支持股票/量化或期权/波动率因子合同")
    return (
        ATTRIBUTION_FACTOR_CONTRACT_DIRECTORY / f"{pool}.csv",
        ATTRIBUTION_FACTOR_CONTRACT_DIRECTORY / f"{pool}.json",
    )


def contract_metadata(pool: str, *, as_of_date: date | None = None) -> dict[str, Any]:
    contract = CONTRACTS[pool]
    csv_path, manifest_path = _paths(pool)
    rows = load_factor_rows(pool, as_of_date=as_of_date)
    counts = Counter(row["factor_name"] for row in rows)
    expected = [name for name, _ in contract["factors"]]
    # 20 aligned observations is also the regression engine's minimum.
    covered = bool(rows) and all(counts[name] >= 20 for name in expected)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return {
        "product_pool": pool,
        "contract_version": contract["version"],
        "label": contract["label"],
        "status": "available" if covered else "not_covered",
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "source": manifest.get("source"),
        "data_version": manifest.get("data_version"),
        "factors": [
            {"name": name, "label": label, "status": "available" if counts[name] >= 20 else "not_covered", "observations": counts[name]}
            for name, label in contract["factors"]
        ],
        "storage": str(csv_path),
    }


def _normalise_row(value: Any, expected: set[str]) -> dict[str, Any]:
    row = value.model_dump() if hasattr(value, "model_dump") else dict(value)
    try:
        observation_date = date.fromisoformat(str(row["observation_date"]))
        available_date = date.fromisoformat(str(row["available_date"]))
        factor_value = float(row["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise MultiAssetFactorContractError("因子行需要有效的日期与数值") from error
    factor_name = str(row.get("factor_name", "")).strip()
    if factor_name not in expected:
        raise MultiAssetFactorContractError(f"因子 {factor_name or '—'} 不属于当前合同")
    if available_date > observation_date:
        raise MultiAssetFactorContractError("available_date 不能晚于 observation_date")
    if not pd.notna(factor_value) or not pd.api.types.is_number(factor_value):
        raise MultiAssetFactorContractError("因子值必须是有限数值")
    return {"observation_date": observation_date, "available_date": available_date, "factor_name": factor_name, "value": factor_value}


def save_factor_rows(pool: str, rows: Iterable[Any], *, source: str, data_version: str) -> dict[str, Any]:
    contract = CONTRACTS.get(pool)
    if contract is None:
        raise MultiAssetFactorContractError("不支持的产品池")
    if not source.strip() or not data_version.strip():
        raise MultiAssetFactorContractError("必须声明数据来源和数据版本")
    expected = {name for name, _ in contract["factors"]}
    normalized = [_normalise_row(row, expected) for row in rows]
    if not normalized:
        raise MultiAssetFactorContractError("至少需要一条因子观测")
    keys = [(row["observation_date"], row["factor_name"]) for row in normalized]
    if len(keys) != len(set(keys)):
        raise MultiAssetFactorContractError("同一日期和因子只能有一条观测")
    normalized.sort(key=lambda row: (row["observation_date"], row["factor_name"]))
    ATTRIBUTION_FACTOR_CONTRACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = _paths(pool)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["observation_date", "available_date", "factor_name", "value"])
        writer.writeheader()
        writer.writerows([{**row, "observation_date": row["observation_date"].isoformat(), "available_date": row["available_date"].isoformat()} for row in normalized])
    manifest_path.write_text(json.dumps({
        "contract_version": contract["version"], "source": source.strip(), "data_version": data_version.strip(),
        "row_count": len(normalized), "observation_start": normalized[0]["observation_date"].isoformat(),
        "observation_end": normalized[-1]["observation_date"].isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract_metadata(pool)


def load_factor_rows(pool: str, *, as_of_date: date | None = None) -> list[dict[str, Any]]:
    csv_path, _ = _paths(pool)
    if not csv_path.is_file():
        return []
    expected = {name for name, _ in CONTRACTS[pool]["factors"]}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [_normalise_row(row, expected) for row in csv.DictReader(handle)]
    if as_of_date is not None:
        rows = [row for row in rows if row["observation_date"] <= as_of_date and row["available_date"] <= as_of_date]
    return rows


def load_factor_series(pool: str, factor_name: str, *, as_of_date: date | None = None) -> pd.Series | None:
    rows = [row for row in load_factor_rows(pool, as_of_date=as_of_date) if row["factor_name"] == factor_name]
    if not rows:
        return None
    series = pd.Series({row["observation_date"]: row["value"] for row in rows}, dtype=float).sort_index()
    series.index = pd.to_datetime(series.index)
    series.name = factor_name
    return series
