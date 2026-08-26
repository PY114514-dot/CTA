"""Build reviewable weekly Carry inputs from downloaded SHFE contract reports."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


SHFE_CARRY_SYMBOLS = ("rb", "cu", "al", "zn", "au", "ag", "fu", "bu")
_MIN_TERM_GAP_DAYS = 45
_CONTRACT_PATTERN = re.compile(r"^([a-z]+)(\d{4})$", re.IGNORECASE)


def _delivery_month_midpoint(contract: str) -> date | None:
    """Return a declared delivery-month proxy from a standard SHFE contract code.

    The downloaded reports do not include exchange expiry dates.  This proxy is
    intentionally named in the import manifest and must not be described as an
    exchange-published last-trading date.
    """
    match = _CONTRACT_PATTERN.match(contract)
    if match is None:
        return None
    year_month = match.group(2)
    year, month = 2000 + int(year_month[:2]), int(year_month[2:])
    if not 1 <= month <= 12:
        return None
    return date(year, month, 15)


def _read_report(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_index: int | None = None
    columns: dict[str, int] | None = None
    aliases = {
        "contract": {"合约", "contract"},
        "trade_date": {"日期", "date"},
        "settle": {"结算价", "settle"},
        "volume": {"成交量", "volume"},
        "open_interest": {"持仓量", "oi"},
    }
    for index, row in raw.iterrows():
        labels = [str(value).strip().lower() for value in row]
        matched = {
            field: next((position for position, value in enumerate(labels) if value in names), None)
            for field, names in aliases.items()
        }
        if all(position is not None for position in matched.values()):
            header_index, columns = int(index), {field: int(position) for field, position in matched.items()}
            break
    if header_index is None or columns is None:
        raise ValueError(f"{path.name} 未找到合约表头")
    table = pd.DataFrame({field: raw.iloc[header_index + 1:, position] for field, position in columns.items()})
    table["contract"] = table["contract"].ffill().astype(str).str.strip().str.lower()
    table["trade_date"] = pd.to_datetime(table["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    for column in ("settle", "volume", "open_interest"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["trade_date", "settle"])
    table = table[(table["settle"] > 0) & table["contract"].str.match(_CONTRACT_PATTERN)]
    table["symbol"] = table["contract"].str.extract(_CONTRACT_PATTERN, expand=True)[0].str.lower()
    table["delivery_date"] = table["contract"].map(_delivery_month_midpoint)
    return table[["contract", "trade_date", "settle", "volume", "open_interest", "symbol", "delivery_date"]]


def load_shfe_reports(directory: Path) -> pd.DataFrame:
    """Read all downloaded .xls/.xlsx reports without changing the source files."""
    files = sorted(path for path in directory.rglob("*") if path.suffix.lower() in {".xls", ".xlsx"})
    if not files:
        raise ValueError("未找到上期所 .xls 或 .xlsx 行情报表")
    rows = [_read_report(path) for path in files]
    result = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["contract", "trade_date"], keep="last")
    return result.sort_values(["trade_date", "contract"]).reset_index(drop=True)


def build_shfe_carry_rows(directory: Path, *, start: date, end: date) -> list[dict]:
    """Construct weekly named-contract Carry rows from downloaded SHFE reports.

    A signal uses the last reporting day of a week and is realised on the next
    available weekly reporting day.  Zero-open-interest contracts are excluded;
    the nearest usable and the first contract at least 45 days further out are
    used.  The delivery dates are *mid-month proxies* derived from contract
    codes, which is deliberately recorded by the caller's manifest.
    """
    if start >= end:
        raise ValueError("结束日期必须晚于开始日期")
    data = load_shfe_reports(directory)
    data = data[(data["trade_date"].dt.date >= start) & (data["trade_date"].dt.date <= end)]
    data = data[data["symbol"].isin(SHFE_CARRY_SYMBOLS)]
    data = data[data["open_interest"].fillna(0) > 0]
    if data.empty:
        raise ValueError("上期所报表中没有可用的核心商品合约行情")

    data["week"] = data["trade_date"].dt.to_period("W-SUN")
    snapshots = {
        week: frame[frame["trade_date"] == frame["trade_date"].max()]
        for week, frame in data.groupby("week", sort=True)
    }
    weeks = sorted(snapshots)
    rows: list[dict] = []
    for signal_week, realised_week in zip(weeks, weeks[1:], strict=False):
        signal = snapshots[signal_week]
        realised = snapshots[realised_week]
        signal_day = signal["trade_date"].iloc[0].date()
        realised_day = realised["trade_date"].iloc[0].date()
        for symbol in SHFE_CARRY_SYMBOLS:
            candidates = signal[(signal["symbol"] == symbol) & (signal["delivery_date"] > realised_day)].sort_values("delivery_date")
            if len(candidates) < 2:
                continue
            near = candidates.iloc[0]
            far_options = candidates[candidates["delivery_date"] >= near["delivery_date"] + timedelta(days=_MIN_TERM_GAP_DAYS)]
            if far_options.empty:
                continue
            far = far_options.iloc[0]
            realised_prices = realised.set_index("contract")["settle"]
            if near["contract"] not in realised_prices or far["contract"] not in realised_prices:
                continue
            rows.append({
                "observation_date": realised_day,
                "symbol": symbol,
                "near_contract": str(near["contract"]),
                "far_contract": str(far["contract"]),
                "near_price": float(near["settle"]),
                "far_price": float(far["settle"]),
                "near_next_price": float(realised_prices[near["contract"]]),
                "far_next_price": float(realised_prices[far["contract"]]),
                "near_days_to_expiry": float((near["delivery_date"] - realised_day).days),
                "far_days_to_expiry": float((far["delivery_date"] - realised_day).days),
                "holding_return": float(realised_prices[near["contract"]] / near["settle"] - 1.0),
                "available_date": realised_day,
            })
    return rows
