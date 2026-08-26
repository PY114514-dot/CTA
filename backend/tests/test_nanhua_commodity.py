"""Tests for the local weekly South China Commodity Index factor."""

from datetime import date
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from app.services import factor_library
from app.services.factor_library.factor_regression import run_factor_regression
from app.services.factor_library.nanhua import NanhuaCommodityIndexFactor


def _write_futures_index_database(path: Path, periods: int = 30) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE futures_index_weekly (
                code TEXT NOT NULL,
                week_key TEXT NOT NULL,
                week_end TEXT NOT NULL,
                pc REAL,
                fetched_at TEXT
            )
            """
        )
        rows = []
        for position, timestamp in enumerate(pd.date_range("2025-12-05", periods=periods, freq="W-FRI")):
            iso = timestamp.isocalendar()
            rows.append((
                "142162",
                f"{iso.year}-W{iso.week:02d}",
                timestamp.strftime("%Y-%m-%d"),
                None if position == 0 else 0.001 * ((position % 5) - 2),
                "2026-08-20T08:53:02+08:00",
            ))
        # The source was fetched on Thursday of ISO week 34, so this row is
        # an intraweek snapshot rather than a complete weekly return.
        rows.append(("142162", "2026-W34", "2026-08-19", 0.02, "2026-08-20T08:53:02+08:00"))
        rows.append(("142163", "2026-W33", "2026-08-14", 0.99, "2026-08-20T08:53:02+08:00"))
        connection.executemany(
            "INSERT INTO futures_index_weekly (code, week_key, week_end, pc, fetched_at) VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def test_reads_only_complete_finite_nanhua_weekly_returns(tmp_path: Path) -> None:
    database = tmp_path / "futures.sqlite"
    _write_futures_index_database(database)

    returns = NanhuaCommodityIndexFactor(database).compute({})

    assert returns.name == "nanhua_commodity"
    assert len(returns) == 29  # first pc is null and W34 is still open
    assert returns.index.is_monotonic_increasing
    assert pd.Timestamp("2026-08-19") not in returns.index
    assert np.isfinite(returns.to_numpy()).all()


def test_builds_local_factor_without_daily_market_panels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "futures.sqlite"
    _write_futures_index_database(database)
    factor = NanhuaCommodityIndexFactor(database)
    cache_directory = tmp_path / "factor_cache"

    monkeypatch.setitem(factor_library._FACTORS, factor.meta.name, factor)
    monkeypatch.setattr(factor_library.cache, "FACTORS_DIRECTORY", cache_directory)
    monkeypatch.setattr(factor_library.cache, "MANIFEST_PATH", cache_directory / "_manifest.json")
    monkeypatch.setattr(
        factor_library,
        "fetch_panels",
        lambda *_args, **_kwargs: pytest.fail("local weekly factor must not fetch daily market panels"),
    )

    result = factor_library.build_factors(
        providers=[],
        start=date(2025, 12, 1),
        end=date(2026, 8, 20),
        factor_names=[factor.meta.name],
        use_cache=False,
    )

    assert result["errors"] == {}
    assert result["results"][factor.meta.name]["status"] == "built"
    assert result["results"][factor.meta.name]["rows"] == 29


def test_weekly_regression_uses_nanhua_and_daily_regression_skips_it() -> None:
    rng = np.random.default_rng(20260820)
    weekly_dates = pd.date_range("2024-01-05", periods=80, freq="W-FRI")
    weekly_factor = pd.Series(rng.normal(0.0, 0.015, len(weekly_dates)), index=weekly_dates)
    weekly_product = 1.35 * weekly_factor.to_numpy() + rng.normal(0.0, 0.001, len(weekly_dates))

    weekly_result = run_factor_regression(
        weekly_product,
        [timestamp.date() for timestamp in weekly_dates],
        frequency="weekly",
        factor_names=["nanhua_commodity"],
        factor_series_loader=lambda _name: weekly_factor,
        bootstrap_reps=0,
    )

    assert weekly_result.n_observations == len(weekly_dates)
    assert weekly_result.factors[0].beta == pytest.approx(1.35, abs=0.05)

    daily_dates = pd.bdate_range("2024-01-02", periods=80)
    daily_result = run_factor_regression(
        rng.normal(0.0, 0.01, len(daily_dates)),
        [timestamp.date() for timestamp in daily_dates],
        frequency="daily",
        factor_names=["nanhua_commodity"],
        factor_series_loader=lambda _name: pytest.fail("weekly factor must not be expanded to daily observations"),
        bootstrap_reps=0,
    )

    assert daily_result.factors == []
    assert any("weekly" in warning and "daily" in warning for warning in daily_result.warnings)


def test_weekly_performance_annualizes_with_52_periods(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = pd.Series(
        np.linspace(-0.01, 0.015, 60),
        index=pd.date_range("2024-01-05", periods=60, freq="W-FRI"),
        name="nanhua_commodity",
    )
    monkeypatch.setattr(
        factor_library.cache,
        "load_factor_returns",
        lambda name, *args, **kwargs: returns if name == "nanhua_commodity" else None,
    )
    monkeypatch.setattr(factor_library.cache, "get_cache_info", lambda _name: {})

    performance = next(
        item for item in factor_library.compute_factor_performance() if item["name"] == "nanhua_commodity"
    )

    assert performance["frequency"] == "weekly"
    assert performance["annualized_return"] == round(float(returns.mean() * 52), 4)
