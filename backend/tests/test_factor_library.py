"""Unit tests for the CTA factor library.

Creates a fixed 5-variety synthetic panel with 500 business days of OHLCV data
and verifies that all computable factors produce valid return series (length > 100,
no NaN in the output).

The two fundamental factors (warehouse_receipt, inventory) require external CSV
input and are tested separately for graceful skip behavior.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.factor_library import _FACTORS, get_all_factors, get_factor, get_factor_detail
from app.services.factor_library.base import (
    FactorBase,
    build_close_panel,
    build_volume_panel,
    cross_sectional_zscore,
    equal_weight_long_short,
    long_short_portfolio,
)
from app.services.factor_library.trend import TrendFactor
from app.services.factor_library.volume_price import VolumePriceCorrFactor
from app.services.factor_library.cross_section import CrossSectionMomentumFactor
from app.services.factor_library.basis import BasisCarryFactor
from app.services.factor_library.profit import ProfitMarginFactor
from app.services.factor_library.short_trend import ShortTermTrendFactor
from app.services.factor_library.skewness import SkewnessFactor
from app.services.factor_library.fundamental import WarehouseReceiptFactor, InventoryFactor


# ---------------------------------------------------------------------------
# Fixtures: synthetic 5-variety panel
# ---------------------------------------------------------------------------

_SYMBOLS = ["rb", "cu", "au", "m", "pp"]
_N_DAYS = 500


def _generate_panel(
    symbols: list[str] = _SYMBOLS,
    n_days: int = _N_DAYS,
    seed: int = 2024,
) -> dict[str, pd.DataFrame]:
    """Generate a synthetic OHLCV panel for testing.

    Each variety gets a random-walk price series with realistic daily vol
    (~1-2%) and correlated volume.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2022-01-04", periods=n_days)

    panels: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols):
        # Random walk with slight upward drift
        daily_vol = 0.01 + 0.005 * (i / len(symbols))  # vary vol across varieties
        log_returns = rng.normal(0.0002, daily_vol, n_days)
        close = 100 * np.exp(np.cumsum(log_returns))

        # Derive OHLV from close
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        volume = rng.integers(10000, 500000, n_days).astype(float)

        df = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        panels[symbol] = df

    return panels


@pytest.fixture(scope="module")
def panel() -> dict[str, pd.DataFrame]:
    """Module-scoped fixture: generate the panel once for all tests."""
    return _generate_panel()


# ---------------------------------------------------------------------------
# Tests: individual factor computation
# ---------------------------------------------------------------------------

# Price-volume factors that compute purely from a generic OHLCV panel.
# ProfitMarginFactor is tested separately because it requires specific
# multi-variety recipes (e.g., steel margin = rb - 1.6*i - 0.5*j).
_COMPUTABLE_FACTORS = [
    TrendFactor,
    VolumePriceCorrFactor,
    CrossSectionMomentumFactor,
    BasisCarryFactor,
    ShortTermTrendFactor,
    SkewnessFactor,
]


class TestFactorComputation:
    """Run each computable factor on the synthetic panel and validate output."""

    @pytest.mark.parametrize(
        "factor_cls",
        _COMPUTABLE_FACTORS,
        ids=[cls.meta.name if hasattr(cls, "meta") else cls.__name__ for cls in _COMPUTABLE_FACTORS],
    )
    def test_factor_returns_valid_series(self, factor_cls: type, panel: dict[str, pd.DataFrame]) -> None:
        """Factor output must be a non-empty Series with no NaN and length > 100."""
        factor = factor_cls()
        result = factor.compute(panel)

        assert isinstance(result, pd.Series), f"{factor.meta.name} did not return a Series"
        assert len(result) > 100, (
            f"{factor.meta.name}: expected > 100 rows, got {len(result)}"
        )
        assert not result.isna().any(), f"{factor.meta.name} contains NaN values"
        assert result.name == factor.meta.name

    @pytest.mark.parametrize(
        "factor_cls",
        _COMPUTABLE_FACTORS,
        ids=[cls.meta.name if hasattr(cls, "meta") else cls.__name__ for cls in _COMPUTABLE_FACTORS],
    )
    def test_factor_returns_are_finite(self, factor_cls: type, panel: dict[str, pd.DataFrame]) -> None:
        """All factor returns must be finite (no inf)."""
        factor = factor_cls()
        result = factor.compute(panel)

        if len(result) > 0:
            assert np.isfinite(result.values).all(), f"{factor.meta.name} contains inf values"

    @pytest.mark.parametrize(
        "factor_cls",
        _COMPUTABLE_FACTORS,
        ids=[cls.meta.name if hasattr(cls, "meta") else cls.__name__ for cls in _COMPUTABLE_FACTORS],
    )
    def test_factor_nav_starts_at_one(self, factor_cls: type, panel: dict[str, pd.DataFrame]) -> None:
        """Cumulative NAV derived from factor returns should start near 1.0."""
        factor = factor_cls()
        nav = factor.compute_nav(panel)

        if len(nav) > 0:
            # First NAV value = 1 + first return, should be close to 1
            assert 0.8 < nav.iloc[0] < 1.2, f"{factor.meta.name} NAV starts at {nav.iloc[0]}"


# ---------------------------------------------------------------------------
# Tests: profit margin factor (needs specific multi-variety recipes)
# ---------------------------------------------------------------------------

# Symbols that satisfy at least 4 margin recipes (equal_weight_long_short
# requires >= 4 valid signals per row):
#   rb recipe: rb, i, j
#   pp recipe: pp, sc
#   p  recipe: p, m
#   ta recipe: ta, sc
#   ma recipe: ma, j
_MARGIN_SYMBOLS = ["rb", "i", "j", "pp", "sc", "p", "m", "ta", "ma"]


class TestProfitMarginFactor:
    """ProfitMarginFactor requires specific ingredient symbols for its recipes."""

    @pytest.fixture(scope="class")
    def margin_panel(self) -> dict[str, pd.DataFrame]:
        return _generate_panel(symbols=_MARGIN_SYMBOLS, n_days=_N_DAYS, seed=777)

    def test_produces_valid_series(self, margin_panel: dict[str, pd.DataFrame]) -> None:
        """With recipe-compatible symbols, profit_margin outputs > 100 rows, no NaN."""
        factor = ProfitMarginFactor()
        result = factor.compute(margin_panel)

        assert isinstance(result, pd.Series)
        assert len(result) > 100, f"profit_margin: expected > 100 rows, got {len(result)}"
        assert not result.isna().any(), "profit_margin contains NaN values"
        assert result.name == "profit_margin"

    def test_returns_empty_with_incompatible_panel(self, panel: dict[str, pd.DataFrame]) -> None:
        """With symbols that don't match any recipe, returns empty gracefully."""
        factor = ProfitMarginFactor()
        result = factor.compute(panel)  # panel has rb, cu, au, m, pp — missing i, j, sc, p

        assert isinstance(result, pd.Series)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: fundamental factors graceful skip
# ---------------------------------------------------------------------------


class TestFundamentalFactors:
    def test_warehouse_receipt_input_note(self) -> None:
        """Without CSV input, warehouse_receipt factor reports input_note."""
        factor = WarehouseReceiptFactor()
        note = factor.input_note()
        # Either None (file exists) or a string explaining what's needed
        if note is not None:
            assert "warehouse_receipts.csv" in note

    def test_inventory_input_note(self) -> None:
        """Without CSV input, inventory factor reports input_note."""
        factor = InventoryFactor()
        note = factor.input_note()
        if note is not None:
            assert "inventory" in note.lower() or "库存" in note

    def test_fundamental_compute_empty_without_input(self, panel: dict[str, pd.DataFrame]) -> None:
        """Fundamental factors return empty series when input CSV is missing."""
        for cls in (WarehouseReceiptFactor, InventoryFactor):
            factor = cls()
            if factor.input_note() is not None:
                # Input file missing → compute should return empty
                result = factor.compute(panel)
                assert isinstance(result, pd.Series)
                assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: registry and metadata
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_factors_registered(self) -> None:
        """The registry should expose the configured factor catalog."""
        all_factors = get_all_factors()
        assert len(all_factors) == len(_FACTORS)
        assert {"trend", "mean_reversion_5d"}.issubset({f["name"] for f in all_factors})

    def test_factor_names_unique(self) -> None:
        names = [f["name"] for f in get_all_factors()]
        assert len(names) == len(set(names))

    def test_get_factor_known(self) -> None:
        """get_factor returns a FactorBase instance for known names."""
        factor = get_factor("trend")
        assert factor is not None
        assert isinstance(factor, FactorBase)
        assert factor.meta.name == "trend"

    def test_get_factor_unknown(self) -> None:
        assert get_factor("nonexistent_factor") is None

    def test_get_factor_detail_includes_code(self) -> None:
        """Factor detail should include the compute() source code."""
        detail = get_factor_detail("trend")
        assert detail is not None
        assert "def compute" in detail["code"]
        assert detail["formula"] != ""

    def test_short_trend_formula_has_separated_latex_commands(self) -> None:
        detail = get_factor_detail("short_term_trend_20")
        assert detail is not None
        assert r"\quad w" in detail["formula"]
        assert r"\quadw" not in detail["formula"]

    def test_get_factor_detail_unknown(self) -> None:
        assert get_factor_detail("nonexistent") is None

    def test_all_factors_have_required_metadata(self) -> None:
        """Every factor must have name, display_name, category, description."""
        for f in get_all_factors():
            assert f["name"], "Factor missing name"
            assert f["display_name"], f"Factor {f['name']} missing display_name"
            assert f["category"] in ("量价", "基本面", "市场"), f"Factor {f['name']} has invalid category"
            assert f["description"], f"Factor {f['name']} missing description"


# ---------------------------------------------------------------------------
# Tests: base utilities
# ---------------------------------------------------------------------------


class TestBaseUtilities:
    def test_build_close_panel(self, panel: dict[str, pd.DataFrame]) -> None:
        close = build_close_panel(panel)
        assert not close.empty
        assert set(close.columns) == set(_SYMBOLS)
        assert len(close) == _N_DAYS

    def test_build_volume_panel(self, panel: dict[str, pd.DataFrame]) -> None:
        vol = build_volume_panel(panel)
        assert not vol.empty
        assert set(vol.columns) == set(_SYMBOLS)

    def test_cross_sectional_zscore_mean_zero(self, panel: dict[str, pd.DataFrame]) -> None:
        """Z-scored signals should have row-mean ≈ 0."""
        close = build_close_panel(panel)
        signals = close.pct_change().dropna()
        z = cross_sectional_zscore(signals)
        row_means = z.mean(axis=1).dropna()
        assert (row_means.abs() < 1e-10).all()

    def test_long_short_portfolio_no_lookahead(self) -> None:
        """Portfolio return at t uses weights from t-1 (shift check)."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        cols = ["a", "b", "c", "d", "e"]
        weights = pd.DataFrame(
            np.ones((10, 5)) * 0.1, index=dates, columns=cols
        )
        returns = pd.DataFrame(
            np.ones((10, 5)) * 0.01, index=dates, columns=cols
        )
        port = long_short_portfolio(weights, returns)
        # First day has no prior weights → should be excluded (NaN → dropped)
        assert len(port) < 10
