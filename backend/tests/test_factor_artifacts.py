from datetime import date

import pandas as pd

from app.services.factor_library import cache
from app.services.factor_library.base import turnover_and_costs


def test_factor_artifacts_are_append_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "FACTORS_DIRECTORY", tmp_path)
    monkeypatch.setattr(cache, "MANIFEST_PATH", tmp_path / "_manifest.json")
    series = pd.Series([0.01, -0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    first = cache.save_factor_artifact("trend", "baseline_return", series, date(2024, 1, 1), date(2024, 1, 3))
    second = cache.save_factor_artifact("trend", "risk_overlay_return", series * 0.5, date(2024, 1, 1), date(2024, 1, 3), metadata={"profile": "vol_target"})
    manifest = cache.list_cached_factors()
    assert len(manifest["trend"]["artifacts"]) == 2
    assert first["file"] != second["file"]


def test_turnover_costs_follow_weight_changes() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    weights = pd.DataFrame({"a": [0.0, 0.5, -0.5], "b": [0.0, -0.5, 0.5]}, index=index)
    gross = pd.Series([0.01, 0.0, 0.01], index=index)
    turnover, costs = turnover_and_costs(weights, gross, cost_bps=10)
    assert turnover.iloc[1] == 1.0
    assert costs.iloc[1] == 0.001


def test_active_return_versions_do_not_overwrite_parameter_variants(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "FACTORS_DIRECTORY", tmp_path)
    monkeypatch.setattr(cache, "MANIFEST_PATH", tmp_path / "_manifest.json")
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    first = pd.Series([0.01, -0.02], index=index)
    second = pd.Series([0.02, -0.01], index=index)
    first_path = cache.save_factor_returns("trend", first, date(2024, 1, 1), date(2024, 1, 3), params={"window": 20})
    second_path = cache.save_factor_returns("trend", second, date(2024, 1, 1), date(2024, 1, 3), params={"window": 60})
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()
    entry = cache.get_cache_info("trend")
    assert len(entry["return_versions"]) == 2
    expected = second.rename("trend")
    expected.index.name = "date"
    pd.testing.assert_series_equal(cache.load_factor_returns("trend"), expected)
