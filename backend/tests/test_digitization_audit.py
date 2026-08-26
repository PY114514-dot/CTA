"""数字化抽样误差审计工具的测试（批次 13 精度）。

覆盖三档判定（pass/review/fail）、样本不足、覆盖缺口统计与端点契约。
判定阈值见 app/services/digitization_audit.py 顶部的预注册注释。
"""

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import NetAssetValuePoint
from app.services.digitization_audit import audit_digitized_nav


def _series(start: date, n: int, step: float = 0.001) -> list[NetAssetValuePoint]:
    points = []
    nav = 1.0
    for index in range(n):
        nav *= 1.0 + step
        points.append(NetAssetValuePoint(
            observation_date=start + timedelta(days=index * 7),
            net_asset_value=round(nav, 6),
        ))
    return points


def _shifted(points: list[NetAssetValuePoint], noise: float) -> list[NetAssetValuePoint]:
    return [
        NetAssetValuePoint(
            observation_date=point.observation_date,
            net_asset_value=round(point.net_asset_value * (1.0 + noise), 6),
        )
        for point in points
    ]


def test_perfect_digitization_passes() -> None:
    reference = _series(date(2024, 1, 5), 12)
    report = audit_digitized_nav(reference, reference)
    assert report["verdict"] == "pass"
    assert report["errors"]["mae"] == 0.0
    assert report["errors"]["max_relative_error"] == 0.0
    assert report["errors"]["within_tolerance_ratio"] == 1.0


def test_small_noise_stays_within_pass_band() -> None:
    reference = _series(date(2024, 1, 5), 12)
    digitized = _shifted(reference, noise=0.0005)  # 0.05% 均匀偏差
    report = audit_digitized_nav(digitized, reference)
    assert report["verdict"] == "pass"
    assert report["sample_coverage_ratio"] == 1.0
    assert report["errors"]["within_tolerance_count"] == 12


def test_moderate_error_triggers_review() -> None:
    reference = _series(date(2024, 1, 5), 12)
    digitized = _shifted(reference, noise=0.003)  # 0.3% 均匀偏差
    report = audit_digitized_nav(digitized, reference)
    assert report["verdict"] == "review"


def test_large_error_fails() -> None:
    reference = _series(date(2024, 1, 5), 12)
    digitized = _shifted(reference, noise=0.02)  # 2% 均匀偏差
    report = audit_digitized_nav(digitized, reference)
    assert report["verdict"] == "fail"


def test_insufficient_matched_sample_fails_with_warning() -> None:
    reference = _series(date(2024, 1, 5), 4)
    report = audit_digitized_nav(reference, reference, min_matched=5)
    assert report["verdict"] == "fail"
    assert report["matched_count"] == 4
    assert any("低于最低要求" in warning for warning in report["warnings"])


def test_unmatched_and_uncovered_counts_reported() -> None:
    reference = _series(date(2024, 1, 5), 10)
    # 数字化序列多一个无关日期、少最后两个基准日
    digitized = reference[:-2] + [
        NetAssetValuePoint(observation_date=date(2025, 6, 6), net_asset_value=1.5)
    ]
    report = audit_digitized_nav(digitized, reference)
    assert report["matched_count"] == 8
    assert report["unmatched_digitized_count"] == 1
    assert report["uncovered_reference_count"] == 2
    assert abs(report["sample_coverage_ratio"] - 0.8) < 1e-9
    assert any("覆盖率" in warning for warning in report["warnings"])


def test_digitize_audit_endpoint_contract() -> None:
    payload = {
        "digitized_points": [
            {"observation_date": "2024-01-05", "net_asset_value": 1.001},
            {"observation_date": "2024-01-12", "net_asset_value": 1.002},
            {"observation_date": "2024-01-19", "net_asset_value": 1.004},
            {"observation_date": "2024-01-26", "net_asset_value": 1.006},
            {"observation_date": "2024-02-02", "net_asset_value": 1.008},
            {"observation_date": "2024-02-09", "net_asset_value": 1.010},
        ],
        "reference_points": [
            {"observation_date": "2024-01-05", "net_asset_value": 1.000},
            {"observation_date": "2024-01-12", "net_asset_value": 1.002},
            {"observation_date": "2024-01-19", "net_asset_value": 1.004},
            {"observation_date": "2024-01-26", "net_asset_value": 1.006},
            {"observation_date": "2024-02-02", "net_asset_value": 1.008},
            {"observation_date": "2024-02-09", "net_asset_value": 1.010},
        ],
    }
    with TestClient(app) as client:
        response = client.post("/api/nav/digitize-audit", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "digitization-sampling-error-audit-v1"
    assert body["matched_count"] == 6
    assert body["verdict"] in {"pass", "review", "fail"}
    assert set(body["errors"]) == {
        "mae", "rmse", "max_abs_error", "mean_relative_error",
        "median_relative_error", "max_relative_error",
        "within_tolerance_count", "within_tolerance_ratio",
    }
    assert body["worst_points"][0]["observation_date"] == "2024-01-05"
