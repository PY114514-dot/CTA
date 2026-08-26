"""Golden sample regression tests for FOF Agent.

These tests ensure core functionality is not broken by changes.
Each golden sample defines input and expected output for a specific scenario.
"""

from pathlib import Path

import yaml

from app.schemas import (
    DataFrequency,
    FofFundInput,
    FofRecommendationRequest,
    FofRiskProfile,
    NetAssetValuePoint,
)
from app.services.fof_agent import FofRecommendationAgent


def _load_golden_samples() -> list[dict]:
    """Load golden samples from YAML file."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "fof_agent"
    with open(fixtures_dir / "golden_samples.yaml", encoding="utf-8") as f:
        content = f.read()

    # Split by '---' separator
    samples = []
    current = {}
    for line in content.splitlines():
        if line == "---":
            if current:
                samples.append(current)
                current = {}
        else:
            if ":" in line:
                key, value = line.split(":", 1)
                current[key.strip()] = value.strip()

    if current:
        samples.append(current)

    return samples


def _parse_fund_from_dict(data: dict) -> FofFundInput:
    """Parse FOF fund input from dictionary."""
    nav_points = [
        NetAssetValuePoint(
            observation_date=point["observation_date"],
            net_asset_value=point["net_asset_value"],
        )
        for point in data.get("nav_points", [])
    ]

    return FofFundInput(
        fund_id=data["fund_id"],
        fund_name=data["fund_name"],
        frequency=DataFrequency(data["frequency"]),
        nav_points=nav_points,
    )


def test_golden_sample_balanced_risk(tmp_path: Path) -> None:
    """Test golden sample: balanced risk profile with 3 funds."""
    request = FofRecommendationRequest(
        funds=[
            _parse_fund_from_dict({
                "fund_id": "fund_a",
                "fund_name": "稳盈一号",
                "frequency": "monthly",
                "nav_points": [
                    {"observation_date": "2024-01-01", "net_asset_value": 1.0000},
                    {"observation_date": "2024-02-01", "net_asset_value": 1.0100},
                    {"observation_date": "2024-03-01", "net_asset_value": 1.0250},
                    {"observation_date": "2024-04-01", "net_asset_value": 1.0300},
                    {"observation_date": "2024-05-01", "net_asset_value": 1.0400},
                    {"observation_date": "2024-06-01", "net_asset_value": 1.0550},
                    {"observation_date": "2024-07-01", "net_asset_value": 1.0600},
                    {"observation_date": "2024-08-01", "net_asset_value": 1.0750},
                    {"observation_date": "2024-09-01", "net_asset_value": 1.0800},
                    {"observation_date": "2024-10-01", "net_asset_value": 1.0900},
                    {"observation_date": "2024-11-01", "net_asset_value": 1.1000},
                    {"observation_date": "2024-12-01", "net_asset_value": 1.1150},
                ],
            }),
            _parse_fund_from_dict({
                "fund_id": "fund_b",
                "fund_name": "成长二号",
                "frequency": "monthly",
                "nav_points": [
                    {"observation_date": "2024-01-01", "net_asset_value": 1.0000},
                    {"observation_date": "2024-02-01", "net_asset_value": 1.0200},
                    {"observation_date": "2024-03-01", "net_asset_value": 1.0500},
                    {"observation_date": "2024-04-01", "net_asset_value": 1.0300},
                    {"observation_date": "2024-05-01", "net_asset_value": 1.0700},
                    {"observation_date": "2024-06-01", "net_asset_value": 1.1000},
                    {"observation_date": "2024-07-01", "net_asset_value": 1.0800},
                    {"observation_date": "2024-08-01", "net_asset_value": 1.1200},
                    {"observation_date": "2024-09-01", "net_asset_value": 1.1500},
                    {"observation_date": "2024-10-01", "net_asset_value": 1.1800},
                    {"observation_date": "2024-11-01", "net_asset_value": 1.2000},
                    {"observation_date": "2024-12-01", "net_asset_value": 1.2500},
                ],
            }),
            _parse_fund_from_dict({
                "fund_id": "fund_c",
                "fund_name": "保守三号",
                "frequency": "monthly",
                "nav_points": [
                    {"observation_date": "2024-01-01", "net_asset_value": 1.0000},
                    {"observation_date": "2024-02-01", "net_asset_value": 1.0050},
                    {"observation_date": "2024-03-01", "net_asset_value": 1.0080},
                    {"observation_date": "2024-04-01", "net_asset_value": 1.0120},
                    {"observation_date": "2024-05-01", "net_asset_value": 1.0150},
                    {"observation_date": "2024-06-01", "net_asset_value": 1.0180},
                    {"observation_date": "2024-07-01", "net_asset_value": 1.0200},
                    {"observation_date": "2024-08-01", "net_asset_value": 1.0220},
                ],
            }),
        ],
        risk_profile=FofRiskProfile.BALANCED,
        max_single_fund_weight=0.6,
    )

    result = FofRecommendationAgent(tmp_path).recommend(request)

    # Assertions for golden sample expectations
    assert result.reflection.passed, f"Expected reflection to pass, got warnings: {result.reflection.warnings}"

    # Should have at least 2 recommendations
    assert len(result.recommendations) >= 2, f"Expected at least 2 recommendations, got {len(result.recommendations)}"

    # Weights should sum to ~1.0
    total_weight = sum(item.weight for item in result.recommendations)
    assert abs(total_weight - 1.0) < 0.01, f"Expected weights sum to ~1.0, got {total_weight}"

    # Each weight should be <= 0.6
    for item in result.recommendations:
        assert item.weight <= 0.6, f"Expected weight <= 0.6, got {item.weight}"

    # Should have tool trace
    assert len(result.tool_trace) > 0, "Expected tool trace to be non-empty"


def test_golden_sample_insufficient_data(tmp_path: Path) -> None:
    """Test golden sample: insufficient data triggers reflection failure."""
    request = FofRecommendationRequest(
        funds=[
            _parse_fund_from_dict({
                "fund_id": "insufficient_fund",
                "fund_name": "样本不足基金",
                "frequency": "monthly",
                "nav_points": [
                    {"observation_date": "2024-01-01", "net_asset_value": 1.0000},
                    {"observation_date": "2024-02-01", "net_asset_value": 1.0100},
                    {"observation_date": "2024-03-01", "net_asset_value": 1.0200},
                ],
            }),
            _parse_fund_from_dict({
                "fund_id": "insufficient_fund2",
                "fund_name": "另一个样本不足基金",
                "frequency": "monthly",
                "nav_points": [
                    {"observation_date": "2024-01-01", "net_asset_value": 1.0000},
                    {"observation_date": "2024-02-01", "net_asset_value": 1.0050},
                ],
            }),
        ],
        risk_profile=FofRiskProfile.CONSERVATIVE,
    )

    result = FofRecommendationAgent(tmp_path).recommend(request)

    # Assertions for golden sample expectations
    assert not result.reflection.passed, "Expected reflection to fail for insufficient data"

    # Should have at least one warning about insufficient data
    assert any("少于 2 个" in w or "样本不足" in w for w in result.reflection.warnings), \
        f"Expected warning about insufficient data, got: {result.reflection.warnings}"

    # Should have replanned
    assert result.plan_revisions >= 1, f"Expected at least 1 plan revision, got {result.plan_revisions}"
