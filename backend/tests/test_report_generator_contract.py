from app.services.analysis.factor_analyzer import FactorAnalysisResult, FactorExposureDetail
from app.services.analysis.report_generator import _build_charts_data, _build_structured
from app.services.analysis.variety_identifier import (
    SectorExposure,
    VarietyCandidate,
    VarietyIdentificationResult,
)


def test_report_structured_output_matches_frontend_analysis_contract() -> None:
    factors = FactorAnalysisResult(
        factors=[FactorExposureDetail(
            factor_name="momentum", factor_label="动量", exposure_beta=0.2,
            correlation=0.3, t_statistic=1.5, p_value=0.1,
            confidence_pct=60, confidence_label="中",
        )],
        r_squared=0.2, adj_r_squared=0.1,
    )
    varieties = VarietyIdentificationResult(
        top_varieties=[VarietyCandidate("RB", "螺纹钢", "黑色", 60, 0.4, 0.2, 0.5)],
        sector_exposure=[SectorExposure("黑色", "#000", 60, "螺纹钢", 1)],
        method_stability=0.5,
    )

    structured = _build_structured(None, factors, varieties)

    factor = structured["factors"]["factors"][0]
    assert factor["factor_label"] == "动量"
    assert factor["exposure_beta"] == 0.2
    assert factor["t_statistic"] == 1.5
    assert structured["varieties"]["method_stability"] == 0.5
    assert structured["varieties"]["sector_exposure"][0]["variety_count"] == 1


def test_report_chart_output_matches_frontend_chart_contract() -> None:
    factors = FactorAnalysisResult(
        factors=[FactorExposureDetail(
            factor_name="momentum", factor_label="动量", exposure_beta=0.2,
            correlation=0.3, t_statistic=1.5, p_value=0.1,
            confidence_pct=60, confidence_label="中",
        )],
        r_squared=0.2, adj_r_squared=0.1,
    )
    varieties = VarietyIdentificationResult(
        top_varieties=[VarietyCandidate("RB", "螺纹钢", "黑色", 60, 0.4, 0.2, 0.5)],
        sector_exposure=[SectorExposure("黑色", "#000", 60, "螺纹钢", 1)],
        method_stability=0.5,
    )

    charts = _build_charts_data(None, factors, varieties)

    assert charts["factor_radar"] == {"indicators": ["动量"], "values": [0.3]}
    assert charts["variety_bar"] == {"names": ["螺纹钢"], "probabilities": [60]}
    assert charts["sector_pie"] == [{"name": "黑色", "color": "#000", "value": 60}]
