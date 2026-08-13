"""Analysis pipeline sub-package: strategy classification, factor analysis,
variety identification, and report generation."""

from app.services.analysis.strategy_classifier import (
    classify_strategy,
    StrategyClassification,
)
from app.services.analysis.factor_analyzer import (
    analyze_factors,
    FactorAnalysisResult,
    FactorExposureDetail,
)
from app.services.analysis.variety_identifier import (
    identify_varieties,
    VarietyIdentificationResult,
    VarietyCandidate,
    SectorExposure,
)
from app.services.analysis.equity_cta_identifier import identify_equity_references
from app.services.analysis.report_generator import (
    generate_report,
    AnalysisReport,
    ReportConfig,
)
from app.services.analysis.deep_attribution import (
    deep_attribution,
    DeepAttributionResult,
    as_dict as deep_attribution_as_dict,
)

__all__ = [
    "classify_strategy",
    "StrategyClassification",
    "analyze_factors",
    "FactorAnalysisResult",
    "FactorExposureDetail",
    "identify_varieties",
    "VarietyIdentificationResult",
    "VarietyCandidate",
    "SectorExposure",
    "identify_equity_references",
    "generate_report",
    "AnalysisReport",
    "ReportConfig",
    "deep_attribution",
    "DeepAttributionResult",
    "deep_attribution_as_dict",
]
