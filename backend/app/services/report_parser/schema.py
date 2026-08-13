"""Data schema for weekly report profile snapshots.

Each weekly report produces one ProfileSnapshot per product, capturing the
multi-dimensional portrait described in the planning document (L2-L4 layers).
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class PerformanceMetrics:
    """L0/L1: Disclosed performance metrics from the report."""

    cumulative_nav: float | None = None
    annualized_return: float | None = None
    annualized_vol: float | None = None
    max_drawdown: float | None = None
    sharpe: float | None = None
    calmar: float | None = None
    # Backtest vs live comparison (if disclosed)
    backtest_max_drawdown: float | None = None
    is_live: bool = True


@dataclass
class ExposureStructure:
    """L2: Long/short exposure and leverage."""

    long_exposure: float | None = None      # e.g., 82.57%
    short_exposure: float | None = None     # e.g., -46.96%
    net_exposure: float | None = None       # e.g., 35.62%
    leverage: float | None = None           # e.g., 1.295


@dataclass
class FactorContribution:
    """L3: Per-factor return contribution for the reporting period."""

    profit: float | None = None             # 利润
    inventory: float | None = None          # 库存
    basis: float | None = None              # 基差
    long_term_rule: float | None = None     # 长期规则
    volume_price_corr: float | None = None  # 量价相关性
    long_term_cross: float | None = None    # 长期截面


@dataclass
class SectorPnl:
    """L3: Sector-level P&L breakdown."""

    sector: str
    pnl_pct: float
    direction: str = ""  # "多" | "空" | ""


@dataclass
class VarietyFactorRow:
    """L4: One row in the variety x factor matrix."""

    variety: str
    profit: float | None = None
    inventory: float | None = None
    basis: float | None = None
    long_term_rule: float | None = None
    volume_price_corr: float | None = None
    long_term_cross: float | None = None
    total_weight: float | None = None
    direction: str = ""  # "多" | "空"


@dataclass
class CorrelationMatrix:
    """L2: Correlation with benchmarks at multiple horizons."""

    # Keys: benchmark name, Values: {horizon: corr_value}
    # Horizons: "1m", "3m", "6m", "1y", "ytd"
    data: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ProfileSnapshot:
    """Complete multi-layer portrait of a product from one weekly report."""

    product_name: str
    report_date: date
    # L0/L1
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    # L2
    exposure: ExposureStructure = field(default_factory=ExposureStructure)
    correlations: CorrelationMatrix = field(default_factory=CorrelationMatrix)
    # L3
    factor_contribution: FactorContribution = field(default_factory=FactorContribution)
    sector_pnl: list[SectorPnl] = field(default_factory=list)
    # L4
    variety_factor_matrix: list[VarietyFactorRow] = field(default_factory=list)
    # Metadata
    source_file: str = ""
    parse_confidence: float = 0.0  # 0-1, how confident we are in extraction
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        import dataclasses

        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            if isinstance(obj, date):
                return obj.isoformat()
            return obj

        return _convert(self)
