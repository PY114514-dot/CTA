"""Shared application state and dependency helpers.

Module-level provider instances persist across requests in the single-process
dev server.  Routers import these via ``from app.dependencies import ...``.
"""

from app.services.market_data import AKShareProvider, CsvMarketDataProvider
from app.services.analysis import ReportConfig

# ---------------------------------------------------------------------------
# Singleton providers (state persists across requests in dev mode)
# ---------------------------------------------------------------------------

akshare_provider = AKShareProvider()
csv_provider = CsvMarketDataProvider()
report_config = ReportConfig.from_env()


def set_report_config(config: ReportConfig) -> None:
    """Replace the module-level report config (called by PUT /api/analysis/config)."""
    global report_config
    report_config = config


def get_report_config() -> ReportConfig:
    """Return the current runtime report config, not an import-time snapshot."""
    return report_config
