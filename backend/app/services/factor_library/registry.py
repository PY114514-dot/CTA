"""Core variety subset for the factor library (P1 phase).

Selects 18 highly liquid futures varieties spanning all major sectors.
This keeps data-fetching manageable while ensuring cross-sectional factors
have enough breadth to be meaningful.

The full 28-variety registry in market_data/index_registry.py remains the
authoritative source for the analysis pipeline; this module is a focused
subset for factor construction.
"""

from app.services.market_data.index_registry import FUTURES_VARIETIES, VarietyInfo

# Core subset: symbol -> reason for inclusion
# Selection criteria: high average daily turnover, continuous main contract
# available since ~2019, sector diversification.
CORE_VARIETY_SYMBOLS: list[str] = [
    # 黑色 (3)
    "rb",   # 螺纹钢 - 最活跃商品期货
    "i",    # 铁矿石 - 黑色链核心
    "j",    # 焦炭 - 黑色链上游
    # 有色 (3)
    "cu",   # 沪铜 - 有色龙头
    "al",   # 沪铝 - 产量最大有色金属
    "zn",   # 沪锌 - 流动性好
    # 贵金属 (2)
    "au",   # 沪金 - 避险资产
    "ag",   # 沪银 - 贵金属+工业双属性
    # 能化 (5)
    "sc",   # 原油 - 能化之锚
    "ta",   # PTA - 化工龙头
    "ma",   # 甲醇 - 煤化工核心
    "eg",   # 乙二醇 - 聚酯链
    "pp",   # 聚丙烯 - 通用塑料
    # 农产品 (5)
    "m",    # 豆粕 - 农产品最活跃
    "p",    # 棕榈油 - 油脂龙头
    "cf",   # 棉花 - 软商品
    "sr",   # 白糖 - 全球定价
    "oi",   # 菜油 - 油脂补充
]


def get_core_varieties() -> dict[str, VarietyInfo]:
    """Return VarietyInfo for each core symbol."""
    return {s: FUTURES_VARIETIES[s] for s in CORE_VARIETY_SYMBOLS if s in FUTURES_VARIETIES}


def get_core_symbols() -> list[str]:
    """Return the ordered list of core variety symbols."""
    return list(CORE_VARIETY_SYMBOLS)


def get_sector_coverage() -> dict[str, list[str]]:
    """Return {sector: [symbols]} for the core subset."""
    coverage: dict[str, list[str]] = {}
    for sym in CORE_VARIETY_SYMBOLS:
        info = FUTURES_VARIETIES.get(sym)
        if info:
            coverage.setdefault(info.sector, []).append(sym)
    return coverage
