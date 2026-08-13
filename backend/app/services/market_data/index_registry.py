"""Registry of benchmark indices and futures varieties for CTA analysis.

Each entry carries a stable symbol key, display name, exchange, sector, and
asset class.  The registry is the single source of truth for what the analysis
pipeline can fetch and attribute against.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndexInfo:
    """A market index used for strategy-type classification."""

    symbol: str
    name: str
    code: str  # provider-specific query code
    asset_class: str  # "equity" | "commodity" | "bond"


@dataclass(frozen=True)
class VarietyInfo:
    """A futures variety used for variety-level attribution."""

    symbol: str
    name: str
    exchange: str  # SHFE | DCE | CZCE | CFFEX | INE
    sector: str  # 黑色 | 有色 | 贵金属 | 能化 | 农产品 | 金融


# ---------------------------------------------------------------------------
# Benchmark indices for strategy-type classification
# ---------------------------------------------------------------------------

INDEX_BENCHMARKS: dict[str, IndexInfo] = {
    "hs300": IndexInfo("hs300", "沪深300", "sh000300", "equity"),
    "zz500": IndexInfo("zz500", "中证500", "sh000905", "equity"),
    "zz1000": IndexInfo("zz1000", "中证1000", "sh000852", "equity"),
    "nh_commodity": IndexInfo("nh_commodity", "南华商品指数", "NHCI", "commodity"),
    "nh_industrial": IndexInfo("nh_industrial", "南华工业品指数", "NHII", "commodity"),
    "nh_agriculture": IndexInfo("nh_agriculture", "南华农产品指数", "NHAI", "commodity"),
    "nh_metal": IndexInfo("nh_metal", "南华金属指数", "NHMI", "commodity"),
    "nh_energy": IndexInfo("nh_energy", "南华能化指数", "NHEI", "commodity"),
}

# ---------------------------------------------------------------------------
# Active domestic futures varieties (~28)
# ---------------------------------------------------------------------------

FUTURES_VARIETIES: dict[str, VarietyInfo] = {
    # 黑色板块
    "rb": VarietyInfo("rb", "螺纹钢", "SHFE", "黑色"),
    "hc": VarietyInfo("hc", "热卷", "SHFE", "黑色"),
    "i": VarietyInfo("i", "铁矿石", "DCE", "黑色"),
    "j": VarietyInfo("j", "焦炭", "DCE", "黑色"),
    "jm": VarietyInfo("jm", "焦煤", "DCE", "黑色"),
    "ss": VarietyInfo("ss", "不锈钢", "SHFE", "黑色"),
    # 有色金属
    "cu": VarietyInfo("cu", "沪铜", "SHFE", "有色"),
    "al": VarietyInfo("al", "沪铝", "SHFE", "有色"),
    "zn": VarietyInfo("zn", "沪锌", "SHFE", "有色"),
    "ni": VarietyInfo("ni", "沪镍", "SHFE", "有色"),
    "sn": VarietyInfo("sn", "沪锡", "SHFE", "有色"),
    # 贵金属
    "au": VarietyInfo("au", "沪金", "SHFE", "贵金属"),
    "ag": VarietyInfo("ag", "沪银", "SHFE", "贵金属"),
    # 能源化工
    "sc": VarietyInfo("sc", "原油", "INE", "能化"),
    "fu": VarietyInfo("fu", "燃料油", "SHFE", "能化"),
    "ta": VarietyInfo("ta", "PTA", "CZCE", "能化"),
    "ma": VarietyInfo("ma", "甲醇", "CZCE", "能化"),
    "pp": VarietyInfo("pp", "聚丙烯", "DCE", "能化"),
    "eg": VarietyInfo("eg", "乙二醇", "DCE", "能化"),
    "bu": VarietyInfo("bu", "沥青", "SHFE", "能化"),
    # 农产品
    "m": VarietyInfo("m", "豆粕", "DCE", "农产品"),
    "y": VarietyInfo("y", "豆油", "DCE", "农产品"),
    "p": VarietyInfo("p", "棕榈油", "DCE", "农产品"),
    "oi": VarietyInfo("oi", "菜油", "CZCE", "农产品"),
    "cf": VarietyInfo("cf", "棉花", "CZCE", "农产品"),
    "sr": VarietyInfo("sr", "白糖", "CZCE", "农产品"),
    "rm": VarietyInfo("rm", "菜粕", "CZCE", "农产品"),
    # 金融期货
    "IF": VarietyInfo("IF", "沪深300股指", "CFFEX", "金融"),
    "IC": VarietyInfo("IC", "中证500股指", "CFFEX", "金融"),
    "T": VarietyInfo("T", "十年国债", "CFFEX", "金融"),
}

# Sector color mapping for frontend visualization
SECTOR_COLORS: dict[str, str] = {
    "黑色": "#4a4a4a",
    "有色": "#e8890c",
    "贵金属": "#d4af37",
    "能化": "#1890ff",
    "农产品": "#52c41a",
    "金融": "#722ed1",
}


def get_all_symbols() -> list[dict]:
    """Return a flat list of all available symbols for the frontend."""
    result = []
    for info in INDEX_BENCHMARKS.values():
        result.append({
            "symbol": info.symbol,
            "name": info.name,
            "type": "index",
            "asset_class": info.asset_class,
            "sector": None,
            "exchange": None,
        })
    for info in FUTURES_VARIETIES.values():
        result.append({
            "symbol": info.symbol,
            "name": info.name,
            "type": "futures",
            "asset_class": "commodity" if info.sector != "金融" else "financial",
            "sector": info.sector,
            "exchange": info.exchange,
        })
    return result


def get_varieties_by_sector(sector: str) -> list[VarietyInfo]:
    """Filter futures varieties by sector name."""
    return [v for v in FUTURES_VARIETIES.values() if v.sector == sector]


def get_all_sectors() -> list[str]:
    """Return unique sector names in display order."""
    order = ["黑色", "有色", "贵金属", "能化", "农产品", "金融"]
    return [s for s in order if s in {v.sector for v in FUTURES_VARIETIES.values()}]
