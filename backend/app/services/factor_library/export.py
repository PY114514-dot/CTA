"""Factor library export helpers: CSV and Markdown formats.

Produces a combined report of the factor catalog (metadata) and, when
available, cached performance statistics.  Used by the /export endpoint.
"""

import csv
import io
from datetime import datetime

from app.services import factor_library


def _catalog_rows() -> list[dict]:
    """Return one row per registered factor with metadata + performance."""
    perf_by_name: dict[str, dict] = {
        p["name"]: p for p in factor_library.compute_factor_performance()
    }
    rows: list[dict] = []
    for f in factor_library.get_all_factors():
        p = perf_by_name.get(f["name"], {})
        rows.append({
            "因子代码": f["name"],
            "因子名称": f["display_name"],
            "类别": f["category"],
            "说明": f["description"],
            "默认参数": str(f["params"]),
            "已缓存": "是" if f["cached"] else "否",
            "年化收益": f'{p["annualized_return"]:.2%}' if p else "—",
            "年化波动": f'{p["annualized_vol"]:.2%}' if p else "—",
            "夏普": f'{p["sharpe"]:.2f}' if p else "—",
            "最大回撤": f'{p["max_drawdown"]:.2%}' if p else "—",
            "卡玛": f'{p["calmar"]:.2f}' if p else "—",
            "胜率": f'{p["win_rate"]:.2%}' if p else "—",
            "样本行数": str(p.get("rows", "—")) if p else "—",
            "区间": f'{p.get("start", "")} ~ {p.get("end", "")}' if p else "—",
        })
    return rows


def export_csv() -> str:
    """Return Excel-compatible UTF-8 CSV (with BOM) for Chinese headers."""
    rows = _catalog_rows()
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    # Excel on Chinese Windows otherwise commonly assumes GBK/ANSI when a CSV
    # is opened directly, corrupting factor names and descriptions.
    return "\ufeff" + buf.getvalue()


def export_markdown() -> str:
    """Return the factor catalog + performance as a Markdown document."""
    rows = _catalog_rows()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        "## CTA 因子库总览",
        "",
        f"> 导出时间：{now}",
        "",
    ]

    if not rows:
        lines.append("（无已注册因子）")
        return "\n".join(lines)

    # Catalog table
    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    lines.append("")

    # Formula quick-reference
    lines.extend(["### 公式速查", ""])
    lines.append("| 因子 | 信号公式 |")
    lines.append("| --- | --- |")
    for f in factor_library.get_all_factors():
        detail = factor_library.get_factor_detail(f["name"])
        if detail is None:
            continue
        formula = detail.get("formula", "").strip()
        # Flatten to single line for table cell
        formula_flat = " ".join(formula.split())
        lines.append(f"| {f['display_name']} | `{formula_flat}` |")

    lines.extend([
        "",
        "---",
        "*本表由 CTA 研究平台自动生成。公式记号约定见 base.py 模块文档。*",
        "",
    ])
    return "\n".join(lines)
