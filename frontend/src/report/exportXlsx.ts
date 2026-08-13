/**
 * XLSX export for NAV data and metrics.
 *
 * Extracted from main.tsx for maintainability.
 */
import type { NavAnalysisResponse } from "../api";
import { parseNavText } from "../parser/navParser";

/** Export NAV points and metrics to an XLSX workbook. */
export async function exportNavXlsx(
  navText: string,
  metrics: NavAnalysisResponse["metrics"] | undefined,
  productName: string,
): Promise<void> {
  // Lazy-load the heavy xlsx library only when the user actually exports.
  const XLSX = await import("xlsx");
  const points = parseNavText(navText);
  const navData = points.map((p) => ({ 日期: p.observation_date, 单位净值: p.net_asset_value }));

  const wb = XLSX.utils.book_new();
  const wsNav = XLSX.utils.json_to_sheet(navData);
  wsNav["!cols"] = [{ wch: 12 }, { wch: 14 }];
  XLSX.utils.book_append_sheet(wb, wsNav, "净值曲线");

  if (metrics) {
    const metricsData = [
      { 指标: "累计收益", 数值: `${(metrics.cumulative_return * 100).toFixed(2)}%` },
      { 指标: "年化收益", 数值: `${(metrics.annualized_return * 100).toFixed(2)}%` },
      { 指标: "年化波动", 数值: `${(metrics.annualized_volatility * 100).toFixed(2)}%` },
      { 指标: "夏普比率", 数值: metrics.sharpe_ratio !== null ? metrics.sharpe_ratio.toFixed(4) : "—" },
      { 指标: "最大回撤", 数值: `${(metrics.maximum_drawdown * 100).toFixed(2)}%` },
      { 指标: "卡玛比率", 数值: metrics.calmar_ratio !== null ? metrics.calmar_ratio.toFixed(4) : "—" },
    ];
    const wsMetrics = XLSX.utils.json_to_sheet(metricsData);
    wsMetrics["!cols"] = [{ wch: 12 }, { wch: 14 }];
    XLSX.utils.book_append_sheet(wb, wsMetrics, "业绩指标");
  }

  const safeName = (productName || "产品").replace(/[\\/:*?"<>|]/g, "_");
  XLSX.writeFile(wb, `${safeName}_净值曲线.xlsx`);
}
