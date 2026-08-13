import { Alert, Button, Card, Col, Divider, Empty, Row, Space } from "antd";
import type { NavAnalysisResponse } from "../../api";
import { Metric } from "../display";
import { decimal, formatCumulativeReturn, percentage } from "../../utils/format";

/** Results presentation only; calculations and evidence decisions remain in the page workflow. */
export default function PerformanceResultCard({ analysis, error, disclosedDifference, benchmarkReturn, onExport, onSaveHistory }: { analysis?: NavAnalysisResponse; error?: string; disclosedDifference?: number; benchmarkReturn?: number; onExport: () => void; onSaveHistory: () => void }): React.JSX.Element {
  return <Card title="业绩结果" className="card-accent-top card-hover" style={{ flex: 1 }}>
    {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} closable />}
    {disclosedDifference !== undefined && <Alert type={Math.abs(disclosedDifference) > 0.02 ? "warning" : "success"} showIcon style={{ marginBottom: 16 }} message={Math.abs(disclosedDifference) > 0.02 ? `图像候选累计收益与披露值相差 ${(Math.abs(disclosedDifference) * 100).toFixed(2)} 个百分点；请核对截至日期、复权口径和是否为同一产品。` : "图像候选累计收益与披露值的差异在 2 个百分点以内。"} closable />}
    {analysis && benchmarkReturn !== undefined && <Alert type="info" showIcon style={{ marginBottom: 16 }} message={`图例灰色基准候选累计收益 ${formatCumulativeReturn(benchmarkReturn)}；产品相对基准的候选超额收益 ${formatCumulativeReturn(analysis.metrics.cumulative_return - benchmarkReturn)}。`} closable />}
    {analysis ? <><Row gutter={[16, 24]}><Metric label="累计收益" value={percentage(analysis.metrics.cumulative_return)} /><Metric label="年化收益" value={percentage(analysis.metrics.annualized_return)} /><Metric label="年化波动" value={percentage(analysis.metrics.annualized_volatility)} /><Metric label="夏普比率" value={decimal(analysis.metrics.sharpe_ratio)} /><Metric label="最大回撤" value={percentage(analysis.metrics.maximum_drawdown)} /><Metric label="卡玛比率" value={decimal(analysis.metrics.calmar_ratio)} /></Row><Divider style={{ margin: "16px 0 12px" }} /><Space wrap><Button onClick={onExport}>导出净值 XLSX</Button><Button onClick={onSaveHistory}>保存到历史记录</Button></Space></> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="提交经核验的净值数据后显示结果" style={{ marginTop: 24 }} />}
  </Card>;
}
