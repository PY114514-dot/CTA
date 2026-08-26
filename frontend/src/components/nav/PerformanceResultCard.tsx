import { Alert, Button, Card, Col, Descriptions, Divider, Empty, Row, Space } from "antd";
import type { KbProduct, NavAnalysisResponse } from "../../api";
import { Metric } from "../display";
import { decimal, formatCumulativeReturn, percentage } from "../../utils/format";

/** Results presentation only; calculations and evidence decisions remain in the page workflow. */
type Disclosure = NonNullable<KbProduct["nav_quality"]>;

export default function PerformanceResultCard({ analysis, error, disclosedDifference, benchmarkReturn, disclosure, onExport, onSaveHistory }: { analysis?: NavAnalysisResponse; error?: string; disclosedDifference?: number; benchmarkReturn?: number; disclosure?: Disclosure; onExport: () => void; onSaveHistory: () => void }): React.JSX.Element {
  const hasDisclosure = Boolean(disclosure && [disclosure.disclosed_cumulative_return, disclosure.disclosed_annualized_return, disclosure.disclosed_maximum_drawdown, disclosure.disclosed_sharpe_ratio].some((value) => value != null));
  const comparison = (computed: number | null, disclosed: number | null, tolerance: number): string => {
    if (computed == null || disclosed == null) return "未披露";
    return Math.abs(computed - disclosed) <= tolerance ? "吻合" : "需复核";
  };
  return <Card title="业绩结果" className="card-accent-top card-hover" style={{ flex: 1 }}>
    {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} closable />}
    {disclosedDifference !== undefined && <Alert type={Math.abs(disclosedDifference) > 0.02 ? "warning" : "success"} showIcon style={{ marginBottom: 16 }} message={Math.abs(disclosedDifference) > 0.02 ? `图像候选累计收益与披露值相差 ${(Math.abs(disclosedDifference) * 100).toFixed(2)} 个百分点；请核对截至日期、复权口径和是否为同一产品。` : "图像候选累计收益与披露值的差异在 2 个百分点以内。"} closable />}
    {analysis && benchmarkReturn !== undefined && <Alert type="info" showIcon style={{ marginBottom: 16 }} message={`图例灰色基准候选累计收益 ${formatCumulativeReturn(benchmarkReturn)}；产品相对基准的候选超额收益 ${formatCumulativeReturn(analysis.metrics.cumulative_return - benchmarkReturn)}。`} closable />}
    {analysis ? <><Row gutter={[16, 24]}><Metric label="累计收益" value={percentage(analysis.metrics.cumulative_return)} /><Metric label="年化收益" value={percentage(analysis.metrics.annualized_return)} /><Metric label="年化波动" value={percentage(analysis.metrics.annualized_volatility)} /><Metric label="夏普比率" value={decimal(analysis.metrics.sharpe_ratio)} /><Metric label="最大回撤" value={percentage(analysis.metrics.maximum_drawdown)} /><Metric label="卡玛比率" value={decimal(analysis.metrics.calmar_ratio)} /></Row>{hasDisclosure && <Card size="small" type="inner" title="材料披露校验" style={{ marginTop: 16 }} extra="仅辅助复核，不修改净值">
      <Descriptions size="small" bordered column={{ xs: 1, sm: 2, lg: 4 }}>
        <Descriptions.Item label="累计收益">曲线 {percentage(analysis.metrics.cumulative_return)} · 披露 {percentage(disclosure?.disclosed_cumulative_return ?? null)} · {comparison(analysis.metrics.cumulative_return, disclosure?.disclosed_cumulative_return ?? null, 0.03)}</Descriptions.Item>
        <Descriptions.Item label="年化收益">曲线 {percentage(analysis.metrics.annualized_return)} · 披露 {percentage(disclosure?.disclosed_annualized_return ?? null)} · {comparison(analysis.metrics.annualized_return, disclosure?.disclosed_annualized_return ?? null, 0.03)}</Descriptions.Item>
        <Descriptions.Item label="最大回撤">曲线 {percentage(analysis.metrics.maximum_drawdown)} · 披露 {percentage(disclosure?.disclosed_maximum_drawdown ?? null)} · {comparison(analysis.metrics.maximum_drawdown, disclosure?.disclosed_maximum_drawdown ?? null, 0.05)}</Descriptions.Item>
        <Descriptions.Item label="夏普比率">曲线 {decimal(analysis.metrics.sharpe_ratio)} · 披露 {decimal(disclosure?.disclosed_sharpe_ratio ?? null)} · 仅参考（可能采用不同无风险利率或年化口径）</Descriptions.Item>
      </Descriptions>
    </Card>}<Divider style={{ margin: "16px 0 12px" }} /><Space wrap><Button onClick={onExport}>导出净值 XLSX</Button><Button onClick={onSaveHistory}>保存到历史记录</Button></Space></> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="提交经核验的净值数据后显示结果" style={{ marginTop: 24 }} />}
  </Card>;
}
