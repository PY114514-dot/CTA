/** Return-based style risk view; deliberately separate from attribution phases. */

import React, { useEffect, useState } from "react";
import { Alert, Descriptions, Empty, Spin, Table } from "antd";
import { type KbProduct } from "../api";
import { getCtaStyleRisk, type CtaStyleRiskResponse } from "./ctaStyleRiskApi";

const FACTOR_LABELS: Record<string, string> = {
  trend: "中长期趋势",
  short_term_trend_20: "短期动量",
  cross_section_mom: "截面动量",
  mean_reversion_5d: "短期反转",
  term_structure_carry: "期限结构 Carry",
  calendar_spread_momentum: "跨期价差动量",
  volatility_state: "波动率状态",
};

export default function CtaStyleRiskPanel({ product }: { product: KbProduct | undefined }): React.JSX.Element {
  const [result, setResult] = useState<CtaStyleRiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ready = product?.research_workflow?.stage === "research_ready" && (product.reviewed_nav_count ?? 0) >= 20;
  useEffect(() => {
    if (!product || !ready) { setResult(null); return; }
    let active = true;
    setError(null);
    getCtaStyleRisk(product.id).then((value) => { if (active) setResult(value); }).catch((cause) => {
      if (active) setError(cause instanceof Error ? cause.message : "CTA 风格风险评价失败");
    });
    return () => { active = false; };
  }, [product, ready]);

  if (!product) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="未选择产品" />;
  if (!ready) return <Alert type="info" showIcon message="风格风险评价样本不足" description={`已审核净值 ${product.reviewed_nav_count} 条。`} />;
  if (error) return <Alert type="error" showIcon message={error} />;
  if (!result) return <div style={{ padding: 42, textAlign: "center" }}><Spin tip="正在读取已审核净值和公开因子…" /></div>;
  return <>
    <Alert type="warning" showIcon message="基于净值的风格风险参考" description="这是公开市场因子下的统计关联与风险拆分，不代表实际持仓、品种权重或交易意图。" />
    <Descriptions size="small" bordered column={3} style={{ marginTop: 12 }}>
      <Descriptions.Item label="对齐期数">{result.alignment.aligned_observation_count}</Descriptions.Item>
      <Descriptions.Item label="系统风险占比">{result.risk.total_variance > 0 ? `${(result.risk.systematic_variance / result.risk.total_variance * 100).toFixed(1)}%` : "—"}</Descriptions.Item>
      <Descriptions.Item label="特质风险占比">{result.risk.idiosyncratic_risk_share == null ? "—" : `${(result.risk.idiosyncratic_risk_share * 100).toFixed(1)}%`}</Descriptions.Item>
      <Descriptions.Item label="风格漂移窗口">{result.style_drift.window} 期</Descriptions.Item>
      <Descriptions.Item label="因子条件数">{result.diagnostics.condition_number.toFixed(1)}</Descriptions.Item>
      <Descriptions.Item label="使用未来数据">{result.style_drift.causal ? "否" : "是"}</Descriptions.Item>
      <Descriptions.Item label="样本范围">{result.alignment.start_date} 至 {result.alignment.end_date}</Descriptions.Item>
    </Descriptions>
    <Table size="small" pagination={false} rowKey="factor_name" dataSource={result.factor_exposures} style={{ marginTop: 12 }} columns={[
      { title: "风格因子", dataIndex: "factor_name", render: (value: string) => FACTOR_LABELS[value] ?? value },
      { title: "关联系数", dataIndex: "beta", render: (value: number) => value.toFixed(4) },
      { title: "风险贡献 / 总方差", dataIndex: "risk_contribution", render: (value: number | null) => value == null ? "—" : `${(value * 100).toFixed(1)}%` },
      { title: "分窗口系数范围", render: (_: unknown, row: CtaStyleRiskResponse["factor_exposures"][number]) => result.style_drift.beta_range[row.factor_name]?.toFixed(4) ?? "—" },
    ]} />
    {result.warnings.length > 2 && <Alert type="warning" showIcon style={{ marginTop: 12 }} message="风格暴露稳定性限制" description={result.warnings.slice(2).join("；")} />}
  </>;
}
