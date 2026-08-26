/** Read-only CTA product score panel.
 *
 * Shows one summary row per product (quality / confidence / attribution
 * quality scores plus headline regression numbers) and lets the user expand
 * a row for the full drill-down detail: quality dimensions, confidence
 * components, factor attribution with trend/choppy regime split, and the
 * quality-score history.  Allocation (配置分) is deliberately not shown
 * here because it is a portfolio-conditional score.
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Collapse, Empty, Space, Spin, Table, Tag, Typography, message } from "antd";
import { PlayCircleOutlined } from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import {
  getLatestCtaProductScoreItem,
  getLatestCtaProductScoreSnapshot,
  kbGetNavSeries,
  persistCtaProductScoreSnapshot,
  type CtaAttributionDetail,
  type CtaConfidenceScore,
  type CtaProductScoreItem,
  type CtaProductScoreListItem,
  type CtaProductScoreListResponse,
  type CtaQualityHistoryPoint,
  type CtaReturnAnalysis,
  type CtaRiskExposure,
  type CtaScoreComponentScore,
} from "../api";
import { buildCtaUniverse } from "./ctaUniverse";
import { dimensionLabel } from "./ctaScoreLabels";

const { Text } = Typography;

const BAND_COLORS: Record<string, string> = { high: "green", medium: "blue", low: "orange" };
const BAND_LABELS: Record<string, string> = { high: "高", medium: "中", low: "低" };
const FACTOR_LABELS: Record<string, string> = {
  trend: "长期规则（趋势）",
  short_term_trend_20: "短期时序动量（20日）",
  cross_section_mom: "长期截面（截面动量）",
  mean_reversion_5d: "短期反转（5日）",
  term_structure_carry: "期限结构收益",
  calendar_spread_momentum: "跨期价差动量",
  volatility_state: "波动率状态",
};

function toNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function fmt(value: number | null | undefined, digits = 2): string {
  return value == null ? "—" : value.toFixed(digits);
}

function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

function readableWarning(warning: string): string {
  if (warning.startsWith("Phase D 证据缺失")) return "当前评分未纳入非线性模型的样本外验证";
  return warning;
}

function factorLabel(name: string, displayName: string): string {
  return FACTOR_LABELS[name] ?? displayName;
}

function bandTag(band: string | null | undefined): React.JSX.Element | null {
  if (!band || !BAND_LABELS[band]) return null;
  return <Tag color={BAND_COLORS[band]} style={{ marginInlineEnd: 0 }}>{BAND_LABELS[band]}</Tag>;
}

function HeadlineLine({ item }: { item: CtaProductScoreListItem }): React.JSX.Element {
  const headline = item.summary.headline;
  const returnRiskParts = [
    `年化收益 ${fmtPct(headline.annualized_return)}`,
    `累计 ${fmtPct(headline.cumulative_return)}`,
    `最大回撤 ${fmtPct(headline.maximum_drawdown)}`,
    `年化波动 ${fmtPct(headline.annualized_volatility)}`,
    `VaR95 ${fmtPct(headline.var_95)}`,
  ];
  return (
    <span style={{ display: "block" }}>
      <Text type="secondary" style={{ display: "block", fontSize: 11 }}>
        {returnRiskParts.join(" · ")}
      </Text>
    </span>
  );
}

function QualityDimensions({ item }: { item: CtaProductScoreItem }): React.JSX.Element {
  const explanation = item.detail.quality_explanation;
  const columns = [
    { title: "评分维度", dataIndex: "label" },
    { title: "权重", dataIndex: "weight", width: 92, render: (value: number) => `${value.toFixed(1)}%` },
    { title: "覆盖", key: "coverage", width: 88, render: (_value: unknown, row: typeof explanation.dimensions[number]) => `${row.covered_metric_count}/${row.metric_count} 项` },
    { title: "得分", dataIndex: "adjusted_score", width: 88, render: (value: number | null) => value == null ? "未形成" : fmt(value, 1) },
  ];
  return (
    <Card size="small" type="inner" title="评分构成">
      <Table
        rowKey="dimension"
        size="small"
        pagination={false}
        columns={columns}
        dataSource={explanation.dimensions.map((dimension) => ({ ...dimension, label: dimensionLabel(dimension.dimension, dimension.label) }))}
      />
      {explanation.warnings.length > 0 && <Alert type="warning" showIcon message={explanation.warnings.map(readableWarning).join("；")} style={{ marginTop: 10 }} />}
    </Card>
  );
}

function ConfidenceCard({ confidence }: { confidence: CtaConfidenceScore }): React.JSX.Element {
  const componentLabels: Record<string, string> = {
    sample_length: "样本长度",
    data_coverage: "数据覆盖",
    beta_uncertainty: "因子关联不确定性",
    factor_alignment_coverage: "因子对齐覆盖",
    state_sample: "状态样本",
    oos_stability: "样本外稳定性",
    input_version_completeness: "输入版本完整度",
  };
  return (
    <Card size="small" type="inner" title="结论置信度">
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Space wrap>
          <Text strong>置信分 {fmt(confidence.score, 1)}</Text>
          {bandTag(confidence.band)}
        </Space>
        <Space wrap size={[6, 6]}>
          {Object.entries(confidence.components).map(([name, value]) => (
            <Tag key={name} color={value == null ? "default" : "blue"}>
              {componentLabels[name] ?? name}：{value == null ? "未纳入" : `${fmt(value, 1)}（权重 ${fmt(confidence.weights[name], 2)}）`}
            </Tag>
          ))}
        </Space>
        {confidence.warnings.map((warning) => <Tag key={warning} color="warning">{readableWarning(warning)}</Tag>)}
      </Space>
    </Card>
  );
}

function ComponentScoreCard({ title, score }: { title: string; score: CtaScoreComponentScore | null | undefined }): React.JSX.Element {
  const componentLabels: Record<string, string> = {
    return_level: "收益水平",
    consistency: "正收益期占比",
    drawdown_control: "回撤控制",
    volatility_control: "波动控制",
    tail_control: "尾部控制",
    concentration_control: "集中度控制",
  };
  if (!score) {
    return (
      <Card size="small" type="inner" title={title}>
        <Tag color="warning">旧版本快照暂无该评分</Tag>
      </Card>
    );
  }
  return (
    <Card size="small" type="inner" title={title}>
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Space wrap>
          {score.score == null ? (
            <Tag color="warning">{score.status === "short_sample" ? "样本不足未打分" : "未形成评分"}</Tag>
          ) : (
            <>
              <Text strong>{fmt(score.score, 1)}</Text>
              {bandTag(score.band)}
            </>
          )}
        </Space>
        <Space wrap size={[6, 6]}>
          {Object.entries(score.components).map(([name, value]) => (
            <Tag key={name} color={value == null ? "default" : "blue"}>
              {componentLabels[name] ?? name}：{value == null ? "未纳入" : `${fmt(value, 1)}（权重 ${fmt(score.weights[name], 2)}）`}
            </Tag>
          ))}
        </Space>
        {score.warnings.map((warning) => <Tag key={warning} color="warning">{warning}</Tag>)}
      </Space>
    </Card>
  );
}

function DrawdownSparkline({ path }: { path: Array<{ date: string; drawdown: number }> }): React.JSX.Element | null {
  if (path.length < 2) return null;
  const width = 900;
  const height = 96;
  const padding = 2;
  const drawdowns = path.map((point) => point.drawdown);
  const min = Math.min(0, ...drawdowns);
  const max = 0;
  const span = Math.max(max - min, 1e-12);
  const step = (width - 2 * padding) / (path.length - 1);
  const points = path.map((point, index) => {
    const x = padding + index * step;
    const y = padding + ((max - point.drawdown) / span) * (height - 2 * padding);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ display: "block", background: "#fafafa", borderRadius: 4 }}>
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#d9d9d9" strokeWidth={1} />
      <polygon
        points={`${padding},${height - padding} ${points.join(" ")} ${width - padding},${height - padding}`}
        fill="rgba(245, 34, 45, 0.12)"
        stroke="none"
      />
      <polyline points={points.join(" ")} fill="none" stroke="#f5222d" strokeWidth={1.5} />
    </svg>
  );
}

function ProductNavChart({ productId }: { productId: string }): React.JSX.Element | null {
  const [nav, setNav] = useState<Array<{ date: string; value: number }> | null>(null);
  useEffect(() => {
    let active = true;
    void kbGetNavSeries(productId, true)
      .then((points) => {
        if (!active) return;
        setNav(points
          .filter((point) => Number.isFinite(point.nav) && point.nav > 0)
          .sort((left, right) => left.observation_date.localeCompare(right.observation_date))
          .map((point) => ({ date: point.observation_date, value: point.nav })));
      })
      .catch(() => { if (active) setNav([]); });
    return () => { active = false; };
  }, [productId]);
  if (nav == null) return <div style={{ height: 150, display: "grid", placeItems: "center" }}><Spin size="small" /></div>;
  if (nav.length < 2) return null;
  return <ReactECharts
    option={{
      grid: { left: 52, right: 16, top: 12, bottom: 28 },
      tooltip: { trigger: "axis", valueFormatter: (value: number | string) => Number(value).toFixed(4) },
      xAxis: { type: "category", data: nav.map((point) => point.date), axisLabel: { hideOverlap: true, fontSize: 10 } },
      yAxis: { type: "value", scale: true, name: "单位净值", nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 } },
      series: [{ type: "line", data: nav.map((point) => point.value), showSymbol: false, smooth: true, lineStyle: { color: "#1677ff", width: 2 }, areaStyle: { color: "rgba(22,119,255,.10)" } }],
    }}
    style={{ height: 150 }}
  />;
}

function ReturnsAnalysisCard({ productId, analysis }: { productId: string; analysis: CtaReturnAnalysis | null | undefined }): React.JSX.Element {
  if (!analysis || analysis.status !== "available") {
    return (
      <Card size="small" type="inner" title="收益分析">
        <Alert type="warning" showIcon message="收益分析暂不可用" description={analysis ? analysis.warnings.join("；") : "旧版本快照暂无收益分析数据，重新生成评分报告后可用。"} />
      </Card>
    );
  }
  const m = analysis.metrics;
  const years = Object.entries(analysis.calendar_year_returns);
  const yearColumns = [
    { title: "年份", dataIndex: "year", width: 70 },
    { title: "年度收益", dataIndex: "value", width: 90, render: (value: number) => <Text style={{ color: value >= 0 ? "#52c41a" : "#f5222d" }}>{fmtPct(value)}</Text> },
  ];
  return (
    <Card size="small" type="inner" title={`收益分析（${analysis.start_date} ~ ${analysis.end_date}，${analysis.observation_count} 期）`}>
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Space wrap size={[6, 6]}>
          <Tag color="green">累计收益 {fmtPct(m.cumulative_return)}</Tag>
          <Tag color="green">年化收益 {fmtPct(m.annualized_return)}</Tag>
          <Tag>年化波动 {fmtPct(m.annualized_volatility)}</Tag>
          <Tag>正收益期占比 {fmtPct(m.positive_period_ratio, 1)}</Tag>
          <Tag>盈亏不对称比 {fmt(m.gain_loss_asymmetry)}</Tag>
          <Tag>最佳期 {fmtPct(m.best_period_return)}</Tag>
          <Tag>最差期 {fmtPct(m.worst_period_return)}</Tag>
          <Tag>滚动 26 期 {fmtPct(m.rolling_26_period_return)}</Tag>
          <Tag>滚动 52 期 {fmtPct(m.rolling_52_period_return)}</Tag>
        </Space>
        <Space wrap size={[6, 6]}>
          <Tag color="red">最大回撤 {fmtPct(m.maximum_drawdown)}</Tag>
          <Tag>当前回撤 {fmtPct(m.current_drawdown)}</Tag>
          <Tag>最长回撤持续 {fmt(m.longest_drawdown_duration_periods, 0)} 期</Tag>
          <Tag>平均修复 {fmt(m.average_recovery_periods, 1)} 期</Tag>
          <Tag>回撤片段 {m.drawdown_episode_count} 次</Tag>
          <Tag color={m.recovery_completed ? "success" : "warning"}>{m.recovery_completed ? "当前已修复" : "当前处于回撤中"}</Tag>
        </Space>
        <div style={{ display: "grid", gridTemplateColumns: years.length > 0 ? "220px minmax(0, 1fr)" : "minmax(0, 1fr)", gap: 20, alignItems: "end", marginTop: 10 }}>
          {years.length > 0 && <Table
            rowKey="year"
            size="small"
            pagination={false}
            columns={yearColumns}
            dataSource={years.map(([year, value]) => ({ year, value }))}
          />}
          <div>
            <Text type="secondary" style={{ display: "block", marginBottom: 6, fontSize: 12 }}>产品净值走势（已复核净值）</Text>
            <ProductNavChart productId={productId} />
            <Text type="secondary" style={{ display: "block", marginBottom: 6, fontSize: 12 }}>回撤走势（相对历史最高净值的下跌幅度）</Text>
            <DrawdownSparkline path={analysis.drawdown_path} />
          </div>
        </div>
        {analysis.warnings.map((warning) => <Tag key={warning} color="warning">{warning}</Tag>)}
      </Space>
    </Card>
  );
}

function RiskExposureCard({ exposure }: { exposure: CtaRiskExposure | null | undefined }): React.JSX.Element {
  if (!exposure || exposure.status !== "available") {
    return (
      <Card size="small" type="inner" title="风险暴露">
        <Alert type="warning" showIcon message="风险暴露暂不可用" description={exposure ? exposure.warnings.join("；") : "旧版本快照暂无风险暴露数据，重新生成评分报告后可用。"} />
      </Card>
    );
  }
  const s = exposure.statistical;
  const factorColumns = [
    { title: "因子", dataIndex: "display_name", ellipsis: true, render: (value: string, row: { name: string }) => factorLabel(row.name, value) },
    { title: "关联系数", dataIndex: "beta", width: 84, render: (value: number) => fmt(value, 3) },
    { title: "t", dataIndex: "t_stat", width: 70, render: (value: number) => fmt(value, 1) },
    {
      title: "风险贡献占比",
      dataIndex: "risk_contribution_pct",
      width: 112,
      render: (value: number | null) => value == null ? "—" : `${fmt(value, 1)}%`,
    },
    {
      title: "显著",
      dataIndex: "significant",
      width: 60,
      render: (value: boolean) => value ? <Tag color="success">是</Tag> : <Tag>否</Tag>,
    },
  ];
  return (
    <Card size="small" type="inner" title="风险暴露">
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Space wrap size={[6, 6]}>
          <Tag color="orange">年化波动 {fmtPct(s.annualized_volatility)}</Tag>
          <Tag>下行波动 {fmtPct(s.downside_deviation)}</Tag>
          <Tag color="red">VaR95 {fmtPct(s.var_95)}</Tag>
          <Tag color="red">VaR99 {fmtPct(s.var_99)}</Tag>
          <Tag color="red">CVaR95 {fmtPct(s.cvar_95)}</Tag>
          <Tag>偏度 {fmt(s.skewness)}</Tag>
          <Tag>超额峰度 {fmt(s.excess_kurtosis)}</Tag>
          <Tag>最差期 {fmtPct(s.worst_period_return)}</Tag>
        </Space>
        <Space wrap size={[6, 6]}>
          <Tag>系统风险占比 {fmt(exposure.systematic_variance_share)}</Tag>
          <Tag>未解释占比 {fmt(exposure.unexplained_variance_share)}</Tag>
          <Tag>残差年化波动 {fmtPct(exposure.residual_annual_volatility)}</Tag>
          {exposure.concentration.normalized_hhi != null && (
            <Tag color={exposure.concentration.normalized_hhi > 0.5 ? "red" : "blue"}>
              风险集中度 {fmt(exposure.concentration.normalized_hhi)}
            </Tag>
          )}
        </Space>
        {exposure.factor_exposure_status === "available" && exposure.factor_exposures.length > 0 && (
          <Table
            rowKey="name"
            size="small"
            pagination={false}
            columns={factorColumns}
            dataSource={exposure.factor_exposures}
            scroll={{ x: 480 }}
          />
        )}
        {exposure.warnings.length > 0 && (
          <Alert type="warning" showIcon message={exposure.warnings.join("；")} />
        )}
      </Space>
    </Card>
  );
}

function RegimeCard({ attribution }: { attribution: CtaAttributionDetail }): React.JSX.Element | null {
  const regime = attribution.regime;
  if (!regime) return null;
  return (
    <Card
      size="small"
      type="inner"
      title={`趋势 / 震荡期 R² 拆分（驱动因子 ${regime.regime_factor}，${regime.window} 期滚动窗口，趋势期占比 ${(regime.trending_share * 100).toFixed(0)}%）`}
    >
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Space wrap size={[6, 6]}>
          <Tag color="green">趋势期：{regime.trending.observations} 期 · R² {fmt(regime.trending.r_squared)} · 年化 α {fmtPct(regime.trending.annualized_alpha)}</Tag>
          <Tag color="orange">震荡期：{regime.choppy.observations} 期 · R² {fmt(regime.choppy.r_squared)} · 年化 α {fmtPct(regime.choppy.annualized_alpha)}</Tag>
          <Tag>加入状态哑变量后 R² {fmt(regime.augmented.r_squared)}（增益 {fmt(regime.augmented.r_squared_gain)}）</Tag>
        </Space>
      </Space>
    </Card>
  );
}

function AttributionQualityCard({ attribution }: { attribution: CtaAttributionDetail }): React.JSX.Element | null {
  const quality = attribution.attribution_quality;
  if (!quality) return null;
  const componentLabels: Record<string, string> = {
    alpha_significance: "未解释收益不明显",
    alpha_bootstrap: "未解释收益区间覆盖零",
    oos_predictability: "样本外解释",
    factor_significance: "因子关系明确",
    sample_coverage: "样本覆盖",
  };
  return (
    <Card size="small" type="inner" title="归因可信度">
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Space wrap>
          <Text strong>归因可信度 {fmt(quality.score, 1)}</Text>
          {bandTag(quality.band)}
        </Space>
        <Text type="secondary" style={{ fontSize: 12 }}>分数衡量公开因子能否稳定解释历史收益，不评价产品好坏。未解释收益的 p 值越低，说明仍有较多收益未被当前因子解释，因此不会加分。</Text>
        <Space wrap size={[6, 6]}>
          {Object.entries(quality.components).map(([name, value]) => (
            <Tag key={name} color={value == null ? "default" : "blue"}>
              {componentLabels[name] ?? name}：{value == null ? "未纳入" : `${fmt(value, 1)}（权重 ${fmt(quality.weights[name], 2)}）`}
            </Tag>
          ))}
        </Space>
        {quality.warnings.map((warning) => <Tag key={warning} color="warning">{readableWarning(warning)}</Tag>)}
      </Space>
    </Card>
  );
}

function FactorTable({ attribution }: { attribution: CtaAttributionDetail }): React.JSX.Element | null {
  const factors = attribution.factors ?? [];
  if (factors.length === 0) return null;
  const columns = [
    { title: "因子", dataIndex: "display_name", ellipsis: true, render: (value: string, row: { name: string }) => factorLabel(row.name, value) },
    { title: "关联系数", dataIndex: "beta", width: 80, render: (value: number) => fmt(value, 3) },
    { title: "t", dataIndex: "t_stat", width: 70, render: (value: number) => fmt(value, 1) },
    {
      title: "统计可信度",
      dataIndex: "p_value",
      width: 132,
      render: (value: number) => value < 0.05
        ? <Tag color="success">p {value < 0.001 ? "<0.001" : `=${fmt(value, 4)}`}（&lt;0.05，显著）</Tag>
        : <Tag>p ={fmt(value, 4)}（≥0.05，不显著）</Tag>,
    },
    { title: "平均期收益解释", dataIndex: "mean_return_contribution", width: 112, render: (value: number) => fmtPct(value) },
  ];
  return (
    <Card size="small" type="inner" title="因子关联明细">
      <Table
        rowKey="name"
        size="small"
        pagination={false}
        columns={columns}
        dataSource={factors}
        scroll={{ x: 560 }}
      />
    </Card>
  );
}

function AttributionBlock({ attribution }: { attribution: CtaAttributionDetail }): React.JSX.Element {
  if (attribution.status !== "available") {
    return (
      <Card size="small" type="inner" title="因子归因">
        <Alert
          type="warning"
          showIcon
          message="归因暂不可用"
          description={attribution.warnings.join("；") || "因子库未构建，无法生成归因明细。"}
        />
      </Card>
    );
  }
  const oos = attribution.out_of_sample ?? {};
  const ci = attribution.alpha_bootstrap_ci;
  const reconciliation = attribution.return_reconciliation;
  const oosR2 = toNumber(oos.r_squared);
  const conclusion = oosR2 != null && oosR2 > 0
    ? "公开因子对该产品的收益特征有一定解释力，可作为辅助参考。"
    : "公开因子对该产品的解释力有限，不据此判断产品真实策略或管理能力。";
  return (
    <Card size="small" type="inner" title="因子归因">
      <Space direction="vertical" size={8} style={{ width: "100%" }}>
        <Text>{conclusion}</Text>
        <Text type="secondary">基于 {attribution.n_observations} 期已审核净值和公开因子，不代表真实持仓。</Text>
        <Collapse size="small" items={[{
          key: "attribution-details",
          label: "查看数据与方法明细",
          children: <Space direction="vertical" size={8} style={{ width: "100%" }}>
            <Space wrap size={[6, 6]}>
              <Tag>{attribution.start_date} 至 {attribution.end_date}</Tag>
              <Tag>{attribution.n_observations} 期已审核净值</Tag>
              <Tag color={(attribution.r_squared ?? 0) >= 0.3 ? "success" : "warning"}>样本内解释度 {fmtPct(attribution.r_squared, 0)}</Tag>
              <Tag color={(oosR2 ?? 0) > 0 ? "success" : "warning"}>样本外解释度 {fmtPct(oosR2, 0)}</Tag>
            </Space>
            <Space wrap size={[6, 6]}>
              <Tag color={attribution.alpha_p_value != null && attribution.alpha_p_value < 0.05 ? "warning" : "success"}>未解释收益 {attribution.alpha_p_value != null && attribution.alpha_p_value < 0.05 ? "仍较明显" : "不明显"}（p 值 {fmt(attribution.alpha_p_value, 4)}）</Tag>
              {ci && <Tag>未解释收益区间 [{fmtPct(ci.low)}, {fmtPct(ci.high)}]</Tag>}
              <Tag>统计方法：{attribution.inference_method}</Tag>
            </Space>
            <RegimeCard attribution={attribution} />
            <AttributionQualityCard attribution={attribution} />
            <FactorTable attribution={attribution} />
            {reconciliation && <Space wrap size={[6, 6]}>
              <Tag>产品平均期收益 {fmtPct(reconciliation.mean_product_return)}</Tag>
              <Tag>因子解释 {fmtPct(reconciliation.mean_factor_explained_return)}</Tag>
              <Tag>截距 {fmtPct(reconciliation.mean_intercept_return)}</Tag>
              <Tag>对账残差 {fmtPct(reconciliation.mean_residual_return)}</Tag>
            </Space>}
            {attribution.warnings.length > 0 && <Alert type="warning" showIcon message={attribution.warnings.map(readableWarning).join("；")} />}
          </Space>,
        }]} />
      </Space>
    </Card>
  );
}

function QualityHistoryBlock({ history }: { history: CtaQualityHistoryPoint[] }): React.JSX.Element | null {
  const scoredHistory = history.filter((point) => point.quality_score != null);
  if (scoredHistory.length < 2) return null;
  return (
    <Card size="small" type="inner" title="评分变化">
      <Space direction="vertical" size={4} style={{ width: "100%" }}>
        {scoredHistory.map((point) => (
          <div key={point.as_of_date} style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <Text style={{ fontSize: 12 }}>{point.as_of_date}</Text>
            <Text strong style={{ fontSize: 12 }}>{fmt(point.quality_score, 1)}</Text>
            {point.score_delta != null && (
              <Tag color={point.score_delta >= 0 ? "green" : "red"} style={{ fontSize: 11 }}>
                {point.score_delta >= 0 ? "+" : ""}{fmt(point.score_delta, 2)}
              </Tag>
            )}
            {point.change_reasons.slice(0, 2).map((reason) => (
              <Text key={`${point.as_of_date}-${reason.dimension}`} type="secondary" style={{ fontSize: 11 }}>
                {reason.reason}
              </Text>
            ))}
          </div>
        ))}
      </Space>
    </Card>
  );
}

function ExpandedDetail({ item }: { item: CtaProductScoreItem }): React.JSX.Element {
  return (
    <Space direction="vertical" size={8} style={{ width: "100%" }}>
      <ReturnsAnalysisCard productId={item.product_id} analysis={item.detail.returns_analysis} />
      <ComponentScoreCard title="收益分" score={item.detail.return_score} />
      <RiskExposureCard exposure={item.detail.risk_exposure} />
      <ComponentScoreCard title="风险暴露分" score={item.detail.risk_score} />
      <QualityDimensions item={item} />
      <ConfidenceCard confidence={item.detail.confidence} />
      <AttributionBlock attribution={item.detail.attribution} />
      <QualityHistoryBlock history={item.detail.quality_history} />
    </Space>
  );
}

interface Props {
  selectedProductIds?: string[];
  /** 对比栏选中的产品（上限两只），与对话用的 selectedProductIds 完全独立。 */
  compareIds?: string[];
  onCompareChange?: (ids: string[], names: Record<string, string>) => void;
  onOpenCompare?: () => void;
}

export default function CtaScorePanel({ selectedProductIds = [], compareIds = [], onCompareChange, onOpenCompare }: Props): React.JSX.Element {
  const [report, setReport] = useState<CtaProductScoreListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingSnapshot, setLoadingSnapshot] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [universeNote, setUniverseNote] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, CtaProductScoreItem>>({});
  const [loadingDetailIds, setLoadingDetailIds] = useState<ReadonlySet<string>>(new Set());
  const [expandedProductIds, setExpandedProductIds] = useState<string[]>([]);

  const runScores = useCallback(async (): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const universe = await buildCtaUniverse([], "评分");
      setUniverseNote(
        universe.excludedCount > 0
          ? `本次按${universe.frequency}频率评分，排除了${universe.excludedCount}只不同频率或净值不足的产品。`
          : `本次使用${universe.totalCount}只${universe.frequency}产品，全部来自可研究产品的已复核净值。`,
      );
      const snapshot = await persistCtaProductScoreSnapshot({
        products: universe.products,
        as_of_date: universe.asOfDate,
      });
      setReport(snapshot.report);
      setDetails({});
      setExpandedProductIds([]);
    } catch (cause) {
      const detail = cause instanceof Error ? cause.message : "CTA 产品评分生成失败";
      setError(detail);
      message.error(detail);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoadingSnapshot(true);
    void getLatestCtaProductScoreSnapshot()
      .then((snapshot) => {
        setReport(snapshot.report);
        setUniverseNote(`当前评分已固化：${snapshot.report.universe_size ?? snapshot.report.products.length}只${"周频"}产品，截至${snapshot.report.as_of_date}。`);
      })
      .catch(() => undefined)
      .finally(() => setLoadingSnapshot(false));
  }, []);

  const loadDetail = useCallback(async (productId: string): Promise<void> => {
    if (details[productId] || loadingDetailIds.has(productId)) return;
    setLoadingDetailIds((current) => new Set(current).add(productId));
    try {
      const result = await getLatestCtaProductScoreItem(productId);
      setDetails((current) => ({ ...current, [productId]: result.item }));
    } catch (cause) {
      message.error(cause instanceof Error ? cause.message : "加载产品画像失败");
    } finally {
      setLoadingDetailIds((current) => {
        const next = new Set(current);
        next.delete(productId);
        return next;
      });
    }
  }, [details, loadingDetailIds]);

  const visibleProducts = useMemo(
    () => report?.products.filter((item) => selectedProductIds.length === 0 || selectedProductIds.includes(item.product_id)) ?? [],
    [report, selectedProductIds],
  );

  const showProductProfile = useCallback((productId: string): void => {
    setExpandedProductIds((current) => current.includes(productId) ? current : [...current, productId]);
    void loadDetail(productId);
  }, [loadDetail]);

  const columns = useMemo(() => [
    {
      title: "产品",
      dataIndex: "product_name",
      ellipsis: true,
      render: (value: string, row: CtaProductScoreListItem) => (
        <div>
          <Space size={6}>
            <Text strong>{value}</Text>
            {row.summary.quality_status !== "eligible" && <Tag color="warning">样本不足</Tag>}
          </Space>
          <HeadlineLine item={row} />
        </div>
      ),
    },
    {
      title: "质量分",
      key: "quality",
      width: 84,
      render: (_value: unknown, row: CtaProductScoreListItem) => (
        row.summary.quality_score == null
          ? <Text type="secondary">—</Text>
          : <Text strong>{fmt(row.summary.quality_score, 1)}</Text>
      ),
    },
    {
      title: "置信分",
      key: "confidence",
      width: 96,
      render: (_value: unknown, row: CtaProductScoreListItem) => (
        <Space size={4}><Text>{fmt(row.summary.confidence_score, 1)}</Text>{bandTag(row.summary.confidence_band)}</Space>
      ),
    },
    {
      title: "归因质量分",
      key: "attribution_quality",
      width: 108,
      render: (_value: unknown, row: CtaProductScoreListItem) => (
        row.summary.attribution_quality_score == null
          ? <Text type="secondary">—</Text>
          : <Space size={4}><Text>{fmt(row.summary.attribution_quality_score, 1)}</Text>{bandTag(row.summary.attribution_quality_band)}</Space>
      ),
    },
    {
      title: "收益分",
      key: "return_score",
      width: 96,
      render: (_value: unknown, row: CtaProductScoreListItem) => (
        row.summary.return_score == null
          ? <Text type="secondary">—</Text>
          : <Space size={4}><Text>{fmt(row.summary.return_score, 1)}</Text>{bandTag(row.summary.return_band)}</Space>
      ),
    },
    {
      title: "风险暴露分",
      key: "risk_score",
      width: 104,
      render: (_value: unknown, row: CtaProductScoreListItem) => (
        row.summary.risk_score == null
          ? <Text type="secondary">—</Text>
          : <Space size={4}><Text>{fmt(row.summary.risk_score, 1)}</Text>{bandTag(row.summary.risk_band)}</Space>
      ),
    },
    {
      title: "画像",
      key: "profile",
      width: 90,
      render: (_value: unknown, row: CtaProductScoreListItem) => <Button size="small" type="link" onClick={() => showProductProfile(row.product_id)}>查看画像</Button>,
    },
  ], [showProductProfile]);

  return (
    <Card
      size="small"
      title={<Space size={8}><Text strong>CTA 产品评分与画像</Text><Tag color="blue">全量评分</Tag></Space>}
      extra={<Space size={8}>
        {onCompareChange && compareIds.length === 2 && <Button size="small" type="primary" ghost onClick={onOpenCompare}>对比两只产品</Button>}
        <Button size="small" type="primary" icon={<PlayCircleOutlined />} loading={loading} onClick={() => void runScores()}>更新全量评分</Button>
      </Space>}
      style={{ marginBottom: 14 }}
    >
      {universeNote && <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>{universeNote}</Text>}
      {error && <Alert type="error" showIcon message={error} closable onClose={() => setError(null)} style={{ marginBottom: 10 }} />}
      {selectedProductIds.length > 0 && report && <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>当前仅显示已选中的 {visibleProducts.length} 只产品。</Text>}
      {(loading || loadingSnapshot) && !report ? (
        <div style={{ padding: 18, textAlign: "center" }}><Spin tip="正在计算质量分、置信度与因子归因…" /></div>
      ) : report ? (
        <>
          <Space wrap size={12} style={{ marginBottom: 8 }}>
            <Text type="secondary">截至 {report.as_of_date}</Text>
            <Text type="secondary">评分模型 {report.quality_model_version}</Text>
            <Text type="secondary">报告版本 {report.model_version}</Text>
          </Space>
          <Table<CtaProductScoreListItem>
            rowKey="product_id"
            size="small"
            pagination={{ pageSize: 30, showSizeChanger: false }}
            columns={columns}
            dataSource={visibleProducts}
            scroll={{ x: 880 }}
            rowSelection={onCompareChange ? {
              selectedRowKeys: compareIds,
              onChange: (keys, rows) => {
                const ids = keys.map(String);
                if (ids.length > 2) {
                  message.warning("对比最多选择两只产品，请先取消一只。");
                  return;
                }
                const names: Record<string, string> = {};
                for (const row of rows) names[row.product_id] = row.product_name;
                onCompareChange(ids, names);
              },
            } : undefined}
            expandable={{
              expandedRowKeys: expandedProductIds,
              expandedRowRender: (row) => {
                const detail = details[row.product_id];
                return detail ? <ExpandedDetail item={detail} /> : <div style={{ padding: 12, textAlign: "center" }}><Spin size="small" /></div>;
              },
              rowExpandable: () => true,
              onExpand: (expanded, row) => {
                setExpandedProductIds((current) => expanded ? (current.includes(row.product_id) ? current : [...current, row.product_id]) : current.filter((id) => id !== row.product_id));
                if (expanded) void loadDetail(row.product_id);
              },
            }}
          />
          {report.warnings.length > 0 && (
            <Alert
              type="warning"
              showIcon
              message="模型状态"
              description={report.warnings.join("；")}
              style={{ marginTop: 10 }}
            />
          )}
        </>
      ) : (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无评分结果" />
      )}
    </Card>
  );
}
