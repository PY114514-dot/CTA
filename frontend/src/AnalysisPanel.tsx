import React, { Component, type ErrorInfo, type ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Empty,
  Progress,
  Row,
  Select,
  Space,
  Steps,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import {
  BarChartOutlined,
  ExperimentOutlined,
  FundProjectionScreenOutlined,
  PieChartOutlined,
  RadarChartOutlined,
  RobotOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import {
  generateAnalysisReport,
  getLlmConfig,
  kbGetNavSeries,
  kbListProducts,
  type DataFrequency,
  type DeepAttributionResponse,
  type FactorExposureItem,
  type KbProduct,
  type LlmConfigResponse,
  type NavPoint,
  type ReportGenerateResponse,
  type StrategyConfirmation,
} from "./api";
import type { AnalysisEngineConfig } from "./SettingsDrawer";
import { ACCENT_FALLBACK, cssVar } from "./ui/tokens";

const { Paragraph, Text, Title } = Typography;

function fixedOrDash(value: unknown, digits: number): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

/** A malformed optional chart must never take down the complete research page. */
class AnalysisResultBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {
    // The visible alert below is deliberate: report payload defects are actionable,
    // but should not blank the workspace.
  }

  render() {
    if (this.state.error) {
      return <Alert type="error" showIcon message="报告中有一项图表无法显示" description="其余研究数据未丢失。请重新分析；若问题持续，请反馈该产品。" />;
    }
    return this.props.children;
  }
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface AnalysisPanelProps {
  navPoints: NavPoint[];
  frequency: DataFrequency;
  productName?: string;
  /** OCR/VLM or user-supplied strategy disclosure passed as non-confirmatory context. */
  strategyHint?: string;
  /** Explicit confirmation that a flagged source point was manually checked. */
  qualityOverrideConfirmed?: boolean;
  engineConfig?: AnalysisEngineConfig;
  /** A saved report snapshot shown when a history record is reopened. */
  initialReport?: ReportGenerateResponse;
  /** Called with the full report whenever an analysis run completes. */
  onReport?: (report: ReportGenerateResponse) => void;
}

// ---------------------------------------------------------------------------
// Confidence tag color helper
// ---------------------------------------------------------------------------

function confidenceColor(label: string): string {
  if (label === "高") return "green";
  if (label === "中") return "orange";
  return "red";
}

function strategyTypeLabel(type: string): string {
  const map: Record<string, string> = {
    commodity_cta: "商品 CTA",
    equity_quant: "股票/股指方向（CTA或量化，待确认）",
    mixed: "混合型",
    insufficient_data: "数据不足",
  };
  return map[type] ?? type;
}

// ---------------------------------------------------------------------------
// Chart components
// ---------------------------------------------------------------------------

function FactorRadarChart({ data }: { data?: { indicators?: unknown; values?: unknown } }) {
  const indicators = Array.isArray(data?.indicators) ? data.indicators.filter((name): name is string => typeof name === "string") : [];
  const values = Array.isArray(data?.values) ? data.values.filter((value): value is number => typeof value === "number" && Number.isFinite(value)) : [];
  if (indicators.length === 0 || indicators.length !== values.length) return null;
  const option = {
    title: { text: "风格参考雷达图", left: "center", textStyle: { fontSize: 14 } },
    tooltip: {},
    radar: {
      indicator: indicators.map((name) => ({ name, max: 1.5, min: -1.5 })),
      shape: "polygon" as const,
      splitNumber: 4,
    },
    series: [
      {
        type: "radar",
        data: [{ value: values, name: "风格参考", areaStyle: { opacity: 0.2 } }],
        lineStyle: { width: 2 },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 300 }} />;
}

function VarietyBarChart({ data }: { data?: { names?: unknown; probabilities?: unknown; title?: unknown } }) {
  const names = Array.isArray(data?.names) ? data.names.filter((name): name is string => typeof name === "string") : [];
  const probabilities = Array.isArray(data?.probabilities) ? data.probabilities.filter((value): value is number => typeof value === "number" && Number.isFinite(value)) : [];
  if (names.length === 0 || names.length !== probabilities.length) return null;
  const option = {
    title: { text: typeof data?.title === "string" ? data.title : "品种概率排名", left: "center", textStyle: { fontSize: 14 } },
    tooltip: { trigger: "axis" as const, formatter: "{b}: {c}%" },
    grid: { left: 80, right: 30, top: 40, bottom: 30 },
    xAxis: { type: "value" as const, max: 100, axisLabel: { formatter: "{value}%" } },
    yAxis: { type: "category" as const, data: [...names].reverse(), axisLabel: { fontSize: 12 } },
    series: [
      {
        type: "bar",
        data: [...probabilities].reverse(),
        itemStyle: { color: cssVar("--serif-accent", ACCENT_FALLBACK), borderRadius: [0, 4, 4, 0] },
        barWidth: 20,
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 280 }} />;
}

function SectorPieChart({
  data,
  title = "板块相对暴露分布",
}: {
  data: { name: string; value: number; color: string }[];
  title?: string;
}) {
  const option = {
    title: { text: title, left: "center", textStyle: { fontSize: 14 } },
    tooltip: { trigger: "item" as const, formatter: "{b}<br/>原始候选分数：{c}%<br/>相对分布：{d}%" },
    legend: { bottom: 0, type: "scroll" as const },
    series: [
      {
        type: "pie",
        radius: ["35%", "65%"],
        center: ["50%", "48%"],
        data: data.map((item) => ({ name: item.name, value: item.value, itemStyle: { color: item.color } })),
        // Sector candidate scores can overlap and therefore do not sum to
        // 100%.  Pie labels must use ECharts' normalized percentage ({d}).
        label: { formatter: "{b}\n{d}%", fontSize: 11 },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.2)" } },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 300 }} />;
}

function RollingExposureChart({ data, labels = {} }: { data: Record<string, number[]>; labels?: Record<string, string> }) {
  const factors = Object.keys(data);
  const firstFactor = factors[0];
  const firstSeries = firstFactor === undefined ? undefined : data[firstFactor];
  if (!firstSeries) return null;
  const len = firstSeries.length;
  const xData = Array.from({ length: len }, (_, i) => `W${i + 1}`);
  const option = {
    title: { text: "滚动风格参考", left: "center", textStyle: { fontSize: 14 } },
    tooltip: { trigger: "axis" as const },
    legend: { bottom: 0, data: factors.map((factor) => labels[factor] ?? factor) },
    grid: { left: 50, right: 30, top: 40, bottom: 50 },
    xAxis: { type: "category" as const, data: xData },
    yAxis: { type: "value" as const },
    series: factors.map((factor) => ({
      name: labels[factor] ?? factor,
      type: "line",
      data: data[factor],
      smooth: true,
      lineStyle: { width: 2 },
      symbol: "none",
    })),
  };
  return <ReactECharts option={option} style={{ height: 280 }} />;
}

function FactorRiskContributionChart({
  factors,
  contributions,
}: {
  factors: FactorExposureItem[];
  contributions: Record<string, number | null>;
}) {
  const rows = factors
    .map((factor) => ({ label: factor.factor_label, value: contributions[factor.factor_name] }))
    .filter((row): row is { label: string; value: number } => typeof row.value === "number" && Number.isFinite(row.value));
  if (!rows.length) return null;
  const option = {
    title: { text: "已解释风险内的因子占比", left: "center", textStyle: { fontSize: 14 } },
    tooltip: { trigger: "axis" as const, valueFormatter: (value: number) => `${fixedOrDash(value, 1)}%` },
    grid: { left: 90, right: 30, top: 42, bottom: 28 },
    xAxis: { type: "value" as const, axisLabel: { formatter: (value: number) => `${value.toFixed(0)}%` } },
    yAxis: { type: "category" as const, data: rows.map((row) => row.label).reverse(), axisLabel: { fontSize: 11 } },
    series: [{ type: "bar" as const, data: rows.map((row) => row.value).reverse(), barMaxWidth: 22, itemStyle: { color: "#4f7de8", borderRadius: [0, 4, 4, 0] }, label: { show: true, position: "right" as const, formatter: (item: { value: number }) => `${fixedOrDash(item.value, 1)}%` } }],
  };
  return <ReactECharts option={option} style={{ height: 280 }} />;
}

function RollingR2Chart({ data }: { data?: number[] }) {
  const values = Array.isArray(data) ? data.filter((value): value is number => typeof value === "number" && Number.isFinite(value)) : [];
  if (!values.length) return null;
  const option = {
    title: { text: "滚动解释度（R²）", left: "center", textStyle: { fontSize: 14 } },
    tooltip: { trigger: "axis" as const, valueFormatter: (value: number) => fixedOrDash(value, 3) },
    grid: { left: 46, right: 24, top: 42, bottom: 34 },
    xAxis: { type: "category" as const, data: values.map((_, index) => `窗口 ${index + 1}`), axisLabel: { hideOverlap: true, fontSize: 10 } },
    yAxis: { type: "value" as const, min: 0, max: 1, axisLabel: { formatter: (value: number) => value.toFixed(1) } },
    series: [{ type: "line" as const, data: values, symbol: "none", smooth: true, lineStyle: { width: 2, color: "#7c3aed" }, areaStyle: { opacity: 0.08, color: "#7c3aed" } }],
  };
  return <ReactECharts option={option} style={{ height: 280 }} />;
}

function FactorRelationshipMatrix({
  names,
  matrix,
  labels,
  title,
  multiplier = 1,
}: {
  names: string[];
  matrix: Record<string, Record<string, number>>;
  labels: Record<string, string>;
  title: string;
  multiplier?: number;
}) {
  if (names.length < 2) return null;
  const shortNames = names.map((name) => labels[name] ?? name);
  const data = names.flatMap((row, rowIndex) => names.map((column, columnIndex) => [columnIndex, rowIndex, (matrix[row]?.[column] ?? 0) * multiplier]));
  const scale = Math.max(...data.map((item) => Math.abs(Number(item[2] ?? 0))), 0.000001);
  const option = {
    title: { text: title, left: "center", textStyle: { fontSize: 14 } },
    tooltip: { formatter: (item: { value: [number, number, number] }) => `${shortNames[item.value[1]]} / ${shortNames[item.value[0]]}<br/>${fixedOrDash(item.value[2], 3)}` },
    grid: { left: 82, right: 28, top: 50, bottom: 72 },
    xAxis: { type: "category" as const, data: shortNames, position: "top" as const, axisLabel: { rotate: 28, fontSize: 10, interval: 0 } },
    yAxis: { type: "category" as const, data: shortNames, axisLabel: { fontSize: 10 } },
    visualMap: { min: -scale, max: scale, calculable: false, orient: "horizontal" as const, left: "center", bottom: 0, inRange: { color: ["#2563eb", "#f8fafc", "#dc2626"] } },
    series: [{ type: "heatmap" as const, data, label: { show: true, fontSize: 9, formatter: (item: { value: [number, number, number] }) => fixedOrDash(item.value[2], 2) }, emphasis: { itemStyle: { shadowBlur: 8, shadowColor: "rgba(0,0,0,0.25)" } } }],
  };
  return <ReactECharts option={option} style={{ height: 320 }} />;
}

type ExplainabilityPoint = {
  date: string;
  factor_contributions: Record<string, number>;
};

function FactorExplainabilityCharts({
  data,
}: {
  data?: { series: ExplainabilityPoint[]; factor_labels: Record<string, string>; description?: string };
}): React.JSX.Element | null {
  if (!data?.series?.length) return null;
  const dates = data.series.map((point) => point.date);
  const contributionNames = Object.keys(data.factor_labels);
  const contributionSeries = contributionNames.map((name) => ({
    name: data.factor_labels[name] ?? name,
    type: "line" as const,
    data: data.series.map((point) => point.factor_contributions[name] ?? 0),
    symbol: "none",
    smooth: true,
  }));
  const contributionOption = {
    title: { text: "因子累计贡献（关联系数 × 因子收益）", left: "center", textStyle: { fontSize: 14 } },
    tooltip: { trigger: "axis" as const },
    legend: { bottom: 0, type: "scroll" as const },
    grid: { left: 54, right: 24, top: 42, bottom: 48 },
    xAxis: { type: "category" as const, data: dates, axisLabel: { hideOverlap: true, fontSize: 10 } },
    yAxis: { type: "value" as const, axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(1)}%` } },
    series: contributionSeries,
  };
  return (
    <div style={{ marginTop: 18 }}>
      <ReactECharts option={contributionOption} style={{ height: 300 }} />
      {data.description && <Text type="secondary" style={{ display: "block", fontSize: 11, marginTop: 4 }}>{data.description}</Text>}
    </div>
  );
}

function DeepAttributionSection({ data }: { data?: DeepAttributionResponse }): React.JSX.Element | null {
  if (!data) return null;
  const asset = data.asset_class;
  const diagnostics = data.diagnostics ?? {};
  const strategyRows = data.strategy_fingerprints.slice(0, 8);
  const sectorRows = data.sector_exposures.slice(0, 8);
  return (
    <Card size="small" title="④ 深度 CTA 归因：板块候选与策略行为指纹" style={{ marginBottom: 16 }}>
      <Alert
        type={diagnostics.reliability_label === "高" ? "success" : diagnostics.reliability_label === "不可识别" ? "warning" : "info"}
        showIcon
        message={`归因结论可信度：${diagnostics.reliability_label ?? "—"} · 计算稳定性：${diagnostics.statistical_stability_label ?? "—"} · 资产大类候选：${asset.most_likely_label ?? "不可识别"}（${fixedOrDash(asset.confidence_pct, 1)}%）`}
        description={data.disclaimer}
        style={{ marginBottom: 12 }}
      />
      <Descriptions size="small" column={{ xs: 1, sm: 3 }} style={{ marginBottom: 12 }}>
        <Descriptions.Item label="有效共同期数">{diagnostics.observation_count ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="线性样本外 R²">{fixedOrDash(diagnostics.linear_oos_r2, 3)}</Descriptions.Item>
        <Descriptions.Item label="非线性相对提升">{diagnostics.nonlinear_uplift === null || diagnostics.nonlinear_uplift === undefined ? "—" : fixedOrDash(diagnostics.nonlinear_uplift, 3)}</Descriptions.Item>
      </Descriptions>
      {sectorRows.length > 0 && (
        <>
          <Text strong style={{ display: "block", marginBottom: 6 }}>板块候选（不是实际仓位）</Text>
          <Text type="secondary" style={{ display: "block", fontSize: 11, marginBottom: 8 }}>
            相对候选权重由公开市场代理与产品收益的中位相关性（35%）、Elastic Net 标准化系数（25%）、时间切分非线性模型置换重要性（20%）和 Bootstrap 符号一致性（20%）合成后归一化；不是发生概率、仓位比例或收益归因。
          </Text>
          <Table
            size="small"
            pagination={false}
            rowKey="sector"
            dataSource={sectorRows}
            columns={[
              { title: "板块", dataIndex: "label" },
              { title: "相对候选权重", dataIndex: "candidate_probability_pct", render: (value: number) => `${fixedOrDash(value, 1)}%` },
              { title: "方向", dataIndex: "direction" },
              { title: "稳定性", dataIndex: "stability_pct", render: (value: number) => `${fixedOrDash(value, 0)}%` },
              { title: "证据", dataIndex: "evidence", render: (values: string[]) => <Text type="secondary" ellipsis={{ tooltip: values.join("；") }}>{values[0] ?? "—"}</Text> },
            ]}
          />
        </>
      )}
      {strategyRows.length > 0 && (
        <>
          <Divider style={{ margin: "14px 0 8px" }}>策略行为指纹</Divider>
          <Table
            size="small"
            pagination={false}
            rowKey="strategy"
            dataSource={strategyRows}
            columns={[
              { title: "行为", dataIndex: "label" },
              { title: "证据分数", dataIndex: "evidence_score_pct", render: (value: number) => `${fixedOrDash(value, 1)}%` },
              { title: "方向", dataIndex: "direction" },
              { title: "稳定性", dataIndex: "stability_pct", render: (value: number) => `${fixedOrDash(value, 0)}%` },
              { title: "状态", dataIndex: "status", render: (value: string) => <Tag color={value === "supported" ? "green" : value === "not_available" ? "default" : "orange"}>{value === "supported" ? "支持" : value === "not_available" ? "不可识别" : "弱/不稳定"}</Tag> },
            ]}
          />
        </>
      )}
      {data.state_analysis.length > 0 && (
        <Descriptions size="small" column={{ xs: 1, sm: 2 }} title="商品 CTA 市场状态表现（探索）" style={{ marginTop: 12 }}>
          {data.state_analysis.map((state) => (
            <Descriptions.Item key={state.state} label={state.state}>
              {state.periods} 期；平均收益 {fixedOrDash(state.product_mean_return * 100, 2)}%（较全样本 {state.relative_mean_return >= 0 ? "+" : ""}{fixedOrDash(state.relative_mean_return * 100, 2)}%）；正收益率 {fixedOrDash(state.product_positive_rate * 100, 1)}%
            </Descriptions.Item>
          ))}
        </Descriptions>
      )}
      {data.warnings.length > 0 && <Text type="secondary" style={{ display: "block", marginTop: 10, fontSize: 11 }}>{data.warnings.join("；")}</Text>}
    </Card>
  );
}

type PeerCurve = { id: string; name: string; points: Array<{ date: string; value: number }> };
type PeerExclusion = { name: string; reason: string };

function isComparableStrategy(strategy: string | null, strategyType: string | undefined): boolean {
  const text = (strategy ?? "").toLowerCase();
  if (strategyType === "commodity_cta") return /商品|期货|commodity|cta/.test(text);
  if (strategyType === "equity_quant") return /股指|股票|权益|equity/.test(text);
  if (strategyType === "mixed") return /混合|多资产|配置|mixed/.test(text);
  return false;
}

function PeerComparison({
  productName,
  navPoints,
  frequency,
  strategyType,
}: {
  productName?: string;
  navPoints: NavPoint[];
  frequency: DataFrequency;
  strategyType?: string;
}): React.JSX.Element | null {
  const [peers, setPeers] = useState<PeerCurve[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [peerExclusions, setPeerExclusions] = useState<PeerExclusion[]>([]);
  const [eligibleCount, setEligibleCount] = useState(0);

  useEffect(() => {
    if (!strategyType || strategyType === "insufficient_data") {
      setPeers([]);
      setSelectedIds([]);
      setPeerExclusions([]);
      setEligibleCount(0);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setLoadError("");
    void (async () => {
      try {
        const exclusions: PeerExclusion[] = [];
        const eligible = (await kbListProducts({ limit: 1000 }))
          .filter((product: KbProduct) => product.standard_name !== productName)
          .filter((product: KbProduct) => {
            if (product.research_workflow?.stage !== "research_ready") {
              exclusions.push({ name: product.standard_name, reason: product.research_workflow?.blocking_reasons[0] ?? "产品尚不可研究" });
              return false;
            }
            if (product.reviewed_nav_count < 10) {
              exclusions.push({ name: product.standard_name, reason: "已复核净值不足 10 个" });
              return false;
            }
            if (!isComparableStrategy(product.strategy, strategyType)) {
              exclusions.push({ name: product.standard_name, reason: "策略类型不匹配" });
              return false;
            }
            return true;
          });
        const products = eligible.slice(0, 8);
        for (const product of eligible.slice(8)) {
          exclusions.push({ name: product.standard_name, reason: "超过当前最多展示 8 个同类产品的上限" });
        }
        const curves = await Promise.all(products.map(async (product) => {
          const nav = (await kbGetNavSeries(product.id)).filter((point) => point.review_status === "reviewed" && Number.isFinite(point.nav));
          if (nav.length < 10 || !nav[0]?.nav) return null;
          const base = nav[0].nav;
          return {
            id: product.id,
            name: product.standard_name,
            points: nav.map((point) => ({ date: point.observation_date, value: point.nav / base })),
          } satisfies PeerCurve;
        }));
        if (cancelled) return;
        const available = curves.filter((curve): curve is PeerCurve => curve !== null);
        setPeers(available);
        setSelectedIds(available.slice(0, 3).map((curve) => curve.id));
        setEligibleCount(eligible.length);
        setPeerExclusions(exclusions);
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : "同类产品加载失败");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [productName, strategyType]);

  const currentBase = navPoints[0]?.net_asset_value;
  const currentPoints = currentBase
    ? navPoints.map((point) => ({ date: point.observation_date, value: point.net_asset_value / currentBase }))
    : [];
  const selectedPeers = peers.filter((curve) => selectedIds.includes(curve.id));
  const dates = Array.from(new Set([currentPoints, ...selectedPeers.map((curve) => curve.points)].flat().map((point) => point.date))).sort();
  const seriesFor = (points: Array<{ date: string; value: number }>) => {
    const byDate = new Map(points.map((point) => [point.date, point.value]));
    return dates.map((date) => byDate.get(date) ?? null);
  };
  const currentLabel = productName?.trim() || "当前产品";
  const synchronousComparison = useMemo(() => {
    if (currentPoints.length < 8 || selectedPeers.length === 0) return null;
    const currentDates = new Set(currentPoints.map((point) => point.date));
    const commonDates = selectedPeers.reduce(
      (shared, peer) => shared.filter((date) => peer.points.some((point) => point.date === date)),
      Array.from(currentDates),
    ).sort();
    if (commonDates.length < 8) return null;
    const valuesFor = (points: Array<{ date: string; value: number }>) => {
      const values = new Map(points.map((point) => [point.date, point.value]));
      return commonDates.map((date) => values.get(date) as number);
    };
    const pearson = (left: number[], right: number[]) => {
      if (left.length < 3 || left.length !== right.length) return null;
      const leftMean = left.reduce((sum, value) => sum + value, 0) / left.length;
      const rightMean = right.reduce((sum, value) => sum + value, 0) / right.length;
      const numerator = left.reduce((sum, value, index) => sum + (value - leftMean) * (right[index]! - rightMean), 0);
      const denominator = Math.sqrt(left.reduce((sum, value) => sum + (value - leftMean) ** 2, 0) * right.reduce((sum, value) => sum + (value - rightMean) ** 2, 0));
      return denominator > 1e-12 ? numerator / denominator : null;
    };
    const metrics = (values: number[]) => {
      const returns = values.slice(1).map((value, index) => value / values[index]! - 1);
      const periods = frequency === "daily" ? 252 : frequency === "monthly" ? 12 : 52;
      const mean = returns.reduce((sum, value) => sum + value, 0) / Math.max(returns.length, 1);
      const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(returns.length - 1, 1);
      let peak = values[0]!; let maximumDrawdown = 0;
      values.forEach((value) => { peak = Math.max(peak, value); maximumDrawdown = Math.min(maximumDrawdown, value / peak - 1); });
      return {
        annualizedReturn: (values.at(-1)! / values[0]!) ** (periods / Math.max(values.length - 1, 1)) - 1,
        annualizedVolatility: Math.sqrt(variance) * Math.sqrt(periods),
        maximumDrawdown,
        returns,
      };
    };
    const target = metrics(valuesFor(currentPoints));
    const peerRows = selectedPeers.map((peer) => {
      const peerMetrics = metrics(valuesFor(peer.points));
      const stressIndices = target.returns.map((value, index) => value < 0 && peerMetrics.returns[index]! < 0 ? index : -1).filter((index) => index >= 0);
      return {
        key: peer.id,
        name: peer.name,
        annualizedReturn: peerMetrics.annualizedReturn,
        annualizedVolatility: peerMetrics.annualizedVolatility,
        maximumDrawdown: peerMetrics.maximumDrawdown,
        returnCorrelation: pearson(target.returns, peerMetrics.returns),
        downsideCorrelation: pearson(stressIndices.map((index) => target.returns[index]!), stressIndices.map((index) => peerMetrics.returns[index]!)),
      };
    });
    const percentile = (targetValue: number, peerValues: number[], higherIsBetter = true) => {
      if (!peerValues.length) return null;
      const count = peerValues.filter((value) => higherIsBetter ? targetValue >= value : targetValue <= value).length;
      return count / (peerValues.length + 1) * 100;
    };
    return {
      commonDates,
      target,
      peerRows,
      returnPercentile: percentile(target.annualizedReturn, peerRows.map((row) => row.annualizedReturn)),
      volatilityPercentile: percentile(target.annualizedVolatility, peerRows.map((row) => row.annualizedVolatility), false),
      drawdownPercentile: percentile(target.maximumDrawdown, peerRows.map((row) => row.maximumDrawdown)),
    };
  }, [currentPoints, frequency, selectedPeers]);
  if (!strategyType || strategyType === "insufficient_data") return null;
  const option = {
    tooltip: { trigger: "axis" as const, valueFormatter: (value: number) => value ? `${((value - 1) * 100).toFixed(2)}%` : "—" },
    legend: { bottom: 0, type: "scroll" as const },
    grid: { left: 55, right: 22, top: 18, bottom: 52 },
    xAxis: { type: "category" as const, data: dates, axisLabel: { hideOverlap: true, fontSize: 10 } },
    yAxis: { type: "value" as const, scale: true, axisLabel: { formatter: (value: number) => `${((value - 1) * 100).toFixed(0)}%` } },
    series: [
      { name: currentLabel, type: "line" as const, data: seriesFor(currentPoints), symbol: "none", lineStyle: { width: 3, color: "#1677ff" } },
      ...selectedPeers.map((curve) => ({ name: curve.name, type: "line" as const, data: seriesFor(curve.points), symbol: "none", lineStyle: { width: 1.5 } })),
    ],
  };
  return (
    <div>
      {loading && <Text type="secondary" style={{ display: "block", marginBottom: 8 }}>正在加载已复核的同类产品…</Text>}
      {!loading && (
        <Alert
          type={peers.length > 0 ? "info" : "warning"}
          showIcon
          style={{ marginBottom: 8 }}
          message={`同类候选：${peers.length} 个可比较 / ${eligibleCount} 个符合初筛`}
          description={peerExclusions.length
            ? `未纳入：${peerExclusions.slice(0, 5).map((item) => `${item.name}（${item.reason}）`).join("；")}${peerExclusions.length > 5 ? `；另 ${peerExclusions.length - 5} 个` : ""}`
            : "筛选条件：已确认、已复核净值不少于 10 个、策略类型匹配；比较仅使用共同日期。"}
        />
      )}
      {peers.length > 0 && (
        <Select
          mode="multiple"
          size="small"
          value={selectedIds}
          onChange={setSelectedIds}
          options={peers.map((peer) => ({ value: peer.id, label: peer.name }))}
          placeholder="选择已复核的同类产品"
          style={{ width: "100%", marginBottom: 8 }}
        />
      )}
      {selectedPeers.length > 0 ? <ReactECharts option={option} style={{ height: 300 }} /> : (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={loadError || "没有足够的已复核同类产品可比较"} />
      )}
      {synchronousComparison && (
        <>
          <Descriptions size="small" column={{ xs: 1, sm: 3 }} title={`共同区间画像（${synchronousComparison.commonDates[0]} 至 ${synchronousComparison.commonDates.at(-1)}）`} style={{ marginTop: 12 }}>
            <Descriptions.Item label="当前年化收益">{fixedOrDash(synchronousComparison.target.annualizedReturn * 100, 2)}%（分位 {fixedOrDash(synchronousComparison.returnPercentile, 0)}）</Descriptions.Item>
            <Descriptions.Item label="当前年化波动">{fixedOrDash(synchronousComparison.target.annualizedVolatility * 100, 2)}%（低波分位 {fixedOrDash(synchronousComparison.volatilityPercentile, 0)}）</Descriptions.Item>
            <Descriptions.Item label="当前最大回撤">{fixedOrDash(synchronousComparison.target.maximumDrawdown * 100, 2)}%（较小回撤分位 {fixedOrDash(synchronousComparison.drawdownPercentile, 0)}）</Descriptions.Item>
          </Descriptions>
          <Table
            size="small"
            pagination={false}
            style={{ marginTop: 8 }}
            dataSource={synchronousComparison.peerRows}
            columns={[
              { title: "同类产品", dataIndex: "name", ellipsis: true },
              { title: "年化收益", dataIndex: "annualizedReturn", render: (value: number) => `${fixedOrDash(value * 100, 2)}%` },
              { title: "年化波动", dataIndex: "annualizedVolatility", render: (value: number) => `${fixedOrDash(value * 100, 2)}%` },
              { title: "最大回撤", dataIndex: "maximumDrawdown", render: (value: number) => `${fixedOrDash(value * 100, 2)}%` },
              { title: "收益相关", dataIndex: "returnCorrelation", render: (value: number | null) => fixedOrDash(value, 2) },
              { title: "共同下跌相关", dataIndex: "downsideCorrelation", render: (value: number | null) => fixedOrDash(value, 2) },
            ]}
          />
        </>
      )}
      <Text type="secondary" style={{ fontSize: 11, lineHeight: 1.5 }}>
        曲线均从各自首个已复核净值归一化为 1；更新当前产品净值后会即时重绘。表格只取全部所选产品的共同日期，分位仅在当前小样本内计算，不构成投资评级。
      </Text>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Analysis Panel
// ---------------------------------------------------------------------------

export default function AnalysisPanel({ navPoints, frequency, productName, strategyHint, qualityOverrideConfirmed = false, engineConfig, initialReport, onReport }: AnalysisPanelProps) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReportGenerateResponse | null>(initialReport ?? null);
  const [error, setError] = useState("");
  const [currentStep, setCurrentStep] = useState(-1);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [llmConfig, setLlmConfig] = useState<LlmConfigResponse | null>(null);
  const [strategyConfirmation, setStrategyConfirmation] = useState<StrategyConfirmation>("auto");

  // Load LLM config on mount
  useEffect(() => {
    void getLlmConfig().then(setLlmConfig).catch(() => {});
  }, []);

  // Reopening a history record should show its saved report immediately.  A
  // fresh run still replaces this snapshot through handleRunAnalysis.
  useEffect(() => {
    if (initialReport) setResult(initialReport);
  }, [initialReport]);

  const handleRunAnalysis = useCallback(async () => {
    if (navPoints.length < 10) {
      setError("至少需要 10 个净值数据点才能进行 AI 归因分析");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    setCurrentStep(0);
    setElapsedSeconds(0);

    // Elapsed time counter
    const startTime = Date.now();
    const elapsedTimer = setInterval(() => {
      setElapsedSeconds(Math.round((Date.now() - startTime) / 1000));
    }, 1000);

    // Simulate step progress (slower: 3s per step to match backend pace)
    const stepTimer = setInterval(() => {
      setCurrentStep((prev) => (prev < 3 ? prev + 1 : prev));
    }, 3000);

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 90000);
      const report = await generateAnalysisReport(
        navPoints,
        frequency,
        engineConfig?.dataSource ?? "api",
        engineConfig?.rollingWindow ?? 12,
        engineConfig?.topNVarieties ?? 8,
        strategyHint,
        strategyConfirmation,
        qualityOverrideConfirmed,
        controller.signal,
      );
      clearTimeout(timeout);
      setResult(report);
      onReport?.(report);
      setCurrentStep(4);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("分析超时（90秒），请检查网络连接或切换为模板引擎后重试");
      } else {
        setError(err instanceof Error ? err.message : "分析请求失败");
      }
      setCurrentStep(-1);
    } finally {
      clearInterval(stepTimer);
      clearInterval(elapsedTimer);
      setLoading(false);
    }
  }, [navPoints, frequency, strategyHint, strategyConfirmation, qualityOverrideConfirmed, engineConfig]);

  const hasData = navPoints.length >= 10;
  const classification = result?.structured?.classification;
  const factors = result?.structured?.factors;
  const factorRiskProfile = factors?.risk_profile;
  const varieties = result?.structured?.varieties;
  const deepAttribution = result?.structured?.deep_attribution;
  const externalFactors = result?.structured?.external_factors;
  const marketReference = result?.structured?.market_reference;
  const navProfile = result?.structured?.nav_profile;
  const userConfirmation = classification?.details?.user_confirmation as {
    confirmed?: boolean;
    requested_type?: string;
  } | undefined;
  const isExploratory = (navProfile?.point_count ?? 0) < 52;
  const factorLabels = useMemo(
    () => Object.fromEntries((factors?.factors ?? []).map((factor) => [factor.factor_name, factor.factor_label])),
    [factors],
  );

  return (
    <Card
      className="card-accent-top"
      title={
        <Space>
          <ExperimentOutlined />
          <span>AI 策略归因分析</span>
          {llmConfig?.enabled && <Tag color="blue" icon={<RobotOutlined />}>LLM: {llmConfig.provider || llmConfig.model}</Tag>}
        </Space>
      }
      style={{ marginTop: 24 }}
    >
      {/* Trigger area */}
      {!result && !loading && (
        <div style={{ textAlign: "center", padding: "24px 0" }}>
          <ThunderboltOutlined style={{ fontSize: 36, color: "var(--serif-accent)", marginBottom: 12 }} />
          <Paragraph type="secondary">
            基于已录入的 {navPoints.length} 个净值数据点，自动执行策略分类 → 风格参考 → 品种推断 → 生成报告四步流水线。
          </Paragraph>
          <div style={{ display: "flex", justifyContent: "center", alignItems: "flex-end", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
            <Space direction="vertical" size={4} style={{ width: 320 }}>
              <Text type="secondary" style={{ fontSize: 12, textAlign: "left" }}>
                策略范围确认（可选；不确认则只按统计结果路由）
              </Text>
              <Select<StrategyConfirmation>
                aria-label="策略范围确认"
                value={strategyConfirmation}
                onChange={setStrategyConfirmation}
                style={{ width: "100%", textAlign: "left" }}
                options={[
                  { value: "auto", label: "自动判定（推荐）" },
                  { value: "commodity_cta", label: "用户确认：商品 CTA" },
                  { value: "equity_cta", label: "用户确认：股指 CTA" },
                  { value: "mixed", label: "用户确认：混合 / 多资产" },
                ]}
              />
            </Space>
            <Button size="large" onClick={() => void handleRunAnalysis()} disabled={!hasData} icon={<FundProjectionScreenOutlined />}>
              一键 AI 归因分析
            </Button>
          </div>
          {!hasData && (
            <Paragraph type="warning" style={{ marginTop: 8, fontSize: 12 }}>
              请先在上方录入至少 10 个净值数据点
            </Paragraph>
          )}
        </div>
      )}

      {/* Loading with step progress */}
      {loading && (
        <div style={{ padding: "16px 0" }}>
          <Steps
            current={currentStep}
            status="process"
            items={[
              { title: "策略分类", description: "相关性 + 滚动稳定性" },
              { title: "风格参考", description: "历史回归分析" },
              { title: "品种推断", description: "LASSO 稀疏回归" },
              { title: "生成报告", description: llmConfig?.enabled ? "LLM 摘要" : "模板摘要" },
            ]}
          />
          <Progress percent={Math.min(95, (currentStep + 1) * 25)} status="active" style={{ marginTop: 16 }} />
          <div style={{ textAlign: "center", marginTop: 8 }}>
            <Text type="secondary">
              已用时 {elapsedSeconds}s
              {currentStep >= 3 && llmConfig?.enabled && " · 正在等待 LLM 响应，首次调用可能较慢…"}
              {currentStep >= 3 && !llmConfig?.enabled && " · 正在生成模板报告…"}
            </Text>
          </div>
        </div>
      )}

      {/* Error */}
      {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} closable onClose={() => setError("")} />}

      {/* Results */}
      <AnalysisResultBoundary>
      {result && (
        <div>
          {/* Warnings */}
          {result.warnings.length > 0 && (
            <Alert type="warning" showIcon message="数据质量与研究限制" description={result.warnings.join("；")} style={{ marginBottom: 16 }} closable />
          )}

          {result.method_provenance?.report_version && (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
              message={`本报告使用 ${result.method_provenance.report_version.point_count ?? "—"} 条 ${result.method_provenance.report_version.frequency ?? ""} 净值点`}
              description={`净值版本：${result.method_provenance.report_version.nav_fingerprint ?? "未记录"}。人工重新校准净值后，请重新生成报告；旧报告不应继续作为当前结论。`}
            />
          )}

          {externalFactors?.available && (
            <Alert
              type={externalFactors.verified ? "success" : "info"}
              showIcon
              style={{ marginBottom: 16 }}
              message={`外部因子参考：截至 ${externalFactors.latest_as_of_date ?? "未知"} · ${externalFactors.verified ? "已核验" : "待核验"}`}
              description={`覆盖 ${externalFactors.factor_count ?? 0} 个因子、${externalFactors.observation_count ?? 0} 条观测；${externalFactors.usable_for_regression ? "可用于回归" : "当前仅作为已保存的披露数据参考，未直接进入回归"}。`}
            />
          )}
          {externalFactors && !externalFactors.available && externalFactors.date_relation === "no_snapshot_on_or_before_product_end" && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message="外部因子参考未对齐，已自动排除"
              description={`产品截止日为 ${externalFactors.product_end ?? "未知"}；现有外部周报最早可用数据截至 ${externalFactors.latest_available_as_of_date ?? "未知"}，晚于该日期。请导入产品截止日前的周报后重新生成报告。`}
            />
          )}
          {marketReference?.available && (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
              message={`同期 AKShare 市场参考：${marketReference.start_date} 至 ${marketReference.end_date}`}
              description={`${Object.entries(marketReference.references).map(([name, value]) => `${name} ${value >= 0 ? "+" : ""}${value.toFixed(2)}`).join(" · ")}。${marketReference.note}`}
            />
          )}

          {navProfile && (
            <Card size="small" title="① 数据质量与研究边界" style={{ marginBottom: 16 }}>
              <Descriptions size="small" column={{ xs: 2, sm: 4 }}>
                <Descriptions.Item label="净值样本">{navProfile.point_count} 条 · {navProfile.frequency === "weekly" ? "周频" : navProfile.frequency === "monthly" ? "月频" : "日频"}</Descriptions.Item>
                <Descriptions.Item label="研究区间">{navProfile.start_date} 至 {navProfile.end_date}</Descriptions.Item>
                <Descriptions.Item label="累计收益">{fixedOrDash(navProfile.cumulative_return * 100, 2)}%</Descriptions.Item>
                <Descriptions.Item label="最大回撤">{fixedOrDash(navProfile.maximum_drawdown * 100, 2)}%</Descriptions.Item>
                <Descriptions.Item label="年化收益">{navProfile.annualized_return === null ? "—" : `${fixedOrDash(navProfile.annualized_return * 100, 2)}%`}</Descriptions.Item>
                <Descriptions.Item label="年化波动">{navProfile.annualized_volatility === null ? "—" : `${fixedOrDash(navProfile.annualized_volatility * 100, 2)}%`}</Descriptions.Item>
              </Descriptions>
              {isExploratory && <Alert type="warning" showIcon style={{ marginTop: 10 }} message={`当前只有 ${navProfile.point_count} 个${navProfile.frequency === "weekly" ? "周度" : ""}点：滚动回归、压力测试与同类分位仅作探索性参考`} />}
              <Text type="secondary" style={{ display: "block", fontSize: 12, marginTop: 8 }}>以上为净值序列直接计算；以下因子结果是历史统计代理，不代表真实持仓、未来收益或投资建议。</Text>
            </Card>
          )}

          {/* Natural language summary */}
          <Card size="small" style={{ marginBottom: 16, background: "var(--serif-muted)", borderColor: "var(--serif-border)" }}>
            <Space style={{ marginBottom: 8 }}>
              <RobotOutlined />
              <Text strong>分析摘要</Text>
              <Tag>{result.engine === "template" ? "模板引擎" : result.engine}</Tag>
            </Space>
            <Paragraph style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>{result.strategy_summary}</Paragraph>
          </Card>

          {result.method_provenance && (
            <Card size="small" title="本次结果的方法来源" style={{ marginBottom: 16 }}>
              <Space wrap size={[6, 6]}>
                {Object.entries(result.method_provenance).map(([key, item]) => (
                  <Tooltip key={key} title={item.detail ?? ""}>
                    <Tag color={key === "summary" && result.engine.startsWith("llm:") ? "blue" : "default"}>
                      {key === "nav_extraction" ? "净值识别" : key === "classification" ? "策略分类" : key === "factor_analysis" ? "因子分析" : "报告摘要"}：{item.method ?? "未记录"}
                    </Tag>
                  </Tooltip>
                ))}
              </Space>
            </Card>
          )}

          {/* Strategy is context, not the main result. */}
          {classification && (
            <Card size="small" title="研究范围与策略候选" style={{ marginBottom: 16 }}>
              <Descriptions column={{ xs: 1, sm: 3 }} size="small">
                <Descriptions.Item label="策略类型">
                  <Tag color={classification.strategy_type === "commodity_cta" ? "blue" : classification.strategy_type === "equity_quant" ? "purple" : "default"}>
                    {userConfirmation?.confirmed ? `已确认：${strategyTypeLabel(classification.strategy_type)}` : strategyTypeLabel(classification.strategy_type)}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="统计识别置信度">
                  <Progress percent={Math.round(classification.confidence_pct)} size="small" style={{ width: 120 }} />
                </Descriptions.Item>
                <Descriptions.Item label="统计结果">
                  <Tag color={confidenceColor(classification.confidence_label)}>仅凭净值：{classification.confidence_label}</Tag>
                </Descriptions.Item>
              </Descriptions>
              {userConfirmation?.confirmed && <Text type="secondary" style={{ display: "block", marginTop: 8, fontSize: 12 }}>研究范围已按你的确认执行；统计识别置信度仅衡量净值与公开市场代理的关系，不会推翻该确认。</Text>}
              {classification.strategy_type === "insufficient_data" && (
                <Alert
                  type="info"
                  showIcon
                  style={{ marginTop: 10 }}
                  message="尚未运行单一资产因子回归"
                  description={
                    <Space direction="vertical" size={8} style={{ width: "100%" }}>
                      <span>净值与市场指数的统计关系不足以自动判断产品属于商品或股指范围，因此本次只给出了市场背景相关性。若你了解产品策略，可确认研究范围后重新运行滚动回归、R²、风险贡献和压力测试。</span>
                      <Space wrap>
                        <Select<StrategyConfirmation>
                          aria-label="确认因子研究范围"
                          size="small"
                          value={strategyConfirmation}
                          onChange={setStrategyConfirmation}
                          style={{ width: 180 }}
                          options={[
                            { value: "auto", label: "请选择研究范围" },
                            { value: "commodity_cta", label: "确认：商品 CTA" },
                            { value: "equity_cta", label: "确认：股指 CTA" },
                            { value: "mixed", label: "确认：混合 / 多资产" },
                          ]}
                        />
                        <Button size="small" type="primary" disabled={strategyConfirmation === "auto"} loading={loading} onClick={() => void handleRunAnalysis()}>
                          按确认范围运行归因
                        </Button>
                      </Space>
                    </Space>
                  }
                />
              )}
              {(() => {
                const confirmation = classification.details?.user_confirmation as {
                  confirmed?: boolean;
                  requested_type?: string;
                  statistical_conflict?: boolean;
                  disclosure_conflict?: boolean;
                } | undefined;
                if (!confirmation?.confirmed) return null;
                const labels: Record<string, string> = {
                  commodity_cta: "商品 CTA",
                  equity_cta: "股指 CTA",
                  mixed: "混合 / 多资产",
                };
                return (
                  <Alert
                    type={confirmation.statistical_conflict || confirmation.disclosure_conflict ? "warning" : "success"}
                    showIcon
                    style={{ marginTop: 10 }}
                    message={`用户确认研究范围：${labels[confirmation.requested_type ?? ""] ?? confirmation.requested_type ?? "未指定"}`}
                    description="因子模型按用户确认范围运行；统计分类和披露差异仍保留在证据和报告中。"
                  />
                );
              })()}
              {(() => {
                const disclosure = classification.details?.disclosure_hint as {
                  label?: string;
                  status?: string;
                  conflict?: boolean;
                } | undefined;
                if (!disclosure?.label) return null;
                const statusLabel = disclosure.conflict
                  ? "与统计冲突"
                  : disclosure.status === "aligned"
                    ? "与统计方向一致"
                    : "披露候选";
                return (
                  <Alert
                    type={disclosure.conflict ? "warning" : "info"}
                    showIcon
                    style={{ marginTop: 10 }}
                    message={
                      <Space size={6} wrap>
                        <span>管理人披露：{disclosure.label}</span>
                        <Tag color={disclosure.conflict ? "warning" : "blue"}>{statusLabel}</Tag>
                      </Space>
                    }
                    description="披露文字只作为研究候选，不等同于实际持仓；冲突时已跳过单一资产因子模型。"
                  />
                );
              })()}
              {classification.evidence.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>依据：</Text>
                  {classification.evidence.map((e, i) => (
                    <Tag key={i} style={{ fontSize: 11, marginBottom: 4 }}>{e}</Tag>
                  ))}
                </div>
              )}
            </Card>
          )}

          {(result.charts_data.rolling_exposure && Object.keys(result.charts_data.rolling_exposure).length > 0 || result.charts_data.rolling_r_squared?.length) && (
            <Card size="small" title="② 滚动暴露与解释度" style={{ marginBottom: 16 }}>
              <Row gutter={[16, 12]}>
                <Col xs={24} lg={12}>
                  {result.charts_data.rolling_exposure && <RollingExposureChart data={result.charts_data.rolling_exposure} labels={factorLabels} />}
                </Col>
                <Col xs={24} lg={12}>
                  <RollingR2Chart data={result.charts_data.rolling_r_squared} />
                </Col>
              </Row>
              <Text type="secondary" style={{ display: "block", fontSize: 11, lineHeight: 1.5 }}>
                每个窗口只使用此前可对齐的历史收益计算；曲线用于观察风格是否稳定。{isExploratory ? "当前样本偏短，窗口数量有限，不能据此判断策略发生切换。" : "仍需结合净值复核和策略披露解读。"}
              </Text>
            </Card>
          )}

          {/* Historical factor relationships */}
          {factors && (
            <Card size="small" title="③ 历史因子关系与回报解释" style={{ marginBottom: 16 }}>
              {factors.collinearity && factors.collinearity.factor_names.length > 0 && (
                <>
                  <Alert
                    type={factors.collinearity.ridge_applied ? "warning" : "info"}
                    showIcon
                    style={{ marginBottom: 8 }}
                    message={factors.collinearity.ridge_applied ? "因子代理重叠较高：滚动趋势已稳定化" : "因子代理共线性诊断"}
                    description={
                      <Space direction="vertical" size={2}>
                        <span>
                          协方差矩阵条件数：{fixedOrDash(factors.collinearity.condition_number, 1)}；最大绝对相关性：{fixedOrDash(factors.collinearity.max_abs_correlation, 2)}。
                        </span>
                        {factors.collinearity.ridge_applied ? (
                          <span>
                            {factors.collinearity.high_correlation_pairs.length > 0
                              ? `重叠较高的代理：${factors.collinearity.high_correlation_pairs.join("；")}`
                              : "条件数偏高，多个因子难以单独区分。"}
                            分窗口曲线使用岭回归（参数 {fixedOrDash(factors.collinearity.ridge_alpha, 1)}）稳定化；表内全样本关联系数仍为普通回归参考。
                          </span>
                        ) : (
                          <span>当前代理之间未发现需要稳定化的强共线性；因子结果仍仅作风格参考。</span>
                        )}
                      </Space>
                    }
                  />
                  <Row gutter={[16, 12]} style={{ marginBottom: 10 }}>
                    <Col xs={24} xl={12}>
                      <FactorRelationshipMatrix
                        title="因子—因子相关矩阵"
                        names={factors.collinearity.factor_names}
                        matrix={factors.collinearity.correlation_matrix}
                        labels={factorLabels}
                      />
                    </Col>
                    <Col xs={24} xl={12}>
                      <FactorRelationshipMatrix
                        title="因子协方差（×10⁴）"
                        names={factors.collinearity.factor_names}
                        matrix={factors.collinearity.covariance_matrix}
                        labels={factorLabels}
                        multiplier={10000}
                      />
                    </Col>
                  </Row>
                </>
              )}
              <Row gutter={16}>
                <Col xs={24} md={12}>
                  {factors.factors.length > 0 ? (
                    <Table<FactorExposureItem>
                      dataSource={factors.factors}
                      rowKey="factor_name"
                      size="small"
                      pagination={false}
                      columns={[
                        { title: "因子", dataIndex: "factor_label", width: 110 },
                        { title: "关联系数", dataIndex: "exposure_beta", width: 90, render: (v: number) => fixedOrDash(v, 3) },
                        {
                          title: "下跌期关联系数",
                          width: 78,
                          render: (_: unknown, row: { factor_name: string }) => fixedOrDash(factorRiskProfile?.downside_betas[row.factor_name], 3),
                        },
                        {
                          title: "风险贡献",
                          width: 82,
                          render: (_: unknown, row: { factor_name: string }) => {
                            const value = factorRiskProfile?.variance_contributions_pct[row.factor_name];
                            return value === null || value === undefined ? "—" : `${fixedOrDash(value, 1)}%`;
                          },
                        },
                        { title: "t值", dataIndex: "t_statistic", width: 60, render: (v: number) => fixedOrDash(v, 2) },
                        {
                          title: "样本外方向",
                          dataIndex: "out_of_sample_hit_rate",
                          width: 110,
                          render: (v: number | null, row: { hit_rate_observations: number }) => (
                            v === null ? <Text type="secondary">数据不足</Text> : (
                              <span title="滚动训练期仅使用此前数据；衡量因子方向与当期产品收益的一致性，不是交易胜率。">
                                {fixedOrDash(v * 100, 1)}% <Text type="secondary">({row.hit_rate_observations}期)</Text>
                              </span>
                            )
                          ),
                        },
                        {
                          title: "置信",
                          dataIndex: "confidence_label",
                          width: 50,
                          render: (v: string) => <Tag color={confidenceColor(v)}>{v}</Tag>,
                        },
                      ]}
                    />
                  ) : (
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="未发现显著因子" />
                  )}
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary">R² = {fixedOrDash(factors.r_squared, 3)}，Adj R² = {fixedOrDash(factors.adj_r_squared, 3)}</Text>
                  </div>
                  <Text type="secondary" style={{ display: "block", fontSize: 11, marginTop: 4, lineHeight: 1.5 }}>
                    下跌期关联系数只在该因子下跌期计算；风险贡献来自因子之间的统计关系，不代表实际持仓比例或未来收益。
                  </Text>
                </Col>
                <Col xs={24} md={12}>
                  {factorRiskProfile && <FactorRiskContributionChart factors={factors.factors} contributions={factorRiskProfile.variance_contributions_pct} />}
                </Col>
              </Row>
              <FactorExplainabilityCharts
                data={result.charts_data.factor_explainability}
              />
              {factorRiskProfile && (
                <>
                  <Divider style={{ margin: "16px 0 10px" }}>风险贡献与压力测试</Divider>
                  <Text type="secondary" style={{ display: "block", fontSize: 11, marginBottom: 8 }}>
                    风险贡献来自回归因子与其协方差矩阵；下行结果只抽取因子下跌的历史期，均为风险画像而非持仓穿透或预测。
                  </Text>
                </>
              )}
              {factorRiskProfile && Object.keys(factorRiskProfile.regime_returns).length > 0 && (
                <Descriptions size="small" column={{ xs: 1, md: 2 }} title="压力情景参考" style={{ marginTop: 12 }}>
                  {factors.factors.map((factor) => {
                    const regime = factorRiskProfile.regime_returns[factor.factor_name];
                    if (!regime) return null;
                    return (
                      <Descriptions.Item key={factor.factor_name} label={`${factor.factor_label} 下跌期`}>
                        {regime.downside_periods} 期；产品平均收益 {regime.product_mean_return_when_factor_down === null
                          ? "—"
                          : `${fixedOrDash(regime.product_mean_return_when_factor_down * 100, 2)}%`}
                      </Descriptions.Item>
                    );
                  })}
                </Descriptions>
              )}
              {factorRiskProfile && factorRiskProfile.principal_components.length > 0 && (
                <Descriptions size="small" column={{ xs: 1, md: 2 }} title="主成分风险参考" style={{ marginTop: 12 }}>
                  {factorRiskProfile.principal_components.map((component) => (
                    <Descriptions.Item key={component.component} label={`共同风险 ${component.component}`}>
                      解释 {fixedOrDash(component.explained_variance_pct, 1)}% 共同波动；主要关联 {component.dominant_factors
                        .map((factor) => `${factor.factor_label} (${fixedOrDash(factor.loading, 2)})`)
                        .join("、") || "—"}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              )}
              {factorRiskProfile && (
                <Text type="secondary" style={{ display: "block", fontSize: 11, marginTop: 8, lineHeight: 1.5 }}>
                  主成分用于发现未预设的共同波动来源；载荷正负受数学方向影响，不能解读为看多、看空或实际仓位。
                </Text>
              )}
            </Card>
          )}

          <DeepAttributionSection data={deepAttribution} />

          {/* Strategy-scope market-reference inference */}
          {varieties && (
            <Card size="small" title={classification?.strategy_type === "equity_quant" ? "股指 CTA 市场线索（统计参考）" : "品种线索（仅商品 CTA 研究范围）"} style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col xs={24} md={12}>
                  {result.charts_data.variety_bar && <VarietyBarChart data={result.charts_data.variety_bar} />}
                </Col>
                <Col xs={24} md={12}>
                  {result.charts_data.sector_pie && (
                    <SectorPieChart
                      data={result.charts_data.sector_pie}
                      title={classification?.strategy_type === "equity_quant" ? "市场参考评分的相对分布" : "板块相对暴露分布"}
                    />
                  )}
                </Col>
              </Row>
              <Divider style={{ margin: "12px 0" }} />
              <Row gutter={16}>
                <Col xs={24} md={12}>
                  <Text type="secondary">方法稳定性（Jaccard）：{fixedOrDash(varieties.method_stability * 100, 0)}%</Text>
                </Col>
                <Col xs={24} md={12}>
                  {(() => {
                    const totalScore = varieties.sector_exposure.reduce(
                      (sum, sector) => sum + Math.max(0, sector.total_probability), 0,
                    );
                    return varieties.sector_exposure.map((sector) => (
                      <Tag key={sector.sector} color={sector.color} style={{ marginBottom: 4 }}>
                        {sector.sector} {fixedOrDash(totalScore > 0 ? sector.total_probability / totalScore * 100 : 0, 0)}%
                      </Tag>
                    ));
                  })()}
                </Col>
              </Row>
              <Text type="secondary" style={{ display: "block", marginTop: 10, fontSize: 11 }}>
                {classification?.strategy_type === "equity_quant"
                  ? "柱状图是相关性、LASSO 系数和分窗口入选率合成的候选评分；圆环图仅将各市场组的评分归一化为相对占比，不是实际持仓、方向或套保比例。"
                  : "品种候选评分不代表实际期货持仓、方向或仓位比例。"}
              </Text>
            </Card>
          )}

          {classification && (
            <Card size="small" title="④ 同类与共同区间比较" style={{ marginBottom: 16 }}>
              <PeerComparison
                productName={productName}
                navPoints={navPoints}
                frequency={frequency}
                strategyType={classification.strategy_type}
              />
            </Card>
          )}

          {/* Disclaimer + re-run */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
            <Text type="secondary" style={{ fontSize: 11 }}>{result.disclaimer}</Text>
            <Select<StrategyConfirmation>
              aria-label="重新分析时的策略范围"
              size="small"
              value={strategyConfirmation}
              onChange={setStrategyConfirmation}
              style={{ width: 170 }}
              options={[
                { value: "auto", label: "自动判定" },
                { value: "commodity_cta", label: "确认：商品 CTA" },
                { value: "equity_cta", label: "确认：股指 CTA" },
                { value: "mixed", label: "确认：混合 / 多资产" },
              ]}
            />
            <Button size="small" onClick={() => void handleRunAnalysis()} loading={loading}>
              重新分析
            </Button>
          </div>
        </div>
      )}
      </AnalysisResultBoundary>

    </Card>
  );
}
