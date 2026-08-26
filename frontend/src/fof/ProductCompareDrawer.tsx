/** 产品两两对比的加宽悬浮抽屉。
 *
 * 数据全部复用现有端点：kbGetProduct 取基础信息、getLatestCtaProductScoreItem
 * 取固化评分画像、kbGetNavSeries 取已复核净值并在前端自算共同区间相关性
 * （周频几百个点，毫秒级）。不加任何新后端。
 */

import React, { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Collapse, Drawer, Empty, Space, Spin, Table, Tag, Typography, message } from "antd";
import ReactECharts from "echarts-for-react";
import {
  getLatestCtaProductScoreItem,
  kbGetNavSeries,
  kbGetProduct,
  type CtaProductScoreItem,
  type KbNavPoint,
  type KbProductDetail,
} from "../api";
import { buildCompareSummary, type CompareProductProfile } from "./compareRules";
import { strategyLabel } from "./productDisplay";
import { deleteCompareSnapshot, loadCompareSnapshots, saveCompareSnapshot, type CompareSnapshotRecord } from "./compareStorage";

const { Text } = Typography;

interface SideData {
  product: KbProductDetail | null;
  score: CtaProductScoreItem | null;
  nav: KbNavPoint[];
}

interface PairStats {
  correlation: number | null;
  overlap: number;
  commonStart: string | null;
  commonEnd: string | null;
}

interface ProductCompareDrawerProps {
  open: boolean;
  productIds: string[];
  onClose: () => void;
  onToggleProduct: (productId: string, name?: string) => void;
  onOpenHistory?: () => void;
  /** 用已保存快照的两只产品替换当前对比栏并重新打开对比。 */
  onReplaceCompare?: (productIds: string[], names: Record<string, string>) => void;
}

function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

function fmtNum(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

/** 从已复核净值点构建 日期→净值 映射，用于共同区间对齐。 */
function navByDate(nav: KbNavPoint[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const point of nav) {
    if (point.review_status !== "reviewed") continue;
    if (!Number.isFinite(point.nav) || point.nav <= 0) continue;
    map.set(point.observation_date, point.nav);
  }
  return map;
}

/** 两只产品的共同区间相关性（基于共同日期的期间收益）。 */
function computePairStats(leftNav: KbNavPoint[], rightNav: KbNavPoint[]): PairStats {
  const leftMap = navByDate(leftNav);
  const rightMap = navByDate(rightNav);
  const dates = [...leftMap.keys()].filter((day) => rightMap.has(day)).sort();
  if (dates.length < 2) return { correlation: null, overlap: 0, commonStart: null, commonEnd: null };
  const leftReturns: number[] = [];
  const rightReturns: number[] = [];
  for (let index = 1; index < dates.length; index += 1) {
    const leftPrev = leftMap.get(dates[index - 1]!)!;
    const rightPrev = rightMap.get(dates[index - 1]!)!;
    const leftCurr = leftMap.get(dates[index]!)!;
    const rightCurr = rightMap.get(dates[index]!)!;
    leftReturns.push(leftCurr / leftPrev - 1);
    rightReturns.push(rightCurr / rightPrev - 1);
  }
  const overlap = leftReturns.length;
  const leftMean = leftReturns.reduce((sum, value) => sum + value, 0) / overlap;
  const rightMean = rightReturns.reduce((sum, value) => sum + value, 0) / overlap;
  const numerator = leftReturns.reduce((sum, value, index) => sum + (value - leftMean) * (rightReturns[index]! - rightMean), 0);
  const leftVar = leftReturns.reduce((sum, value) => sum + (value - leftMean) ** 2, 0);
  const rightVar = rightReturns.reduce((sum, value) => sum + (value - rightMean) ** 2, 0);
  const denominator = Math.sqrt(leftVar * rightVar);
  const correlation = denominator > 1e-12 ? numerator / denominator : null;
  return { correlation, overlap, commonStart: dates[0] ?? null, commonEnd: dates[dates.length - 1] ?? null };
}

function toProfile(side: SideData | undefined): CompareProductProfile | null {
  if (!side?.score) return null;
  const product = side.product;
  const headline = side.score.summary.headline;
  const factors = side.score.detail.attribution?.factors ?? [];
  const topFactor = factors
    .filter((factor) => Number.isFinite(factor.beta))
    .sort((left, right) => Math.abs(right.beta) - Math.abs(left.beta))[0];
  return {
    id: side.score.product_id,
    name: product?.standard_name ?? side.score.product_name,
    manager: product?.manager_name ?? null,
    strategy: product?.strategy ?? null,
    frequency: product?.nav_frequency ?? null,
    annualizedReturn: headline.annualized_return,
    maxDrawdown: headline.maximum_drawdown,
    annualizedVolatility: headline.annualized_volatility,
    alphaTStat: headline.alpha_t_stat,
    qualityScore: side.score.summary.quality_score,
    confidenceScore: side.score.summary.confidence_score,
    topFactorName: topFactor ? topFactor.display_name : null,
    topFactorBeta: topFactor ? topFactor.beta : null,
  };
}

interface MetricRow {
  key: string;
  label: string;
  left: number | null;
  right: number | null;
  /** 该指标「更好」的方向：higher=越大越好，lower=越小越好，neutral=不评判。 */
  direction: "higher" | "lower" | "neutral";
  /** 差值达到该量级才着色，避免噪声。 */
  significant: number;
  format: (value: number) => string;
}

function buildMetricRows(left: SideData | undefined, right: SideData | undefined): MetricRow[] {
  const leftSummary = left?.score?.summary;
  const rightSummary = right?.score?.summary;
  const leftHeadline = leftSummary?.headline;
  const rightHeadline = rightSummary?.headline;
  const pctRow = (label: string, leftValue: number | null, rightValue: number | null, direction: MetricRow["direction"], significant: number): MetricRow => ({
    key: label, label, left: leftValue, right: rightValue, direction, significant, format: (value) => fmtPct(value),
  });
  const scoreRow = (label: string, leftValue: number | null, rightValue: number | null, significant: number): MetricRow => ({
    key: label, label, left: leftValue, right: rightValue, direction: "higher", significant, format: (value) => fmtNum(value),
  });
  return [
    pctRow("年化收益", leftHeadline?.annualized_return ?? null, rightHeadline?.annualized_return ?? null, "higher", 0.02),
    pctRow("累计收益", leftHeadline?.cumulative_return ?? null, rightHeadline?.cumulative_return ?? null, "higher", 0.05),
    pctRow("最大回撤", leftHeadline?.maximum_drawdown ?? null, rightHeadline?.maximum_drawdown ?? null, "higher", 0.05),
    pctRow("年化波动", leftHeadline?.annualized_volatility ?? null, rightHeadline?.annualized_volatility ?? null, "lower", 0.05),
    pctRow("VaR95", leftHeadline?.var_95 ?? null, rightHeadline?.var_95 ?? null, "higher", 0.03),
    pctRow("年化 α", leftHeadline?.annualized_alpha ?? null, rightHeadline?.annualized_alpha ?? null, "neutral", Number.POSITIVE_INFINITY),
    scoreRow("质量分", leftSummary?.quality_score ?? null, rightSummary?.quality_score ?? null, 5),
    scoreRow("置信分", leftSummary?.confidence_score ?? null, rightSummary?.confidence_score ?? null, 5),
    scoreRow("归因质量分", leftSummary?.attribution_quality_score ?? null, rightSummary?.attribution_quality_score ?? null, 10),
    scoreRow("收益分", leftSummary?.return_score ?? null, rightSummary?.return_score ?? null, 10),
    scoreRow("风险暴露分", leftSummary?.risk_score ?? null, rightSummary?.risk_score ?? null, 10),
    scoreRow("R²", leftHeadline?.r_squared ?? null, rightHeadline?.r_squared ?? null, 0.05),
    scoreRow("样本外 R²", leftHeadline?.oos_r_squared ?? null, rightHeadline?.oos_r_squared ?? null, 0.05),
  ];
}

function metricDiff(row: MetricRow): { text: string; better: "left" | "right" | null } {
  if (row.left == null || row.right == null) return { text: "—", better: null };
  const delta = row.left - row.right;
  const text = `${delta >= 0 ? "+" : "-"}${row.format(Math.abs(delta))}`;
  if (row.direction === "neutral" || Math.abs(delta) < row.significant) return { text, better: null };
  const leftBetter = row.direction === "higher" ? delta > 0 : delta < 0;
  return { text, better: leftBetter ? "left" : "right" };
}

const WIN_COLOR = "#f5222d";

function SideHeader({ side, onSwap }: { side: SideData | undefined; onSwap: () => void }): React.JSX.Element {
  if (!side) return <div style={{ padding: 16, textAlign: "center" }}><Spin size="small" /></div>;
  if (!side.product || !side.score) {
    return <Space><Text type="secondary">产品画像加载失败</Text><Button size="small" type="text" danger onClick={onSwap}>换出</Button></Space>;
  }
  const product = side.product;
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
      <div style={{ minWidth: 0 }}>
        <Text strong style={{ fontSize: 14 }} ellipsis={{ tooltip: product.standard_name }}>{product.standard_name}</Text>
        <div style={{ marginTop: 4 }}>
          <Space size={6} wrap>
            {product.nav_frequency && <Tag>{product.nav_frequency === "weekly" ? "周频" : product.nav_frequency === "monthly" ? "月频" : "日频"}</Tag>}
            {product.strategy && <Tag color="blue">{strategyLabel(product.strategy) ?? product.strategy}</Tag>}
          </Space>
        </div>
        <Text type="secondary" style={{ display: "block", marginTop: 4, fontSize: 12 }}>{product.manager_name || "管理人未披露"} · 净值 {product.nav_count} 条</Text>
      </div>
      <Button size="small" type="text" danger onClick={onSwap}>换出</Button>
    </div>
  );
}

export default function ProductCompareDrawer({ open, productIds, onClose, onToggleProduct, onReplaceCompare }: ProductCompareDrawerProps): React.JSX.Element {
  const [sides, setSides] = useState<Record<string, SideData>>({});
  const [historyOpen, setHistoryOpen] = useState(false);
  const [snapshots, setSnapshots] = useState<CompareSnapshotRecord[]>([]);

  useEffect(() => {
    let active = true;
    setSides({});
    if (productIds.length === 0) return () => { active = false; };
    void Promise.all(productIds.map(async (id) => {
      const [product, scoreResult, nav] = await Promise.all([
        kbGetProduct(id).catch(() => null),
        getLatestCtaProductScoreItem(id).catch(() => null),
        kbGetNavSeries(id, true).catch(() => [] as KbNavPoint[]),
      ]);
      return [id, { product, score: scoreResult?.item ?? null, nav }] as const;
    })).then((results) => {
      if (!active) return;
      setSides(Object.fromEntries(results));
    });
    return () => { active = false; };
  }, [productIds.join(",")]);

  const leftId = productIds[0];
  const rightId = productIds[1];
  const left = leftId ? sides[leftId] : undefined;
  const right = rightId ? sides[rightId] : undefined;

  const pairStats = useMemo(() => computePairStats(left?.nav ?? [], right?.nav ?? []), [left?.nav, right?.nav]);

  const summary = useMemo(() => {
    const leftProfile = toProfile(left);
    const rightProfile = toProfile(right);
    if (!leftProfile || !rightProfile) return null;
    return buildCompareSummary(leftProfile, rightProfile, pairStats.correlation, pairStats.overlap);
  }, [left, right, pairStats]);

  const metricRows = useMemo(() => buildMetricRows(left, right), [left, right]);

  const leftFactors = left?.score?.detail.attribution?.factors ?? [];
  const rightFactors = right?.score?.detail.attribution?.factors ?? [];
  const factorRows = useMemo(() => {
    const rows = new Map<string, { name: string; displayName: string; leftBeta: number | null; leftT: number | null; rightBeta: number | null; rightT: number | null }>();
    const add = (factors: typeof leftFactors, side: "left" | "right") => {
      for (const factor of factors) {
        const row = rows.get(factor.name) ?? { name: factor.name, displayName: factor.display_name, leftBeta: null, leftT: null, rightBeta: null, rightT: null };
        if (side === "left") { row.leftBeta = factor.beta; row.leftT = factor.t_stat; } else { row.rightBeta = factor.beta; row.rightT = factor.t_stat; }
        rows.set(factor.name, row);
      }
    };
    add(leftFactors, "left");
    add(rightFactors, "right");
    return [...rows.values()];
  }, [leftFactors, rightFactors]);

  const chartOption = useMemo(() => {
    const leftMap = navByDate(left?.nav ?? []);
    const rightMap = navByDate(right?.nav ?? []);
    const dates = [...leftMap.keys()].filter((day) => rightMap.has(day)).sort();
    if (dates.length < 2) return null;
    const leftBase = leftMap.get(dates[0]!)!;
    const rightBase = rightMap.get(dates[0]!)!;
    const leftSeries: number[] = [];
    const rightSeries: number[] = [];
    const ratioSeries: (number | null)[] = [];
    for (const day of dates) {
      const leftNav = leftMap.get(day)! / leftBase;
      const rightNav = rightMap.get(day)! / rightBase;
      leftSeries.push(Number(leftNav.toFixed(4)));
      rightSeries.push(Number(rightNav.toFixed(4)));
      ratioSeries.push(rightNav > 1e-9 ? Number((leftNav / rightNav).toFixed(4)) : null);
    }
    const leftName = left?.product?.standard_name ?? leftId ?? "产品 A";
    const rightName = right?.product?.standard_name ?? rightId ?? "产品 B";
    return {
      grid: { left: 52, right: 52, top: 34, bottom: 40 },
      tooltip: { trigger: "axis" },
      legend: { data: [leftName, rightName, "相对强弱 A/B"], top: 0, textStyle: { fontSize: 11 } },
      xAxis: { type: "category" as const, data: dates, axisLabel: { hideOverlap: true, fontSize: 10 } },
      yAxis: [
        { type: "value" as const, name: "归一化净值", scale: true, nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 } },
        { type: "value" as const, name: "A/B", scale: true, splitLine: { show: false }, nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10 } },
      ],
      series: [
        { name: leftName, type: "line" as const, data: leftSeries, showSymbol: false, smooth: true, lineStyle: { color: "#1677ff", width: 2 } },
        { name: rightName, type: "line" as const, data: rightSeries, showSymbol: false, smooth: true, lineStyle: { color: "#fa8c16", width: 2 } },
        { name: "相对强弱 A/B", type: "line" as const, data: ratioSeries, showSymbol: false, smooth: true, yAxisIndex: 1, lineStyle: { color: "#8c8c8c", width: 1.5, type: "dashed" as const } },
      ],
    };
  }, [left, right, leftId, rightId]);

  const loading = productIds.length > 0 && (productIds.some((id) => !sides[id]));

  const snapshotMetrics = useMemo(() => metricRows.map((row) => {
    const diff = metricDiff(row);
    return {
      label: row.label,
      leftText: row.left == null ? "—" : row.format(row.left),
      rightText: row.right == null ? "—" : row.format(row.right),
      diffText: diff.text,
      better: diff.better,
    };
  }), [metricRows]);

  const handleSaveSnapshot = () => {
    if (!leftId || !rightId || !left?.score || !right?.score || !summary) {
      message.warning("两只产品的画像尚未加载完成，暂不能保存。");
      return;
    }
    const leftName = left.product?.standard_name ?? leftId;
    const rightName = right.product?.standard_name ?? rightId;
    const record: CompareSnapshotRecord = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      savedAt: new Date().toISOString(),
      leftId,
      leftName,
      rightId,
      rightName,
      correlation: pairStats.correlation,
      overlap: pairStats.overlap,
      correlationNote: summary.correlationNote ?? null,
      similarities: summary.similarities,
      differences: summary.differences,
      metrics: snapshotMetrics,
    };
    saveCompareSnapshot(record);
    message.success("对比结论已保存，可在「历史对比」中回看。");
  };

  const openHistory = () => {
    setSnapshots(loadCompareSnapshots());
    setHistoryOpen(true);
  };

  const handleDeleteSnapshot = (id: string) => {
    setSnapshots(deleteCompareSnapshot(id));
  };

  const handleReopenSnapshot = (record: CompareSnapshotRecord) => {
    onReplaceCompare?.([record.leftId, record.rightId], { [record.leftId]: record.leftName, [record.rightId]: record.rightName });
    setHistoryOpen(false);
  };

  return (
    <>
      <Drawer
      open={open}
      onClose={onClose}
      placement="right"
      width="min(94vw, 1040px)"
      maskClosable
      title={<Space size={8}><Text strong>产品对比</Text>{!loading && <Text type="secondary" style={{ fontSize: 12 }}>最多两只 · 数据来自已固化评分画像与已复核净值</Text>}</Space>}
      extra={<Space size={8}>
        <Button size="small" onClick={openHistory}>历史对比</Button>
        <Button size="small" type="primary" ghost disabled={loading || !summary} onClick={handleSaveSnapshot}>保存对比结论</Button>
      </Space>}
      styles={{
        mask: { background: "rgba(15, 23, 42, 0.32)" },
        content: {
          margin: 12,
          height: "calc(100% - 24px)",
          borderRadius: 16,
          overflow: "hidden",
          boxShadow: "0 12px 40px rgba(0, 0, 0, 0.22)",
          background: "var(--serif-card)",
        },
      }}
    >
      {productIds.length < 2 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="请先在产品画像表格勾选两只产品，或在产品详情中「加入对比」。" />
      ) : (
        <Space direction="vertical" size={12} style={{ width: "100%" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Card size="small"><SideHeader side={left} onSwap={() => leftId && onToggleProduct(leftId)} /></Card>
            <Card size="small"><SideHeader side={right} onSwap={() => rightId && onToggleProduct(rightId)} /></Card>
          </div>

          <Card size="small" title="相同点与不同点">
            {loading || !summary ? (
              <div style={{ padding: 12, textAlign: "center" }}><Spin size="small" /></div>
            ) : (
              <Space direction="vertical" size={8} style={{ width: "100%" }}>
                <div>
                  <Text strong style={{ fontSize: 12 }}>相同点</Text>
                  <div style={{ marginTop: 4 }}>
                    <Space size={6} wrap>
                      {summary.similarities.length ? summary.similarities.map((item) => <Tag key={item} color="green">{item}</Tag>) : <Text type="secondary" style={{ fontSize: 12 }}>未发现显著共同特征</Text>}
                    </Space>
                  </div>
                </div>
                <div>
                  <Text strong style={{ fontSize: 12 }}>不同点</Text>
                  <div style={{ marginTop: 4 }}>
                    <Space size={6} wrap>
                      {summary.differences.length ? summary.differences.map((item) => <Tag key={item} color="orange">{item}</Tag>) : <Text type="secondary" style={{ fontSize: 12 }}>未发现显著差异</Text>}
                    </Space>
                  </div>
                </div>
                {summary.correlationNote && <Text type="secondary" style={{ fontSize: 12 }}>{summary.correlationNote}</Text>}
              </Space>
            )}
          </Card>

          <Card size="small" title="指标对比">
            <Table<MetricRow>
              rowKey="key"
              size="small"
              pagination={false}
              dataSource={metricRows}
              columns={[
                { title: "指标", dataIndex: "label", width: 120 },
                {
                  title: left?.product?.standard_name ?? "产品 A",
                  dataIndex: "left",
                  width: 140,
                  render: (value: number | null, row) => {
                    if (value == null) return <Text type="secondary">—</Text>;
                    const better = metricDiff(row).better;
                    return <Text style={{ color: better === "left" ? WIN_COLOR : undefined }}>{row.format(value)}</Text>;
                  },
                },
                {
                  title: right?.product?.standard_name ?? "产品 B",
                  dataIndex: "right",
                  width: 140,
                  render: (value: number | null, row) => {
                    if (value == null) return <Text type="secondary">—</Text>;
                    const better = metricDiff(row).better;
                    return <Text style={{ color: better === "right" ? WIN_COLOR : undefined }}>{row.format(value)}</Text>;
                  },
                },
                {
                  title: "差值",
                  key: "diff",
                  width: 120,
                  render: (_value, row) => <Text>{metricDiff(row).text}</Text>,
                },
              ]}
            />
            <Text type="secondary" style={{ display: "block", marginTop: 6, fontSize: 12 }}>差值列以「产品 A − 产品 B」计算；红色代表该项更优的一侧，差值未达显著门槛不着色。</Text>
          </Card>

          <Card size="small" title="净值走势叠加（共同区间归一化）">
            {chartOption ? (
              <>
                <Text type="secondary" style={{ display: "block", marginBottom: 6, fontSize: 12 }}>
                  共同区间 {pairStats.commonStart} ~ {pairStats.commonEnd} · 共同收益期数 {pairStats.overlap} · 相关性 {pairStats.correlation == null ? "—" : pairStats.correlation.toFixed(2)}
                </Text>
                <ReactECharts option={chartOption} style={{ height: 240 }} />
              </>
            ) : (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="共同历史不足，无法叠加净值" />
            )}
          </Card>

          <Card size="small" title="因子暴露并排">
            {factorRows.length === 0 ? (
              <Alert type="warning" showIcon message="因子归因暂不可用" description="两只产品中至少一只没有可用的因子归因明细。" />
            ) : (
              <Table
                rowKey="name"
                size="small"
                pagination={false}
                dataSource={factorRows}
                scroll={{ x: 560 }}
                columns={[
                  { title: "因子", dataIndex: "displayName", ellipsis: true },
                  { title: left?.product?.standard_name ?? "产品 A", key: "leftBeta", width: 130, render: (_value, row) => row.leftBeta == null ? "—" : `${row.leftBeta.toFixed(3)}（t ${fmtNum(row.leftT, 1)}）` },
                  { title: right?.product?.standard_name ?? "产品 B", key: "rightBeta", width: 130, render: (_value, row) => row.rightBeta == null ? "—" : `${row.rightBeta.toFixed(3)}（t ${fmtNum(row.rightT, 1)}）` },
                  {
                    title: "方向",
                    key: "direction",
                    width: 96,
                    render: (_value, row) => {
                      if (row.leftBeta == null || row.rightBeta == null) return <Text type="secondary">单边</Text>;
                      return row.leftBeta * row.rightBeta < 0 ? <Tag color="red">方向相反</Tag> : <Tag color="green">同向</Tag>;
                    },
                  },
                ]}
              />
            )}
          </Card>
        </Space>
      )}
    </Drawer>
      <Drawer
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        placement="right"
        width={560}
        maskClosable
        title="历史对比"
        styles={{
          mask: { background: "rgba(15, 23, 42, 0.32)" },
          content: {
            margin: 12,
            height: "calc(100% - 24px)",
            borderRadius: 16,
            overflow: "hidden",
            boxShadow: "0 12px 40px rgba(0, 0, 0, 0.22)",
            background: "var(--serif-card)",
          },
        }}
      >
        {snapshots.length === 0 ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无保存的对比结论。完成一次两只产品对比后，点击「保存对比结论」即可留存。" />
        ) : (
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            {snapshots.map((record) => (
              <Card
                key={record.id}
                size="small"
                title={<Space size={6} wrap><Text strong style={{ fontSize: 13 }}>{record.leftName}</Text><Text type="secondary">vs</Text><Text strong style={{ fontSize: 13 }}>{record.rightName}</Text></Space>}
                extra={<Space size={4}>
                  <Button size="small" type="primary" ghost onClick={() => handleReopenSnapshot(record)}>重新对比</Button>
                  <Button size="small" danger type="text" onClick={() => handleDeleteSnapshot(record.id)}>删除</Button>
                </Space>}
              >
                <Text type="secondary" style={{ fontSize: 12 }}>
                  保存于 {new Date(record.savedAt).toLocaleString("zh-CN")} · 共同收益期数 {record.overlap} · 相关性 {record.correlation == null ? "—" : record.correlation.toFixed(2)}
                </Text>
                {record.correlationNote && <Text type="secondary" style={{ display: "block", fontSize: 12 }}>{record.correlationNote}</Text>}
                {(record.similarities.length > 0 || record.differences.length > 0) && (
                  <div style={{ marginTop: 8 }}>
                    {record.similarities.length > 0 && <div><Text strong style={{ fontSize: 12 }}>相同点</Text><div style={{ marginTop: 2 }}><Space size={4} wrap>{record.similarities.map((item) => <Tag key={item} color="green" style={{ fontSize: 11 }}>{item}</Tag>)}</Space></div></div>}
                    {record.differences.length > 0 && <div style={{ marginTop: 6 }}><Text strong style={{ fontSize: 12 }}>不同点</Text><div style={{ marginTop: 2 }}><Space size={4} wrap>{record.differences.map((item) => <Tag key={item} color="orange" style={{ fontSize: 11 }}>{item}</Tag>)}</Space></div></div>}
                  </div>
                )}
                <Collapse
                  size="small"
                  ghost
                  style={{ marginTop: 8 }}
                  items={[{
                    key: record.id,
                    label: "关键指标快照",
                    children: (
                      <Table
                        rowKey="label"
                        size="small"
                        pagination={false}
                        dataSource={record.metrics}
                        columns={[
                          { title: "指标", dataIndex: "label", width: 110 },
                          { title: record.leftName, dataIndex: "leftText", width: 110, render: (value: string, row) => <Text style={{ color: row.better === "left" ? WIN_COLOR : undefined }}>{value}</Text> },
                          { title: record.rightName, dataIndex: "rightText", width: 110, render: (value: string, row) => <Text style={{ color: row.better === "right" ? WIN_COLOR : undefined }}>{value}</Text> },
                          { title: "差值", dataIndex: "diffText", width: 100 },
                        ]}
                      />
                    ),
                  }]}
                />
              </Card>
            ))}
          </Space>
        )}
      </Drawer>
    </>
  );
}
