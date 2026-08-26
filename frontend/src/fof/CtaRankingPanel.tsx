/** Read-only CODEX CTA cross-sectional ranking panel. */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Descriptions, Empty, List, Modal, Popconfirm, Space, Spin, Table, Tag, Typography, message } from "antd";
import { DeleteOutlined, PlayCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import {
  deleteCtaRankingSnapshot,
  getCtaRankingSnapshot,
  analyzeNav,
  kbGetNavSeries,
  kbGetProduct,
  kbGetReviewPage,
  kbGetSourcePreview,
  listCtaRankingSnapshots,
  persistCtaRankingSnapshot,
  type CtaRankingProductResult,
  type CtaRankingRequestPayload,
  type CtaRankingResponse,
  type CtaRankingSnapshotSummary,
  type PerformanceMetrics,
} from "../api";
import { getLatestCtaPhaseDEvidence } from "./ctaAttributionApi";
import { buildCtaUniverse } from "./ctaUniverse";

const { Text } = Typography;

const DIMENSION_LABELS: Record<string, string> = {
  absolute: "绝对收益",
  risk_adjusted: "风险调整",
  trend_regime: "趋势状态",
  tail: "极端环境",
  robustness: "稳健性",
};

function formatScore(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(1);
}

function formatPercent(value: number | null | undefined): string {
  return value == null ? "—" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

interface Props {
  selectedProductIds?: string[];
}

function ProductNavModal({ productId, open, onClose }: { productId: string | null; open: boolean; onClose: () => void }): React.JSX.Element {
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [nav, setNav] = useState<Array<{ date: string; value: number }>>([]);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [sourceImage, setSourceImage] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !productId) return;
    let active = true;
    setLoading(true);
    setName("");
    setNav([]);
    setMetrics(null);
    setSourceImage(null);
    void (async () => {
      try {
        const [product, points] = await Promise.all([kbGetProduct(productId), kbGetNavSeries(productId, true)]);
        if (!active) return;
        setName(product.standard_name);
        const reviewedPoints = points
          .filter((point) => Number.isFinite(point.nav) && point.nav > 0)
          .sort((a, b) => a.observation_date.localeCompare(b.observation_date));
        setNav(reviewedPoints.map((point) => ({ date: point.observation_date, value: point.nav })));
        if (reviewedPoints.length >= 2) {
          const frequency = product.nav_frequency === "daily" || product.nav_frequency === "monthly" ? product.nav_frequency : "weekly";
          const analysis = await analyzeNav(reviewedPoints.map((point) => ({ observation_date: point.observation_date, net_asset_value: point.nav })), frequency, 0.015);
          if (!active) return;
          setMetrics(analysis.metrics);
        }
        for (const fileId of product.source_file_ids ?? []) {
          try {
            const location = await kbGetReviewPage(fileId, productId);
            const preview = await kbGetSourcePreview(fileId, Math.max((location.page_number ?? 1) - 1, 0));
            if (preview.type.startsWith("image/")) {
              const url = URL.createObjectURL(preview);
              if (active) setSourceImage(url); else URL.revokeObjectURL(url);
              break;
            }
          } catch {
            // SQLite and other non-image sources use the reviewed NAV chart below.
          }
        }
      } catch (cause) {
        if (active) message.error(cause instanceof Error ? cause.message : "读取产品净值失败");
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => { active = false; };
  }, [open, productId]);

  useEffect(() => () => { if (sourceImage) URL.revokeObjectURL(sourceImage); }, [sourceImage]);

  return <Modal title={name ? `${name} · 产品净值` : "产品净值"} open={open} onCancel={onClose} footer={null} width={860} destroyOnHidden>
    {loading ? <div style={{ padding: 48, textAlign: "center" }}><Spin /></div> : sourceImage ? (
      <>
        {nav.length >= 2 && <Descriptions size="small" column={3} style={{ marginBottom: 12 }}>
          <Descriptions.Item label="净值点数">{nav.length}</Descriptions.Item>
          <Descriptions.Item label="起始日期">{nav.at(0)?.date ?? "—"}</Descriptions.Item>
          <Descriptions.Item label="截止日期">{nav.at(-1)?.date}</Descriptions.Item>
          <Descriptions.Item label="区间收益">{formatPercent(metrics?.cumulative_return)}</Descriptions.Item>
          <Descriptions.Item label="年化收益">{formatPercent(metrics?.annualized_return)}</Descriptions.Item>
          <Descriptions.Item label="最大回撤">{formatPercent(metrics?.maximum_drawdown)}</Descriptions.Item>
          <Descriptions.Item label="年化波动率">{formatPercent(metrics?.annualized_volatility)}</Descriptions.Item>
          <Descriptions.Item label="夏普比率">{metrics?.sharpe_ratio?.toFixed(2) ?? "—"}</Descriptions.Item>
        </Descriptions>}
        <img src={sourceImage} alt={`${name} 原始资料`} style={{ display: "block", width: "100%", maxHeight: "70vh", objectFit: "contain" }} onError={() => setSourceImage(null)} />
      </>
    ) : nav.length >= 2 ? <>
      <Descriptions size="small" column={3} style={{ marginBottom: 12 }}>
        <Descriptions.Item label="净值点数">{nav.length}</Descriptions.Item>
        <Descriptions.Item label="起始日期">{nav.at(0)?.date ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="截止日期">{nav.at(-1)?.date}</Descriptions.Item>
        <Descriptions.Item label="区间收益">{formatPercent(metrics?.cumulative_return)}</Descriptions.Item>
        <Descriptions.Item label="年化收益">{formatPercent(metrics?.annualized_return)}</Descriptions.Item>
        <Descriptions.Item label="最大回撤">{formatPercent(metrics?.maximum_drawdown)}</Descriptions.Item>
        <Descriptions.Item label="年化波动率">{formatPercent(metrics?.annualized_volatility)}</Descriptions.Item>
        <Descriptions.Item label="夏普比率">{metrics?.sharpe_ratio?.toFixed(2) ?? "—"}</Descriptions.Item>
      </Descriptions>
      <ReactECharts
        option={{
          grid: { left: 58, right: 24, top: 22, bottom: 42 },
          tooltip: { trigger: "axis", valueFormatter: (value: number | string) => Number(value).toFixed(4) },
          xAxis: { type: "category", data: nav.map((point) => point.date), axisLabel: { hideOverlap: true } },
          yAxis: { type: "value", scale: true, name: "单位净值" },
          series: [{ type: "line", data: nav.map((point) => point.value), showSymbol: false, smooth: true, lineStyle: { color: "#1677ff", width: 2 }, areaStyle: { color: "rgba(22,119,255,.10)" } }],
        }}
        style={{ height: 420 }}
      />
    </> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无可展示的净值数据" />}
  </Modal>;
}

export default function CtaRankingPanel({ selectedProductIds = [] }: Props): React.JSX.Element {
  const [ranking, setRanking] = useState<CtaRankingResponse | null>(null);
  const [snapshots, setSnapshots] = useState<CtaRankingSnapshotSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [snapshotLoadingId, setSnapshotLoadingId] = useState<string | null>(null);
  const [snapshotDeletingId, setSnapshotDeletingId] = useState<string | null>(null);
  const [activeSnapshotId, setActiveSnapshotId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [universeNote, setUniverseNote] = useState<string | null>(null);
  const [previewProductId, setPreviewProductId] = useState<string | null>(null);

  const refreshHistory = useCallback(async (): Promise<CtaRankingSnapshotSummary[]> => {
    setHistoryLoading(true);
    try {
      const history = await listCtaRankingSnapshots(12);
      setSnapshots(history);
      return history;
    } catch {
      // History is supplementary; ranking remains usable if the table is empty.
      return [];
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => { void refreshHistory(); }, [refreshHistory]);

  const loadSnapshot = useCallback(async (snapshotId: string): Promise<void> => {
    setSnapshotLoadingId(snapshotId);
    setError(null);
    try {
      const snapshot = await getCtaRankingSnapshot(snapshotId);
      setRanking(snapshot.ranking);
      setActiveSnapshotId(snapshot.snapshot_id);
      setUniverseNote(`已加载 ${snapshot.ranking.as_of_date} 的历史排名快照；重新运行本周排名可回到最新数据。`);
    } catch (cause) {
      setRanking(null);
      setActiveSnapshotId(null);
      const fallback = (await refreshHistory()).find((item) => item.snapshot_id !== snapshotId);
      if (fallback) {
        try {
          const snapshot = await getCtaRankingSnapshot(fallback.snapshot_id);
          setRanking(snapshot.ranking);
          setActiveSnapshotId(snapshot.snapshot_id);
          setUniverseNote(`所选历史快照已过期，已恢复 ${snapshot.ranking.as_of_date} 的最近可用排名快照。`);
          message.warning("所选历史快照已过期，已恢复最近可用快照");
          return;
        } catch {
          // The refreshed list is authoritative; show a normal empty state if it races another deletion.
        }
      }
      const detail = cause instanceof Error ? cause.message : "加载历史排名快照失败";
      setError(fallback ? "历史快照暂时不可用，请刷新后重试" : "历史快照已过期，请运行本周排名生成当前结果");
      message.error(detail);
    } finally {
      setSnapshotLoadingId(null);
    }
  }, [refreshHistory]);

  const deleteSnapshot = useCallback(async (snapshotId: string): Promise<void> => {
    setSnapshotDeletingId(snapshotId);
    try {
      await deleteCtaRankingSnapshot(snapshotId);
      if (activeSnapshotId === snapshotId) {
        setRanking(null);
        setActiveSnapshotId(null);
        setUniverseNote("当前历史排名快照已删除；可选择其他历史快照或重新运行本周排名。");
      }
      await refreshHistory();
      message.success("历史排名快照已删除");
    } catch (cause) {
      message.error(cause instanceof Error ? cause.message : "删除历史排名快照失败");
    } finally {
      setSnapshotDeletingId(null);
    }
  }, [activeSnapshotId, refreshHistory]);

  const buildRequest = useCallback(async (): Promise<CtaRankingRequestPayload> => {
    const universe = await buildCtaUniverse(selectedProductIds, "排名");
    setUniverseNote(
      universe.excludedCount > 0
        ? `本次按${universe.frequency}频率排名，排除了${universe.excludedCount}只不同频率或净值不足的产品。`
        : `本次使用${universe.totalCount}只${universe.frequency}产品，全部来自可研究产品的已复核净值。`,
    );
    const attributionEvidence = await getLatestCtaPhaseDEvidence(
      universe.products.map((item) => item.product_id),
      universe.asOfDate,
    ).catch(() => ({}));
    return {
      products: universe.products,
      as_of_date: universe.asOfDate,
      attribution_evidence: attributionEvidence,
    };
  }, [selectedProductIds]);

  const runRanking = useCallback(async (): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const request = await buildRequest();
      const snapshot = await persistCtaRankingSnapshot(request);
      setRanking(snapshot.ranking);
      // A fresh run is the current view.  Only an explicit history action
      // should mark the table as a historical snapshot.
      setActiveSnapshotId(null);
      await refreshHistory();
    } catch (cause) {
      const detail = cause instanceof Error ? cause.message : "CODEX 排名生成失败";
      setError(detail);
      message.error(detail);
    } finally {
      setLoading(false);
    }
 }, [buildRequest, refreshHistory]);

  const columns = useMemo(() => [
    {
      title: "#",
      dataIndex: "rank",
      width: 48,
      render: (value: number | null) => value ?? "—",
    },
    {
      title: "产品",
      dataIndex: "product_name",
      ellipsis: true,
      render: (value: string, row: CtaRankingProductResult) => (
        <Space size={6}>
          <Button type="link" style={{ height: "auto", padding: 0, fontWeight: 600 }} onClick={() => setPreviewProductId(row.product_id)}>{value}</Button>
          {!row.eligible && <Tag color="warning">样本不足</Tag>}
        </Space>
      ),
    },
    {
      title: "质量分",
      dataIndex: "score",
      width: 74,
      render: (value: number | null) => value == null ? "—" : <Text strong>{formatScore(value)}</Text>,
    },
    ...Object.entries(DIMENSION_LABELS).slice(0, 4).map(([key, label]) => ({
      title: label,
      key,
      width: 82,
      render: (_value: unknown, row: CtaRankingProductResult) => formatScore(row.dimension_scores[key]),
    })),
    {
      title: "状态",
      key: "status",
      width: 86,
      render: (_value: unknown, row: CtaRankingProductResult) => (
        <Tag color={row.status === "eligible" ? "success" : "warning"}>
          {row.status === "eligible" ? "可排名" : "短样本"}
        </Tag>
      ),
    },
  ], []);

  return (
    <Card
      size="small"
      title={<Space size={8}><Text strong>CODEX CTA 周度排名</Text><Tag color="blue">只读</Tag></Space>}
      extra={<Button size="small" type="primary" icon={<PlayCircleOutlined />} loading={loading} onClick={() => void runRanking()}>运行本周排名</Button>}
      style={{ marginBottom: 14 }}
    >
      {universeNote && <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>{universeNote}</Text>}
      {error && <Alert type="error" showIcon message={error} closable onClose={() => setError(null)} style={{ marginBottom: 10 }} />}
      {loading && !ranking ? (
        <div style={{ padding: 18, textAlign: "center" }}><Spin tip="正在计算五维分数与稳健排名…" /></div>
      ) : ranking ? (
        <>
          <Space wrap size={12} style={{ marginBottom: 8 }}>
            <Text type="secondary">截至 {ranking.as_of_date}</Text>
            <Text type="secondary">可排名 {ranking.eligible_count}/{ranking.universe_size}</Text>
            <Text type="secondary">模型 {ranking.model_version}</Text>
            {activeSnapshotId && <Tag color="blue">历史快照</Tag>}
            <Button size="small" type="text" icon={<ReloadOutlined />} loading={historyLoading} onClick={() => void refreshHistory()}>刷新历史</Button>
          </Space>
          <Table<CtaRankingProductResult>
            rowKey="product_id"
            size="small"
            pagination={{ pageSize: 50, showSizeChanger: false }}
            columns={columns}
            dataSource={ranking.rankings}
            scroll={{ x: 700 }}
          />
          {ranking.warnings.length > 0 && (
            <Alert
              type="warning"
              showIcon
              message="模型状态"
              description={ranking.warnings.join("；")}
              style={{ marginTop: 10 }}
            />
          )}
        </>
      ) : (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无排名结果" />
      )}
      {snapshots.length > 0 && (
        <List
          size="small"
          header={<Text type="secondary">历史排名快照</Text>}
          dataSource={snapshots.slice(0, 5)}
          renderItem={(snapshot) => (
            <List.Item
              actions={[
                <Button
                  key="view"
                  type="link"
                  size="small"
                  loading={snapshotLoadingId === snapshot.snapshot_id}
                  onClick={() => void loadSnapshot(snapshot.snapshot_id)}
                >
                  查看详情
                </Button>,
                <Popconfirm
                  key="delete"
                  title="删除这份历史排名？"
                  description="删除后无法恢复，不影响产品净值。"
                  okText="删除"
                  okButtonProps={{ danger: true, loading: snapshotDeletingId === snapshot.snapshot_id }}
                  cancelText="取消"
                  onConfirm={() => void deleteSnapshot(snapshot.snapshot_id)}
                >
                  <Button type="link" danger size="small" icon={<DeleteOutlined />} loading={snapshotDeletingId === snapshot.snapshot_id}>删除</Button>
                </Popconfirm>,
              ]}
            >
              <Space size={10}>
                <Text>{snapshot.as_of_date}</Text>
                <Text type="secondary">{snapshot.eligible_count}/{snapshot.universe_size} 只可排名</Text>
                <Text type="secondary">{snapshot.model_version}</Text>
                {activeSnapshotId === snapshot.snapshot_id && <Tag color="blue">当前</Tag>}
              </Space>
            </List.Item>
          )}
          style={{ marginTop: 10 }}
        />
      )}
      <ProductNavModal productId={previewProductId} open={previewProductId !== null} onClose={() => setPreviewProductId(null)} />
    </Card>
  );
}
