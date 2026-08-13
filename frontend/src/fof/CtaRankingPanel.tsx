/** Read-only CODEX CTA cross-sectional ranking panel. */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Empty, List, Space, Spin, Table, Tag, Typography, message } from "antd";
import { PlayCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import {
  kbGetNavSeries,
  kbListProducts,
  getLatestCtaPhaseDEvidence,
  getCtaRankingSnapshot,
  listCtaRankingSnapshots,
  persistCtaRankingSnapshot,
  type CtaRankingProductResult,
  type CtaRankingRequestPayload,
  type CtaRankingResponse,
  type CtaRankingSnapshotSummary,
  type DataFrequency,
} from "../api";

const { Text } = Typography;

const DIMENSION_LABELS: Record<string, string> = {
  absolute: "绝对收益",
  risk_adjusted: "风险调整",
  trend_regime: "趋势状态",
  tail: "极端环境",
  robustness: "稳健性",
};

function normalizeFrequency(value: string | null): DataFrequency {
  return value === "daily" || value === "monthly" ? value : "weekly";
}

function formatScore(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(1);
}

function commonAsOfDate(products: Array<{ nav_end: string | null }>): string | undefined {
  const dates = products.map((product) => product.nav_end).filter((value): value is string => Boolean(value)).sort();
  return dates[0];
}

interface Props {
  selectedProductIds?: string[];
}

export default function CtaRankingPanel({ selectedProductIds = [] }: Props): React.JSX.Element {
  const [ranking, setRanking] = useState<CtaRankingResponse | null>(null);
  const [snapshots, setSnapshots] = useState<CtaRankingSnapshotSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [snapshotLoadingId, setSnapshotLoadingId] = useState<string | null>(null);
  const [activeSnapshotId, setActiveSnapshotId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [universeNote, setUniverseNote] = useState<string | null>(null);

  const refreshHistory = useCallback(async (): Promise<void> => {
    setHistoryLoading(true);
    try {
      setSnapshots(await listCtaRankingSnapshots(12));
    } catch {
      // History is supplementary; ranking remains usable if the table is empty.
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
      const detail = cause instanceof Error ? cause.message : "加载 CODEX 排名快照失败";
      setError(detail);
      message.error(detail);
    } finally {
      setSnapshotLoadingId(null);
    }
  }, []);

  const buildRequest = useCallback(async (): Promise<CtaRankingRequestPayload> => {
    const products = await kbListProducts({ confirmation_status: "confirmed" });
    const selected = selectedProductIds.length
      ? products.filter((product) => selectedProductIds.includes(product.id))
      : products;
    const candidates = selected.filter((product) => (
      product.research_ready !== false
      && (product.reviewed_nav_count ?? 0) >= 2
    ));
    if (candidates.length === 0) {
      throw new Error("没有已确认且已复核净值的产品可排名");
    }

    const enriched = await Promise.all(candidates.map(async (product) => {
      const nav = await kbGetNavSeries(product.id, true);
      return {
        product,
        frequency: normalizeFrequency(product.nav_frequency),
        navPoints: nav
          .filter((point) => point.review_status === "reviewed" && Number.isFinite(point.nav) && point.nav > 0)
          .map((point) => ({
            observation_date: point.observation_date,
            net_asset_value: point.nav,
          })),
      };
    }));
    const frequencyCounts = new Map<DataFrequency, number>();
    for (const item of enriched) {
      frequencyCounts.set(item.frequency, (frequencyCounts.get(item.frequency) ?? 0) + 1);
    }
    const frequency = [...frequencyCounts.entries()]
      .sort((left, right) => right[1] - left[1])[0]?.[0] ?? "weekly";
    const compatible = enriched.filter((item) => item.frequency === frequency && item.navPoints.length >= 2);
    if (compatible.length === 0) {
      throw new Error("没有同频率的已复核净值序列可排名");
    }
    const excluded = enriched.length - compatible.length;
    setUniverseNote(
      excluded > 0
        ? `本次按${frequency}频率排名，排除了${excluded}只不同频率或净值不足的产品。`
        : `本次使用${compatible.length}只${frequency}产品，全部来自已确认且已复核的净值。`,
    );
    const asOfDate = commonAsOfDate(compatible.map((item) => item.product));
    const attributionEvidence = await getLatestCtaPhaseDEvidence(
      compatible.map((item) => item.product.id),
      asOfDate,
    );
    return {
      products: compatible.map((item) => ({
        product_id: item.product.id,
        product_name: item.product.standard_name,
        nav_points: item.navPoints,
        frequency: item.frequency,
        strategy: item.product.strategy,
      })),
      as_of_date: asOfDate,
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
          <Text strong>{value}</Text>
          {!row.eligible && <Tag color="warning">样本不足</Tag>}
        </Space>
      ),
    },
    {
      title: "总分",
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
      <Alert
        type="info"
        showIcon
        message="排名不会修改 FOF 配置"
        description="只使用已确认、已复核的产品净值；净值完整度、更新及时性和 FOF 边际分散化不进入得分。"
        style={{ marginBottom: 10 }}
      />
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
            {activeSnapshotId && <Tag color="gold">历史快照</Tag>}
            <Button size="small" type="text" icon={<ReloadOutlined />} loading={historyLoading} onClick={() => void refreshHistory()}>刷新历史</Button>
          </Space>
          <Table<CtaRankingProductResult>
            rowKey="product_id"
            size="small"
            pagination={false}
            columns={columns}
            dataSource={ranking.rankings}
            scroll={{ x: 700 }}
            expandable={{
              expandedRowRender: (row) => (
                <Space wrap size={[6, 6]}>
                  {row.dimensions.map((dimension) => (
                    <Tag key={dimension.dimension}>{dimension.label} {formatScore(dimension.adjusted_score)}</Tag>
                  ))}
                  {row.warnings.map((warning) => <Tag key={warning} color="warning">{warning}</Tag>)}
                </Space>
              ),
              rowExpandable: (row) => row.eligible,
            }}
          />
          {ranking.warnings.length > 0 && (
            <Alert
              type="warning"
              showIcon
              message="模型提示"
              description={ranking.warnings.join("；")}
              style={{ marginTop: 10 }}
            />
          )}
        </>
      ) : (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="点击“运行本周排名”生成当前可比产品排名" />
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
              ]}
            >
              <Space size={10}>
                <Text>{snapshot.as_of_date}</Text>
                <Text type="secondary">{snapshot.eligible_count}/{snapshot.universe_size} 只可排名</Text>
                <Text type="secondary">{snapshot.model_version}</Text>
                {activeSnapshotId === snapshot.snapshot_id && <Tag color="gold">当前</Tag>}
              </Space>
            </List.Item>
          )}
          style={{ marginTop: 10 }}
        />
      )}
    </Card>
  );
}
