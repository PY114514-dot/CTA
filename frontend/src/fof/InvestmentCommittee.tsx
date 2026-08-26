/** Investment-committee review panel (P4): approval/veto, memo export, outer-loop tracking. */

import React, { useCallback, useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  Alert,
  Button,
  Card,
  Collapse,
  Descriptions,
  Empty,
  Input,
  List,
  Modal,
  Popconfirm,
  Space,
  Statistic,
  Tag,
  Typography,
  message,
} from "antd";
import {
  AuditOutlined,
  CheckOutlined,
  CloseOutlined,
  FileTextOutlined,
  LineChartOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  kbApproveDecision,
  kbDeleteAllocationDraft,
  kbExportMemo,
  kbGetDecisionDetail,
  kbInterpretAllocation,
  kbListDecisions,
  kbReopenDecision,
  kbResolveReviewTask,
  kbRunTracking,
  kbSubmitAllocationDraft,
  kbVetoDecision,
  type DecisionStatus,
  type KbDecision,
  type KbDecisionDetail,
  type KbReviewTask,
  type KbTrackingRecord,
} from "../api";

const { Text, Paragraph } = Typography;

const STATUS_META: Record<DecisionStatus, { color: string; label: string }> = {
  draft: { color: "default", label: "草稿" },
  pending_review: { color: "processing", label: "待审核" },
  approved: { color: "success", label: "已通过" },
  vetoed: { color: "error", label: "已否决" },
  superseded: { color: "warning", label: "已替代" },
};

function pct(value: number | null | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
}

export default function InvestmentCommittee(): React.JSX.Element {
  const [decisions, setDecisions] = useState<KbDecision[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<KbDecisionDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const list = await kbListDecisions();
      setDecisions(list);
    } catch (e) {
      message.error(e instanceof Error ? e.message : "加载决策失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const loadDetail = useCallback(async (id: string) => {
    setDetailLoading(true);
    try {
      const d = await kbGetDecisionDetail(id);
      setDetail(d);
    } catch (e) {
      message.error(e instanceof Error ? e.message : "加载决策详情失败");
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const handleSelect = useCallback(
    (id: string) => {
      setActiveId(id);
      void loadDetail(id);
    },
    [loadDetail],
  );

  const afterMutation = useCallback(async () => {
    await refresh();
    if (activeId) await loadDetail(activeId);
  }, [refresh, loadDetail, activeId]);

  return (
    <Card
      size="small"
      style={{ height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}
      styles={{ body: { flex: 1, overflow: "auto", minHeight: 0, padding: "12px 14px" } }}
      title={
        <Space size={8}>
          <AuditOutlined style={{ color: "var(--serif-accent, #1677ff)" }} />
          <span>配置草案库</span>
          {decisions.length > 0 && <Tag style={{ fontSize: 10 }}>{decisions.length}</Tag>}
        </Space>
      }
      extra={
        <Button size="small" icon={<ReloadOutlined />} onClick={() => void refresh()} loading={loading}>
          刷新
        </Button>
      }
    >
      {decisions.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <Text type="secondary" style={{ fontSize: 12 }}>
              暂无配置草案。先在对话中保存配置草案，再提交人工确认。
            </Text>
          }
          style={{ marginTop: 48 }}
        />
      ) : (
        <Space direction="vertical" size={12} style={{ width: "100%" }}>
          <List
            size="small"
            loading={loading}
            dataSource={decisions}
            renderItem={(d) => {
              const meta = STATUS_META[d.status] ?? STATUS_META.draft;
              const metrics = d.portfolio_metrics;
              return (
                <List.Item
                  onClick={() => handleSelect(d.id)}
                  style={{
                    cursor: "pointer",
                    padding: "8px 10px",
                    borderRadius: 6,
                    border: activeId === d.id ? "1px solid var(--serif-accent, #1677ff)" : "1px solid transparent",
                    background: activeId === d.id ? "var(--serif-accent-soft, rgba(22,119,255,0.06))" : "transparent",
                  }}
                >
                  <div style={{ width: "100%" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <Text strong style={{ fontSize: 12 }}>{d.title ?? d.decision_type}</Text>
                      <Tag color={meta.color} style={{ fontSize: 10, margin: 0 }}>{meta.label}</Tag>
                    </div>
                    <Text type="secondary" style={{ fontSize: 10 }}>
                      {d.created_at ? new Date(d.created_at).toLocaleString("zh-CN") : ""}
                      {d.reviewer ? ` · ${d.reviewer}` : ""}
                    </Text>
                    <Space wrap size={[10, 2]} style={{ display: "flex", marginTop: 5, fontSize: 11 }}>
                      {typeof d.allocation_count === "number" && <Text type="secondary">{d.allocation_count} 只产品</Text>}
                      {metrics ? (
                        <>
                          <span>年化收益 {pct(metrics.annualized_return)}</span>
                          <span>年化波动 {pct(metrics.annualized_volatility)}</span>
                          <span>夏普 {metrics.sharpe_ratio?.toFixed(2) ?? "—"}</span>
                          <span>最大回撤 {pct(metrics.maximum_drawdown)}</span>
                        </>
                      ) : (
                        <Text type="secondary">尚未形成可计算的组合业绩</Text>
                      )}
                    </Space>
                  </div>
                </List.Item>
              );
            }}
          />

          {detail && (
            <DecisionDetailCard
              detail={detail}
              loading={detailLoading}
              onChanged={afterMutation}
              onDeleted={async () => {
                setActiveId(null);
                setDetail(null);
                await refresh();
              }}
            />
          )}
        </Space>
      )}
    </Card>
  );
}

function DecisionDetailCard({
  detail,
  loading,
  onChanged,
  onDeleted,
}: {
  detail: KbDecisionDetail;
  loading: boolean;
  onChanged: () => Promise<void>;
  onDeleted: () => Promise<void>;
}): React.JSX.Element {
  const [vetoOpen, setVetoOpen] = useState(false);
  const [vetoReason, setVetoReason] = useState("");
  const [memoOpen, setMemoOpen] = useState(false);
  const [memoMarkdown, setMemoMarkdown] = useState("");
  const [resolveTaskId, setResolveTaskId] = useState<string | null>(null);
  const [resolution, setResolution] = useState("");
  const [busy, setBusy] = useState(false);

  const allocations = detail.content?.allocations ?? [];
  const isPending = detail.status === "pending_review";
  const portfolio = detail.content?.portfolio;

  const handleApprove = async () => {
    setBusy(true);
    try {
      await kbApproveDecision(detail.id, "ic_chair");
      message.success("已通过该推荐");
      await onChanged();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "审批失败");
    } finally {
      setBusy(false);
    }
  };

  const handleSubmit = async () => {
    setBusy(true);
    try {
      await kbSubmitAllocationDraft(detail.id);
      message.success("已提交人工确认；通过后才会计算评分并冻结版本。");
      await onChanged();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "提交人工确认失败");
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    setBusy(true);
    try {
      await kbDeleteAllocationDraft(detail.id);
      message.success("配置草案已删除。");
      await onDeleted();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "删除配置草案失败");
    } finally {
      setBusy(false);
    }
  };

  const handleVeto = async () => {
    if (!vetoReason.trim()) {
      message.warning("请填写否决理由");
      return;
    }
    setBusy(true);
    try {
      await kbVetoDecision(detail.id, "ic_chair", vetoReason.trim());
      message.success("已否决该推荐");
      setVetoOpen(false);
      setVetoReason("");
      await onChanged();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "否决失败");
    } finally {
      setBusy(false);
    }
  };

  const handleReopen = async () => {
    setBusy(true);
    try {
      await kbReopenDecision(detail.id);
      message.success("已退回重新审议");
      await onChanged();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "操作失败");
    } finally {
      setBusy(false);
    }
  };

  const handleMemo = async () => {
    setBusy(true);
    try {
      const memo = await kbExportMemo(detail.id);
      setMemoMarkdown(memo.markdown);
      setMemoOpen(true);
    } catch (e) {
      message.error(e instanceof Error ? e.message : "导出备忘录失败");
    } finally {
      setBusy(false);
    }
  };

  const handleInterpret = async () => {
    setBusy(true);
    try {
      await kbInterpretAllocation(detail.id);
      message.success("已生成配置解读；未改变产品、权重或评分。");
      await onChanged();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "生成配置解读失败");
    } finally {
      setBusy(false);
    }
  };

  const handleTrack = async () => {
    setBusy(true);
    try {
      const result = await kbRunTracking(detail.id);
      if (result.status === "breach") {
        message.warning(`追踪检测到偏离阈值，已创建方法论复核任务`);
      } else {
        message.success("推荐后追踪完成，表现正常");
      }
      await onChanged();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "追踪失败");
    } finally {
      setBusy(false);
    }
  };

  const handleResolve = async () => {
    if (!resolveTaskId || !resolution.trim()) {
      message.warning("请填写处理结论");
      return;
    }
    setBusy(true);
    try {
      await kbResolveReviewTask(resolveTaskId, resolution.trim(), "ic_chair");
      message.success("复核任务已处理");
      setResolveTaskId(null);
      setResolution("");
      await onChanged();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "处理失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      size="small"
      loading={loading}
      style={{ border: "1px solid var(--serif-border, #f0f0f0)" }}
      styles={{ body: { padding: "10px 12px" } }}
      title={<Text strong style={{ fontSize: 12 }}>决策详情</Text>}
    >
      {/* Allocations */}
      {allocations.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>配置方案</Text>
          <div style={{ marginTop: 4, display: "flex", gap: 12, flexWrap: "wrap" }}>
            {allocations.map((a) => (
              <Statistic
                key={a.product_id}
                title={<span style={{ fontSize: 10 }}>{a.product_name ?? a.name ?? a.product_id}</span>}
                value={(a.weight * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ fontSize: 14, fontVariantNumeric: "tabular-nums" }}
              />
            ))}
          </div>
        </div>
      )}

      {portfolio && (
        <div style={{ marginBottom: 10 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>组合历史表现</Text>
          <Space wrap size={[12, 4]} style={{ display: "flex", marginTop: 4, fontSize: 12 }}>
            <span>年化收益 {pct(portfolio.metrics.annualized_return)}</span>
            <span>年化波动 {pct(portfolio.metrics.annualized_volatility)}</span>
            <span>夏普 {portfolio.metrics.sharpe_ratio?.toFixed(2) ?? "—"}</span>
            <span>最大回撤 {pct(portfolio.metrics.maximum_drawdown)}</span>
          </Space>
          <ReactECharts
            option={{
              grid: { left: 42, right: 16, top: 14, bottom: 26 },
              tooltip: { trigger: "axis", valueFormatter: (value: number) => Number(value).toFixed(3) },
              xAxis: { type: "category", data: portfolio.nav.map((point) => point.date), axisLabel: { fontSize: 10, hideOverlap: true } },
              yAxis: { type: "value", scale: true, axisLabel: { formatter: (value: number) => value.toFixed(2) } },
              series: [{ type: "line", data: portfolio.nav.map((point) => point.nav), showSymbol: false, lineStyle: { color: "#2563EB", width: 2 }, areaStyle: { color: "rgba(37, 99, 235, 0.08)" } }],
            }}
            style={{ height: 200, marginTop: 6 }}
          />
        </div>
      )}

      {detail.content?.method_provenance && Object.keys(detail.content.method_provenance).length > 0 && (
        <Collapse
          size="small"
          ghost
          style={{ marginBottom: 10 }}
          items={[{
            key: "method-provenance",
            label: <span style={{ fontSize: 11, color: "var(--serif-text-secondary, #888)" }}>本次 FOF 结果的方法来源</span>,
            children: (
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {Object.entries(detail.content.method_provenance).map(([key, item]) => (
                  <div key={key} style={{ fontSize: 11, color: "var(--serif-text-secondary, #666)" }}>
                    <Text strong style={{ fontSize: 11 }}>{key}：</Text>
                    {item.method ?? "未记录"}
                    {item.model ? `（${item.model}）` : ""}
                    {item.detail ? ` · ${item.detail}` : ""}
                  </div>
                ))}
                <Text type="secondary" style={{ fontSize: 10 }}>当前配置由本地规则和统计工具生成，LLM 仅可作为解释层，不会替代计算或审核。</Text>
              </div>
            ),
          }]}
        />
      )}

      {detail.veto_reason && (
        <Alert type="error" showIcon message="否决理由" description={detail.veto_reason} style={{ marginBottom: 10 }} />
      )}
      {detail.content?.ic_comment && (
        <Alert type="success" showIcon message="审核意见" description={detail.content.ic_comment} style={{ marginBottom: 10 }} />
      )}
      {detail.content?.llm_interpretation && (
        <Alert type="info" showIcon message="配置解读" description={detail.content.llm_interpretation} style={{ marginBottom: 10 }} />
      )}
      {detail.content?.deterministic_score && (
        <Descriptions size="small" column={2} style={{ marginBottom: 10 }} title="配置评分">
          <Descriptions.Item label="总分">{detail.content.deterministic_score.total}</Descriptions.Item>
          {Object.entries(detail.content.deterministic_score.components ?? {}).map(([name, value]) => <Descriptions.Item key={name} label={name}>{String(value)}</Descriptions.Item>)}
        </Descriptions>
      )}

      {/* IC actions */}
      <Space wrap size={6} style={{ marginBottom: 10 }}>
        {(detail.status === "draft" || detail.status === "vetoed") && (
          <Popconfirm title="删除配置草案？" description="删除后无法恢复。" okText="删除" cancelText="取消" okButtonProps={{ danger: true, loading: busy }} onConfirm={() => void handleDelete()}>
            <Button size="small" danger loading={busy}>删除草案</Button>
          </Popconfirm>
        )}
        {detail.status === "draft" && <Button size="small" type="primary" onClick={() => void handleSubmit()} loading={busy}>提交人工确认</Button>}
        {isPending && (
          <>
            <Button size="small" icon={<CheckOutlined />} onClick={() => void handleApprove()} loading={busy}>
              确认通过
            </Button>
            <Button size="small" danger icon={<CloseOutlined />} onClick={() => setVetoOpen(true)} loading={busy}>
              否决
            </Button>
          </>
        )}
        {(detail.status === "approved" || detail.status === "vetoed") && (
          <Button size="small" onClick={() => void handleReopen()} loading={busy}>
            退回重审
          </Button>
        )}
        <Button size="small" icon={<FileTextOutlined />} onClick={() => void handleMemo()} loading={busy}>
          研究备忘录
        </Button>
        <Button size="small" onClick={() => void handleInterpret()} loading={busy}>生成配置解读</Button>
        {detail.status === "approved" && <Button size="small" icon={<LineChartOutlined />} onClick={() => void handleTrack()} loading={busy}>
          推荐后追踪
        </Button>}
      </Space>

      {/* Tracking history */}
      {detail.tracking.length > 0 && (
        <Collapse
          size="small"
          style={{ marginBottom: 10 }}
          items={[
            {
              key: "tracking",
              label: <span style={{ fontSize: 12 }}>追踪记录（{detail.tracking.length}）</span>,
              children: (
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  {detail.tracking.map((t) => (
                    <TrackingRow key={t.id} record={t} />
                  ))}
                </Space>
              ),
            },
          ]}
        />
      )}

      {/* Review tasks */}
      {detail.review_tasks.length > 0 && (
        <Collapse
          size="small"
          items={[
            {
              key: "tasks",
              label: <span style={{ fontSize: 12 }}>复核任务（{detail.review_tasks.length}）</span>,
              children: (
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  {detail.review_tasks.map((task) => (
                    <ReviewTaskRow
                      key={task.id}
                      task={task}
                      onResolve={() => {
                        setResolveTaskId(task.id);
                        setResolution("");
                      }}
                    />
                  ))}
                </Space>
              ),
            },
          ]}
        />
      )}

      {/* Veto modal */}
      <Modal
        title="否决推荐"
        open={vetoOpen}
        onOk={() => void handleVeto()}
        onCancel={() => setVetoOpen(false)}
        okText="确认否决"
        okButtonProps={{ danger: true, loading: busy }}
        cancelText="取消"
      >
        <Text type="secondary" style={{ fontSize: 12 }}>否决理由将记入审计档案，必填。</Text>
        <Input.TextArea
          rows={3}
          value={vetoReason}
          onChange={(e) => setVetoReason(e.target.value)}
          placeholder="例如：流动性不足 / 相关性过高 / 尽调材料缺失"
          style={{ marginTop: 8 }}
        />
      </Modal>

      {/* Memo modal */}
      <Modal
        title="研究备忘录"
        open={memoOpen}
        onCancel={() => setMemoOpen(false)}
        footer={null}
        width={680}
      >
        <pre
          style={{
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            fontSize: 12,
            maxHeight: 480,
            overflow: "auto",
            background: "var(--serif-bg, #fafafa)",
            padding: 12,
            borderRadius: 6,
          }}
        >
          {memoMarkdown}
        </pre>
      </Modal>

      {/* Resolve task modal */}
      <Modal
        title="处理复核任务"
        open={resolveTaskId !== null}
        onOk={() => void handleResolve()}
        onCancel={() => setResolveTaskId(null)}
        okText="确认处理"
        okButtonProps={{ loading: busy }}
        cancelText="取消"
      >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 8 }}
          message="外环仅提出变更建议；任何方法论调整须经人工审核、版本化并回测后生效。"
        />
        <Input.TextArea
          rows={3}
          value={resolution}
          onChange={(e) => setResolution(e.target.value)}
          placeholder="填写人工处理结论，例如：维持现有权重，补充回撤约束至 15%"
        />
      </Modal>
    </Card>
  );
}

function TrackingRow({ record }: { record: KbTrackingRecord }): React.JSX.Element {
  const breached = record.status === "breach";
  const insufficient = record.status === "insufficient_data";
  const borderColor = breached
    ? "var(--serif-danger, #ff4d4f)"
    : insufficient
      ? "var(--serif-warning, #faad14)"
      : "var(--serif-border, #f0f0f0)";
  return (
    <div
      style={{
        padding: "6px 8px",
        borderRadius: 6,
        border: `1px ${insufficient ? "dashed" : "solid"} ${borderColor}`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Text style={{ fontSize: 11 }}>{record.review_date ?? "—"}</Text>
        <Tag
          color={breached ? "error" : insufficient ? "warning" : "success"}
          style={{ fontSize: 10, margin: 0 }}
        >
          {breached ? "偏离阈值" : insufficient ? "数据不足" : "正常"}
        </Tag>
      </div>
      <div style={{ marginTop: 4, fontSize: 11, color: "var(--serif-text-secondary, #888)" }}>
        预期 {pct(record.expected_return)} · 实际 {pct(record.actual_return)} · 偏差 {pct(record.return_deviation)} · 回撤 {pct(record.actual_max_drawdown)}
      </div>
      {record.breach_reasons && record.breach_reasons.length > 0 && (
        <div style={{ marginTop: 4 }}>
          {record.breach_reasons.map((r, i) => (
            <div key={i} style={{ fontSize: 10, color: "var(--serif-danger, #ff4d4f)" }}>· {r}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function ReviewTaskRow({ task, onResolve }: { task: KbReviewTask; onResolve: () => void }): React.JSX.Element {
  const open = task.status === "open" || task.status === "in_progress";
  return (
    <div style={{ padding: "6px 8px", borderRadius: 6, border: "1px solid var(--serif-border, #f0f0f0)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Text style={{ fontSize: 11 }}>{task.title}</Text>
        <Tag color={open ? "warning" : "default"} style={{ fontSize: 10, margin: 0 }}>
          {open ? "待处理" : "已处理"}
        </Tag>
      </div>
      {task.description && (
        <Paragraph style={{ fontSize: 10, margin: "4px 0 0", color: "var(--serif-text-secondary, #888)" }}>
          {task.description}
        </Paragraph>
      )}
      {task.resolution && (
        <Paragraph style={{ fontSize: 10, margin: "4px 0 0" }}>
          结论：{task.resolution}（{task.resolved_by}）
        </Paragraph>
      )}
      {open && (
        <Button size="small" style={{ marginTop: 6, fontSize: 10 }} onClick={onResolve}>
          处理
        </Button>
      )}
    </div>
  );
}
