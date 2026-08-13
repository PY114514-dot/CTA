/** Phase-A CTA attribution panel: reads only confirmed, reviewed NAV. */

import React, { useCallback, useEffect, useState } from "react";
import { Alert, Button, Card, Descriptions, Empty, List, Segmented, Space, Spin, Table, Tag, Typography, message } from "antd";
import { createCtaAttributionSnapshot, getCtaAttributionSnapshot, getCtaPhaseA, getCtaPhaseB, getCtaPhaseC, getCtaPhaseD, listCtaAttributionSnapshots, type CtaAttributionPhase, type CtaAttributionSnapshotSummary, type CtaPhaseAResponse, type CtaPhaseBResponse, type CtaPhaseCResponse, type CtaPhaseDResponse, type KbProduct } from "../api";

const { Text } = Typography;

function percent(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}

export default function CtaAttributionPanel({ product }: { product: KbProduct | undefined }): React.JSX.Element {
  const [result, setResult] = useState<CtaPhaseAResponse | null>(null);
  const [dynamicResult, setDynamicResult] = useState<CtaPhaseBResponse | null>(null);
  const [regimeResult, setRegimeResult] = useState<CtaPhaseCResponse | null>(null);
  const [nonlinearResult, setNonlinearResult] = useState<CtaPhaseDResponse | null>(null);
  const [view, setView] = useState<"static" | "dynamic" | "regime" | "nonlinear">("static");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [snapshotSaving, setSnapshotSaving] = useState(false);
  const [snapshots, setSnapshots] = useState<CtaAttributionSnapshotSummary[]>([]);
  const [snapshotLoadingId, setSnapshotLoadingId] = useState<string | null>(null);
  const [frozenSnapshotId, setFrozenSnapshotId] = useState<string | null>(null);
  const [frozenSnapshotPhase, setFrozenSnapshotPhase] = useState<CtaAttributionPhase | null>(null);

  const ready = product?.confirmation_status === "confirmed" && (product.reviewed_nav_count ?? 0) >= 20;
  const phaseForView: CtaAttributionPhase = view === "static" ? "phase-a" : view === "dynamic" ? "phase-b" : view === "regime" ? "phase-c" : "phase-d";

  const refreshSnapshots = useCallback(async () => {
    if (!product || !ready) {
      setSnapshots([]);
      return;
    }
    try {
      setSnapshots(await listCtaAttributionSnapshots(product.id, 12));
    } catch {
      // Snapshot history is supplementary; the live read-only result remains usable.
    }
  }, [product, ready]);

  const load = useCallback(async () => {
    if (!product || !ready) return;
    if (frozenSnapshotId && frozenSnapshotPhase === phaseForView) return;
    setLoading(true);
    setError(null);
    try {
      if (view === "static") {
        setResult(await getCtaPhaseA(product.id));
      } else if (view === "dynamic") {
        setDynamicResult(await getCtaPhaseB(product.id));
      } else {
        if (view === "regime") setRegimeResult(await getCtaPhaseC(product.id));
        else setNonlinearResult(await getCtaPhaseD(product.id));
      }
      setFrozenSnapshotId(null);
      setFrozenSnapshotPhase(null);
    } catch (cause) {
      setResult(null);
      setDynamicResult(null);
      setRegimeResult(null);
      setNonlinearResult(null);
      setError(cause instanceof Error ? cause.message : "CTA 已审核净值归因失败");
    } finally {
      setLoading(false);
    }
  }, [frozenSnapshotId, frozenSnapshotPhase, phaseForView, product, ready, view]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => { void refreshSnapshots(); }, [refreshSnapshots]);

  const saveSnapshot = useCallback(async () => {
    if (!product) return;
    setSnapshotSaving(true);
    try {
      const saved = await createCtaAttributionSnapshot(product.id, phaseForView);
      message.success(saved.idempotent ? "已存在相同输入的冻结快照" : "CTA 归因快照已保存");
      await refreshSnapshots();
    } catch (cause) {
      message.error(cause instanceof Error ? cause.message : "保存 CTA 归因快照失败");
    } finally {
      setSnapshotSaving(false);
    }
  }, [phaseForView, product, refreshSnapshots]);

  const loadSnapshot = useCallback(async (snapshotId: string) => {
    setSnapshotLoadingId(snapshotId);
    setError(null);
    try {
      const snapshot = await getCtaAttributionSnapshot(snapshotId);
      setView(snapshot.phase === "phase-a" ? "static" : snapshot.phase === "phase-b" ? "dynamic" : snapshot.phase === "phase-c" ? "regime" : "nonlinear");
      setFrozenSnapshotId(snapshot.snapshot_id);
      setFrozenSnapshotPhase(snapshot.phase);
      if (snapshot.phase === "phase-a") setResult(snapshot.results as CtaPhaseAResponse);
      if (snapshot.phase === "phase-b") setDynamicResult(snapshot.results as CtaPhaseBResponse);
      if (snapshot.phase === "phase-c") setRegimeResult(snapshot.results as CtaPhaseCResponse);
      if (snapshot.phase === "phase-d") setNonlinearResult(snapshot.results as CtaPhaseDResponse);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "读取 CTA 归因快照失败");
    } finally {
      setSnapshotLoadingId(null);
    }
  }, []);

  const clearFrozenSnapshot = useCallback(() => {
    setFrozenSnapshotId(null);
    setFrozenSnapshotPhase(null);
    setError(null);
  }, []);

  const snapshotControls = <SnapshotControls
    phase={phaseForView}
    saving={snapshotSaving}
    snapshots={snapshots}
    loadingId={snapshotLoadingId}
    frozenSnapshotId={frozenSnapshotId}
    onSave={() => void saveSnapshot()}
    onLoad={(snapshotId) => void loadSnapshot(snapshotId)}
    onClear={clearFrozenSnapshot}
  />;

  if (!product) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="从产品库选择一只产品查看动态归因" />;
  if (!ready) return <Alert type="info" showIcon message="尚未达到已审核归因门槛" description="需要产品身份已确认，且至少有 20 条人工审核净值。此处不会调用 VLM、OCR 或图表识别。" />;
  if (loading) return <div style={{ padding: 42, textAlign: "center" }}><Spin tip="正在读取已审核净值和公开因子…" /></div>;
  if (error) return <Alert type="error" showIcon message={error} />;
  if (view === "dynamic") {
    return <Space direction="vertical" size={12} style={{ width: "100%" }}>
      <ViewSelector value={view} onChange={setView} />
      {snapshotControls}
      {dynamicResult ? <DynamicBetaView result={dynamicResult} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无动态 Beta 结果" />}
    </Space>;
  }
  if (view === "regime") {
    return <Space direction="vertical" size={12} style={{ width: "100%" }}>
      <ViewSelector value={view} onChange={setView} />
      {snapshotControls}
      {regimeResult ? <RegimeView result={regimeResult} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无状态归因结果" />}
    </Space>;
  }
  if (view === "nonlinear") {
    return <Space direction="vertical" size={12} style={{ width: "100%" }}>
      <ViewSelector value={view} onChange={setView} />
      {snapshotControls}
      {nonlinearResult ? <NonlinearView result={nonlinearResult} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无非线性增量结果" />}
    </Space>;
  }
  if (!result) return <Space direction="vertical" size={12} style={{ width: "100%" }}><ViewSelector value={view} onChange={setView} />{snapshotControls}</Space>;

  const returnsByFactor = new Map(result.attribution.return_contribution.map((row) => [row.factor_name, row]));
  const risksByFactor = new Map(result.attribution.euler_risk_contribution.map((row) => [row.factor_name, row]));
  const factors = result.attribution.factor_exposure.map((row) => ({ ...row, contribution: returnsByFactor.get(row.factor_name)?.contribution_pct_of_mean_return, risk: risksByFactor.get(row.factor_name)?.component_risk_contribution }));

  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <ViewSelector value={view} onChange={setView} />
    {snapshotControls}
    {frozenSnapshotId && <Alert type="warning" showIcon message="当前显示的是冻结快照" description="历史快照只读，不会重新计算，也不会触发资料解析。点击“返回最新结果”恢复当前数据。" />}
    <Alert type="success" showIcon message="已审核净值 · 只读统计归因" description={`${result.data_contract.reviewed_nav_count} 个${result.frequency}净值点，${result.performance_path.start_date} 至 ${result.performance_path.end_date}。未触发资料解析。`} />
    <Descriptions size="small" bordered column={3}>
      <Descriptions.Item label="CAGR">{percent(result.performance_path.cagr)}</Descriptions.Item>
      <Descriptions.Item label="最大回撤">{percent(result.performance_path.maximum_drawdown)}</Descriptions.Item>
      <Descriptions.Item label="Calmar">{result.performance_path.calmar?.toFixed(2) ?? "—"}</Descriptions.Item>
      <Descriptions.Item label="ES 5%">{percent(result.performance_path.expected_shortfall_5pct)}</Descriptions.Item>
      <Descriptions.Item label="最长回撤">{result.performance_path.drawdown_duration_max_periods} 期</Descriptions.Item>
      <Descriptions.Item label="恢复状态"><Tag color={result.performance_path.recovery_completed ? "success" : "warning"}>{result.performance_path.recovery_completed ? "已恢复" : "尚未恢复"}</Tag></Descriptions.Item>
      <Descriptions.Item label="R²">{result.baseline.r_squared.toFixed(3)}</Descriptions.Item>
      <Descriptions.Item label="样本外 R²">{result.baseline.out_of_sample.r_squared?.toFixed(3) ?? "—"}</Descriptions.Item>
      <Descriptions.Item label="残差 Alpha 候选">{percent(result.baseline.annualized_alpha_candidate)}</Descriptions.Item>
    </Descriptions>
    <Table
      size="small" rowKey="factor_name" pagination={false} dataSource={factors}
      columns={[
        { title: "因子", dataIndex: "display_name" },
        { title: "组别", dataIndex: "factor_group" },
        { title: "Beta", dataIndex: "beta", render: (value: number) => value.toFixed(4) },
        { title: "收益贡献%", dataIndex: "contribution", render: (value: number | undefined) => value == null ? "—" : value.toFixed(1) },
        { title: "Euler 风险贡献", dataIndex: "risk", render: (value: number | null | undefined) => value == null ? "—" : value.toFixed(4) },
      ]}
    />
    <Alert type="warning" showIcon message="解释边界" description="Beta、收益贡献和 Euler 风险贡献是三个不同的收益型统计对象；不代表真实持仓、真实 P&L、交易成本或管理人技能。" />
    {result.baseline.warnings.length > 0 && <Text type="secondary" style={{ fontSize: 12 }}>{result.baseline.warnings.join("；")}</Text>}
  </Space>;
}

function ViewSelector({ value, onChange }: { value: "static" | "dynamic" | "regime" | "nonlinear"; onChange: (value: "static" | "dynamic" | "regime" | "nonlinear") => void }): React.JSX.Element {
  return <Segmented
    size="small"
    value={value}
    onChange={(next) => onChange(next as "static" | "dynamic" | "regime" | "nonlinear")}
    options={[{ label: "静态基线", value: "static" }, { label: "动态 Beta", value: "dynamic" }, { label: "状态归因", value: "regime" }, { label: "非线性增量", value: "nonlinear" }]}
  />;
}

function DynamicBetaView({ result }: { result: CtaPhaseBResponse }): React.JSX.Element {
  const labels: Record<string, string> = {
    trend: "趋势",
    basis_carry: "基差/Carry",
    short_term_trend_20: "短趋势",
    mean_reversion_5d: "反转",
  };
  const latest = result.kalman_beta.paths.at(-1);
  const rollingAlerts = result.rolling_beta.windows.flatMap((window) => window.alerts ?? []);
  const latestRows = result.kalman_beta.factor_names.map((name) => ({
    key: name,
    factor_name: name,
    display_name: labels[name] ?? name,
    long_run_beta: result.kalman_beta.long_run_style_beta[name],
    latest_beta: latest?.betas[name],
    standard_error: latest?.beta_standard_errors[name],
    beta_range: result.kalman_beta.beta_range[name],
    dynamic_contribution: latest?.dynamic_timing_contribution_by_factor[name],
  }));
  const latestPath = result.kalman_beta.paths.slice(-80);
  const chartWidth = 720;
  const chartHeight = 170;
  const values = latestPath.flatMap((point) => result.kalman_beta.factor_names.map((name) => point.betas[name] ?? 0));
  const minValue = values.length ? Math.min(...values) : -1;
  const maxValue = values.length ? Math.max(...values) : 1;
  const span = Math.max(maxValue - minValue, 1e-8);
  const xPosition = (index: number) => latestPath.length <= 1 ? 0 : index / (latestPath.length - 1) * chartWidth;
  const yPosition = (value: number) => chartHeight - (value - minValue) / span * chartHeight;

  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <Alert type="success" showIcon message="动态 Beta · 因果过滤结果" description={`${result.alignment.aligned_observation_count} 个已对齐观测，因子缓存口径为 baseline；平滑器关闭，未使用未来数据。`} />
    <Descriptions size="small" bordered column={3}>
      <Descriptions.Item label="对齐样本">{result.alignment.aligned_observation_count}</Descriptions.Item>
      <Descriptions.Item label="起始日期">{result.alignment.start_date}</Descriptions.Item>
      <Descriptions.Item label="结束日期">{result.alignment.end_date}</Descriptions.Item>
      <Descriptions.Item label="动态贡献累计">{result.kalman_beta.dynamic_timing_contribution_total.toFixed(6)}</Descriptions.Item>
      <Descriptions.Item label="创新项 RMS">{result.kalman_beta.innovation_root_mean_square.toFixed(6)}</Descriptions.Item>
      <Descriptions.Item label="平滑器"><Tag color="green">未使用</Tag></Descriptions.Item>
    </Descriptions>
    <Card size="small" title="Kalman Beta 路径（最近 80 个观测）">
      {latestPath.length > 1 ? <div style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${chartWidth} ${chartHeight + 28}`} width="100%" height="198" role="img" aria-label="Kalman 动态 Beta 路径">
          <line x1="0" y1={yPosition(0)} x2={chartWidth} y2={yPosition(0)} stroke="#d9d9d9" strokeDasharray="4 4" />
          {result.kalman_beta.factor_names.map((name, factorIndex) => {
            const points = latestPath.map((point, index) => `${xPosition(index)},${yPosition(point.betas[name] ?? 0)}`).join(" ");
            const color = ["#1677ff", "#13c2c2", "#722ed1", "#fa8c16"][factorIndex % 4];
            return <g key={name}>
              <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
              <text x={8 + factorIndex * 120} y={chartHeight + 20} fill={color} fontSize="11">{labels[name] ?? name}</text>
            </g>;
          })}
        </svg>
      </div> : <Text type="secondary">动态路径不足以绘图。</Text>}
    </Card>
    <Table
      size="small" rowKey="key" pagination={false} dataSource={latestRows} scroll={{ x: 720 }}
      columns={[
        { title: "因子", dataIndex: "display_name" },
        { title: "长期风格 Beta", dataIndex: "long_run_beta", render: (value: number) => value?.toFixed(4) ?? "—" },
        { title: "最新 Beta", dataIndex: "latest_beta", render: (value: number) => value?.toFixed(4) ?? "—" },
        { title: "最新标准误", dataIndex: "standard_error", render: (value: number) => value?.toFixed(4) ?? "—" },
        { title: "路径范围", dataIndex: "beta_range", render: (value: number) => value?.toFixed(4) ?? "—" },
        { title: "最新动态贡献", dataIndex: "dynamic_contribution", render: (value: number) => value?.toFixed(6) ?? "—" },
      ]}
    />
    {result.rolling_beta.windows.map((window) => <Card key={window.window} size="small" title={`Rolling Beta · ${window.window} 期`}>
      {window.status === "insufficient" ? <Alert type="warning" showIcon message="样本不足" description={window.warnings.join("；")} /> : <>
        <Space wrap size={[6, 6]}>
          {Object.entries(window.summary ?? {}).map(([name, summary]) => <Tag key={name} color={summary.sign_change_count > 0 ? "warning" : "default"}>
            {labels[name] ?? name}：符号一致率 {summary.sign_consistency == null ? "—" : `${(summary.sign_consistency * 100).toFixed(0)}%`} · 变化 {summary.sign_change_count} 次
          </Tag>)}
        </Space>
        {(window.alerts ?? []).length > 0 && <Alert type="warning" showIcon style={{ marginTop: 8 }} message={(window.alerts ?? []).map((alert) => alert.message).join("；")} />}
      </>}
    </Card>)}
    {(rollingAlerts.length > 0 || result.warnings.length > 0 || result.kalman_beta.warnings.length > 0) && <Alert type="warning" showIcon message="模型解释边界" description={[...result.warnings, ...result.kalman_beta.warnings, ...rollingAlerts.map((alert) => alert.message)].join("；")} />}
  </Space>;
}

function RegimeView({ result }: { result: CtaPhaseCResponse }): React.JSX.Element {
  const stateLabels: Record<string, string> = { normal: "Normal", crisis: "Crisis", whipsaw: "Whipsaw" };
  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <Alert type="info" showIcon message="可观测状态归因" description={`状态代理：${result.state_proxy.market_proxy_factor}。${result.state_proxy.interpretation} 状态规则未使用未来数据。`} />
    <Descriptions size="small" bordered column={4}>
      <Descriptions.Item label="Normal">{result.regime_rules.state_counts.normal ?? 0} 期</Descriptions.Item>
      <Descriptions.Item label="Crisis">{result.regime_rules.state_counts.crisis ?? 0} 期</Descriptions.Item>
      <Descriptions.Item label="Whipsaw">{result.regime_rules.state_counts.whipsaw ?? 0} 期</Descriptions.Item>
      <Descriptions.Item label="规则">滞后代理 / 非 HMM</Descriptions.Item>
    </Descriptions>
    <Table
      size="small" rowKey="state" pagination={false} dataSource={result.regime_attribution.regimes}
      columns={[
        { title: "状态", dataIndex: "state", render: (value: string) => <Tag color={value === "crisis" ? "red" : value === "whipsaw" ? "orange" : "blue"}>{stateLabels[value] ?? value}</Tag> },
        { title: "样本数", dataIndex: "observation_count" },
        { title: "条件收益", dataIndex: "conditional_return", render: (value: number | null) => percent(value) },
        { title: "ES 5%", dataIndex: "expected_shortfall_5pct", render: (value: number | null) => percent(value) },
        { title: "条件相关", dataIndex: "conditional_correlation", render: (value: number | null) => value == null ? "—" : value.toFixed(3) },
        { title: "Crisis Alpha 候选", dataIndex: "crisis_alpha_candidate", render: (value: number | null) => percent(value) },
        { title: "状态", dataIndex: "status", render: (value: string) => <Tag color={value === "available" ? "success" : "warning"}>{value === "available" ? "可估计" : "样本不足"}</Tag> },
      ]}
    />
    {result.regime_attribution.regimes.filter((item) => item.status === "available").map((item) => <Card key={item.state} size="small" title={`${stateLabels[item.state]} 条件 Beta`}>
      <Space wrap>{Object.entries(item.conditional_beta ?? {}).map(([factor, value]) => <Tag key={factor}>{factor}: {value.toFixed(4)}</Tag>)}</Space>
    </Card>)}
    <Alert type="warning" showIcon message="解释边界" description={result.warnings.join("；")} />
  </Space>;
}

function NonlinearView({ result }: { result: CtaPhaseDResponse }): React.JSX.Element {
  const increment = result.nonlinear_increment;
  const summary = increment.summary;
  const available = increment.status === "available";
  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <Alert
      type={summary.stable_improvement ? "success" : "info"}
      showIcon
      message={summary.conclusion}
      description={`${increment.parameters.training_scheme} · ${increment.parameters.test_scheme} · 未使用未来数据。线性 OOS 与 HistGradientBoosting OOS 只在连续测试段比较。`}
    />
    <Descriptions size="small" bordered column={4}>
      <Descriptions.Item label="状态"><Tag color={available ? "success" : "warning"}>{available ? "可评估" : "样本不足"}</Tag></Descriptions.Item>
      <Descriptions.Item label="测试段数">{summary.evaluated_delta_count} / {increment.parameters.min_segments}</Descriptions.Item>
      <Descriptions.Item label="平均 OOS R² 增量">{summary.mean_r2_delta == null ? "—" : summary.mean_r2_delta.toFixed(4)}</Descriptions.Item>
      <Descriptions.Item label="正增量比例">{summary.positive_delta_fraction == null ? "—" : `${(summary.positive_delta_fraction * 100).toFixed(0)}%`}</Descriptions.Item>
      <Descriptions.Item label="训练窗口">{increment.parameters.train_window} 期</Descriptions.Item>
      <Descriptions.Item label="测试窗口">{increment.parameters.test_window} 期</Descriptions.Item>
      <Descriptions.Item label="最低 R² 增量">{increment.parameters.min_r2_uplift.toFixed(3)}</Descriptions.Item>
      <Descriptions.Item label="敏感性选择"><Tag color={increment.sensitivity.selection_policy.selected_from_sensitivity ? "warning" : "green"}>{increment.sensitivity.selection_policy.selected_from_sensitivity ? "参与选择" : "仅诊断"}</Tag></Descriptions.Item>
    </Descriptions>
    {increment.segments.length > 0 && <Table
      size="small"
      rowKey="segment"
      pagination={false}
      dataSource={increment.segments}
      columns={[
        { title: "测试段", dataIndex: "segment" },
        { title: "测试期", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => `${row.test_start_date} → ${row.test_end_date}` },
        { title: "线性 OOS R²", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => row.linear.oos_r2 == null ? "—" : row.linear.oos_r2.toFixed(4) },
        { title: "非线性 OOS R²", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => row.nonlinear.oos_r2 == null ? "—" : row.nonlinear.oos_r2.toFixed(4) },
        { title: "R² 增量", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => row.r2_delta == null ? "—" : row.r2_delta.toFixed(4) },
        { title: "相关增量", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => row.correlation_delta == null ? "—" : row.correlation_delta.toFixed(4) },
      ]}
    />}
    <Card size="small" title="参数敏感性（只读诊断）">
      <Space direction="vertical" size={6} style={{ width: "100%" }}>
        <Text type="secondary">窗口、阈值和模型参数情景均未用于挑选模型；只有预登记基准情景决定结论。</Text>
        <Space wrap>
          {increment.sensitivity.model_parameter_sensitivity.map((scenario) => <Tag key={String(scenario.label)} color={scenario.stable_improvement ? "success" : "default"}>{String(scenario.label)}：{scenario.conclusion ? String(scenario.conclusion) : "—"}</Tag>)}
          <Tag>Bonferroni：{increment.sensitivity.multiple_testing.bonferroni_adjusted_p_value == null ? "—" : increment.sensitivity.multiple_testing.bonferroni_adjusted_p_value.toFixed(4)}</Tag>
          <Tag>DSR：不适用</Tag>
        </Space>
      </Space>
    </Card>
    <Alert type="warning" showIcon message="解释边界" description={[...increment.warnings, ...result.warnings].join("；")} />
  </Space>;
}

function SnapshotControls({
  phase,
  saving,
  snapshots,
  loadingId,
  frozenSnapshotId,
  onSave,
  onLoad,
  onClear,
}: {
  phase: CtaAttributionPhase;
  saving: boolean;
  snapshots: CtaAttributionSnapshotSummary[];
  loadingId: string | null;
  frozenSnapshotId: string | null;
  onSave: () => void;
  onLoad: (snapshotId: string) => void;
  onClear: () => void;
}): React.JSX.Element {
  const phaseLabels: Record<CtaAttributionPhase, string> = {
    "phase-a": "静态基线",
    "phase-b": "动态 Beta",
    "phase-c": "状态归因",
    "phase-d": "非线性增量",
  };
  const phaseSnapshots = snapshots.filter((snapshot) => snapshot.phase === phase);
  return <Card size="small" title="不可变归因快照" extra={<Space size={6}>
    {frozenSnapshotId && <Button size="small" onClick={onClear}>返回最新结果</Button>}
    <Button size="small" type="primary" loading={saving} onClick={onSave}>保存{phaseLabels[phase]}快照</Button>
  </Space>}>
    <Text type="secondary" style={{ fontSize: 12 }}>
      快照保存已审核净值指纹、baseline 因子版本、模型参数和结果；相同输入重复保存会复用原快照。
    </Text>
    {phaseSnapshots.length > 0 && <List
      size="small"
      header={<Text type="secondary">{phaseLabels[phase]}历史快照</Text>}
      dataSource={phaseSnapshots.slice(0, 5)}
      renderItem={(snapshot) => <List.Item actions={[<Button key="load" type="link" size="small" loading={loadingId === snapshot.snapshot_id} onClick={() => onLoad(snapshot.snapshot_id)}>查看冻结结果</Button>]}>
        <Space size={8} wrap>
          <Text>{snapshot.as_of_date}</Text>
          <Text type="secondary">{snapshot.model_version}</Text>
          <Text type="secondary">NAV {snapshot.nav_fingerprint}</Text>
          {snapshot.snapshot_id === frozenSnapshotId && <Tag color="gold">当前</Tag>}
        </Space>
      </List.Item>}
      style={{ marginTop: 8 }}
    />}
    {phaseSnapshots.length === 0 && <Text type="secondary" style={{ display: "block", marginTop: 8, fontSize: 12 }}>暂无该阶段快照</Text>}
  </Card>;
}
