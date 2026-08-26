/** Phase-A CTA attribution panel: reads only confirmed, reviewed NAV. */

import React, { useCallback, useEffect, useState } from "react";
import { Alert, Button, Card, Collapse, Descriptions, Empty, Input, List, Popover, Segmented, Select, Space, Spin, Table, Tag, Typography, Upload, message } from "antd";
import { kbUpdateProduct, type CtaPhaseAResponse, type CtaPhaseBResponse, type CtaPhaseCResponse, type CtaPhaseDResponse, type KbProduct, type ModelApplicability, type StrategyDisclosure } from "../api";
import { getCtaPhaseA, getCtaPhaseB, getCtaPhaseC, getCtaPhaseD, getModelApplicability, importEquityFactorsFromAkShare, uploadAttributionFactorCsv } from "./ctaAttributionApi";
import { isResearchReady } from "./productWorkflow";

const { Text } = Typography;

function percent(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}

const FACTOR_EXPLANATIONS: Record<string, string> = {
  trend: "用公开趋势因子衡量产品收益与中长期趋势行情的统计关系。正向关联不等于产品真实持有趋势多头。",
  short_term_trend_20: "衡量产品对较短周期趋势的统计敏感度。",
  cross_section_mom: "衡量产品收益与品种间强弱分化动量的统计关系。",
  mean_reversion_5d: "衡量产品与短周期反转行情的统计关系。",
  term_structure_carry: "以已审核的近远月期限结构构造；未覆盖时不形成结论。",
  calendar_spread_momentum: "观察同一品种近远月价差的四周变化，并用相同合约的下一周价差收益检验；它是商品套利候选信号，不代表真实持仓。",
  volatility_state: "由滞后趋势因子的历史波动构造，用于衡量不同波动环境下的风险特征。",
};

function FactorLabel({ name, label }: { name: string; label: string }): React.JSX.Element {
  return <Popover content={FACTOR_EXPLANATIONS[name] ?? "公开市场代理下的统计因子，不代表实际持仓。"} title={label}>
    <Button type="link" size="small" style={{ padding: 0, height: "auto" }}>{label}</Button>
  </Popover>;
}

const DISCLOSURE_OPTIONS = {
  arbitrage_type: ["跨期", "跨品种", "跨市场", "期限结构", "混合"],
  holding_period: ["日内", "数日", "周度"],
  directional_exposure: ["低", "中", "高"],
};

function StrategyDisclosureCard({ product }: { product: KbProduct }): React.JSX.Element {
  const [disclosure, setDisclosure] = useState<StrategyDisclosure | null>(product.strategy_disclosure ?? null);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDisclosure(product.strategy_disclosure ?? null);
    setEditing(false);
  }, [product.id, product.strategy_disclosure]);

  const draft: StrategyDisclosure = disclosure ?? {
    arbitrage_type: null, markets_or_sectors: [], holding_period: null, directional_exposure: null,
  };
  const update = <K extends keyof StrategyDisclosure>(key: K, value: StrategyDisclosure[K]) => {
    setDisclosure({ ...draft, [key]: value });
  };
  const save = async (): Promise<void> => {
    setSaving(true);
    try {
      const updated = await kbUpdateProduct(product.id, { strategy_disclosure: draft });
      setDisclosure(updated.strategy_disclosure ?? draft);
      setEditing(false);
      message.success("策略信息已保存，仅用于研究口径说明和候选因子筛选");
    } catch (cause) {
      message.error(cause instanceof Error ? cause.message : "策略信息保存失败");
    } finally {
      setSaving(false);
    }
  };

  if (!editing && !disclosure) {
    return <Card size="small" title="策略最小披露" extra={<Button size="small" type="primary" onClick={() => setEditing(true)}>填写</Button>} />;
  }
  if (!editing) {
    const items = [
      ["套利类型", disclosure?.arbitrage_type], ["主要板块/品种", disclosure?.markets_or_sectors.join("、")],
      ["持有周期", disclosure?.holding_period], ["方向敞口", disclosure?.directional_exposure],
    ];
    return <Card size="small" title="策略最小披露" extra={<Button size="small" onClick={() => setEditing(true)}>编辑</Button>}>
      <Space size={[8, 8]} wrap>{items.map(([label, value]) => <Tag key={label} color={value ? "blue" : "default"}>{label}：{value || "未披露"}</Tag>)}</Space>
    </Card>;
  }
  return <Card size="small" title="策略最小披露" extra={<Space><Button size="small" onClick={() => { setDisclosure(product.strategy_disclosure ?? null); setEditing(false); }}>取消</Button><Button size="small" type="primary" loading={saving} onClick={() => void save()}>保存</Button></Space>}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
      <Select allowClear placeholder="套利类型" value={draft.arbitrage_type ?? undefined} options={DISCLOSURE_OPTIONS.arbitrage_type.map((value) => ({ value }))} onChange={(value) => update("arbitrage_type", value ?? null)} />
      <Select mode="tags" maxCount={6} placeholder="主要板块或品种" value={draft.markets_or_sectors} onChange={(value) => update("markets_or_sectors", value)} />
      <Select allowClear placeholder="持有周期" value={draft.holding_period ?? undefined} options={DISCLOSURE_OPTIONS.holding_period.map((value) => ({ value }))} onChange={(value) => update("holding_period", value ?? null)} />
      <Select allowClear placeholder="方向敞口" value={draft.directional_exposure ?? undefined} options={DISCLOSURE_OPTIONS.directional_exposure.map((value) => ({ value }))} onChange={(value) => update("directional_exposure", value ?? null)} />
    </div>
  </Card>;
}

function RiskModelOverview({ product, result }: { product: KbProduct; result: CtaPhaseAResponse | null }): React.JSX.Element {
  const factors = result?.factor_bundle.factors ?? [];
  const available = factors.filter((factor) => factor.status === "available");
  const missing = factors.filter((factor) => factor.status !== "available");
  return <Card size="small" title="CTA 风险模型" extra={<Tag color="success">已审核净值</Tag>}>
    <Descriptions size="small" column={{ xs: 1, sm: 2, lg: 4 }}>
      <Descriptions.Item label="模型范围">公开因子下的统计风格风险</Descriptions.Item>
      <Descriptions.Item label="净值样本">{product.reviewed_nav_count} 条</Descriptions.Item>
      <Descriptions.Item label="因子版本">{result?.factor_bundle.model_version ?? "读取基线结果后显示"}</Descriptions.Item>
      <Descriptions.Item label="因子覆盖">{result ? `${available.length}/${factors.length}` : "读取中"}</Descriptions.Item>
    </Descriptions>
    {missing.length > 0 && <Text type="secondary" style={{ fontSize: 12 }}>未覆盖：{missing.map((factor) => factor.label).join("、")}</Text>}
  </Card>;
}

function ModelApplicabilityCard({ applicability, product, onUploaded }: { applicability: ModelApplicability; product: KbProduct; onUploaded?: () => void }): React.JSX.Element {
  const poolLabel = { commodity_cta: "商品 CTA", equity_quant: "股票/股指", options_volatility: "期权/波动率", mixed_or_unconfirmed: "混合/待确认" }[applicability.product_pool];
  const status = applicability.status === "applicable" ? <Tag color="success">可解释</Tag>
    : applicability.status === "observe_only" ? <Tag color="warning">仅供观察</Tag> : <Tag>模型不适用</Tag>;
  return <Card size="small" title="模型适用性" extra={status}>
    <Descriptions size="small" column={{ xs: 1, sm: 2, lg: 4 }}>
      <Descriptions.Item label="产品池">{poolLabel}</Descriptions.Item>
      <Descriptions.Item label="识别口径">{applicability.classification_source === "confirmed_strategy" ? "已确认策略" : applicability.classification_source === "initial_name_label" ? "名称初步识别" : "资料不足"}</Descriptions.Item>
      <Descriptions.Item label="因子合同">{applicability.factor_contract ?? "尚未覆盖"}</Descriptions.Item>
      <Descriptions.Item label="已审核净值">{applicability.reviewed_nav_count} 条</Descriptions.Item>
    </Descriptions>
    <Text type="secondary">{applicability.reason}</Text>
    {applicability.factor_coverage && <Space wrap size={[6, 6]} style={{ display: "flex", marginTop: 8 }}>
      {applicability.factor_coverage.factors.map((factor) => <Tag key={factor.name} color={factor.status === "available" ? "success" : "default"}>{factor.label}：{factor.status === "available" ? `${factor.observations} 期` : "未覆盖"}</Tag>)}
    </Space>}
    {onUploaded && (applicability.product_pool === "equity_quant" || applicability.product_pool === "options_volatility") && <FactorCsvUpload pool={applicability.product_pool} start={product.nav_start} end={product.nav_end} onUploaded={onUploaded} />}
  </Card>;
}

function FactorCsvUpload({ pool, start, end, onUploaded }: { pool: "equity_quant" | "options_volatility"; start: string | null; end: string | null; onUploaded: () => void }): React.JSX.Element {
  const [source, setSource] = useState("人工审核 CSV");
  const [dataVersion, setDataVersion] = useState(new Date().toISOString().slice(0, 10));
  const [uploading, setUploading] = useState(false);
  const upload = async (file: File): Promise<void> => {
    setUploading(true);
    try {
      await uploadAttributionFactorCsv(pool, file, source, dataVersion);
      message.success("因子数据已保存，正在刷新覆盖状态");
      onUploaded();
    } catch (cause) {
      message.error(cause instanceof Error ? cause.message : "因子 CSV 上传失败");
    } finally {
      setUploading(false);
    }
  };
  const importFromAkShare = async (): Promise<void> => {
    if (!start || !end) return;
    setUploading(true);
    try {
      await importEquityFactorsFromAkShare(start, end);
      message.success("AKShare 因子已保存；价值和质量因子仍需独立数据源");
      onUploaded();
    } catch (cause) {
      message.error(cause instanceof Error ? cause.message : "AKShare 股票因子获取失败");
    } finally {
      setUploading(false);
    }
  };
  return <div style={{ marginTop: 12 }}>
    <Text strong style={{ fontSize: 12 }}>录入因子数据</Text>
    <Text type="secondary" style={{ display: "block", fontSize: 12, margin: "3px 0 8px" }}>CSV 列：observation_date、available_date、factor_name、value。数据不会用缺失值补齐。</Text>
    <Space wrap>
      <Input size="small" value={source} onChange={(event) => setSource(event.target.value)} placeholder="数据来源" style={{ width: 150 }} />
      <Input size="small" value={dataVersion} onChange={(event) => setDataVersion(event.target.value)} placeholder="数据版本" style={{ width: 120 }} />
      <Upload accept=".csv" showUploadList={false} beforeUpload={(file) => { void upload(file); return false; }}>
        <Button size="small" loading={uploading}>上传 CSV</Button>
      </Upload>
      {pool === "equity_quant" && <Button size="small" loading={uploading} disabled={!start || !end} onClick={() => void importFromAkShare()}>从 AKShare 获取基础代理</Button>}
    </Space>
  </div>;
}

function factorLabel(name: string): string {
  return {
    trend: "中长期趋势", short_term_trend_20: "短期动量", cross_section_mom: "截面动量",
    term_structure_carry: "期限结构 Carry", calendar_spread_momentum: "跨期价差动量", mean_reversion_5d: "短期反转", volatility_state: "波动率状态",
  }[name] ?? name;
}

function CandidateFactorGateCard({ gate }: { gate: CtaPhaseAResponse["candidate_factor_gate"] }): React.JSX.Element | null {
  if (!gate) return null;
  const status = gate.admitted ? <Tag color="success">已通过闸门</Tag> : gate.status === "insufficient" ? <Tag>样本不足</Tag> : <Tag color="warning">研究候选</Tag>;
  return <Card size="small" title="候选因子检验" extra={status}>
    <Text strong>{factorLabel(gate.candidate_name)}</Text>
    <Text type="secondary"> · {gate.conclusion}</Text>
    {gate.segments.length > 0 && <Table
      size="small" pagination={false} rowKey="segment" style={{ marginTop: 10 }} dataSource={gate.segments}
      columns={[
        { title: "测试段", dataIndex: "segment", width: 72 },
        { title: "样本外区间", render: (_: unknown, row: NonNullable<CtaPhaseAResponse["candidate_factor_gate"]>["segments"][number]) => `${row.test_start_date} 至 ${row.test_end_date}` },
        { title: "基线 R²", dataIndex: "baseline_oos_r2", render: (value: number) => value.toFixed(3) },
        { title: "加入候选后 R²", dataIndex: "candidate_oos_r2", render: (value: number) => value.toFixed(3) },
        { title: "增量 R²", dataIndex: "incremental_oos_r2", render: (value: number) => <Tag color={value >= 0.01 ? "green" : "default"}>{value.toFixed(3)}</Tag> },
      ]}
    />}
  </Card>;
}

function factorDisplayName(name: string, fallback: string): string {
  return factorLabel(name) === name ? fallback : factorLabel(name);
}

function RiskConclusionCard({ baseline, dynamic, regime }: {
  baseline: CtaPhaseAResponse | null;
  dynamic: CtaPhaseBResponse | null;
  regime: CtaPhaseCResponse | null;
}): React.JSX.Element | null {
  if (!baseline) return null;
  const mainExposure = baseline.attribution.factor_exposure.slice().sort((left, right) => Math.abs(right.beta) - Math.abs(left.beta))[0];
  const mainRisk = baseline.attribution.euler_risk_contribution
    .filter((row) => row.component_risk_contribution != null)
    .sort((left, right) => Math.abs(right.component_risk_contribution ?? 0) - Math.abs(left.component_risk_contribution ?? 0))[0];
  const drift = dynamic?.kalman_beta.factor_names
    .map((name) => ({ name, range: dynamic.kalman_beta.beta_range[name] }))
    .filter((item): item is { name: string; range: number } => item.range != null)
    .sort((left, right) => right.range - left.range)[0];
  const crisis = regime?.regime_attribution.regimes.find((item) => item.state === "crisis");
  const available = baseline.factor_bundle.factors.filter((factor) => factor.status === "available").length;
  const conclusions = [
    ["主要市场特征", mainExposure ? `${factorDisplayName(mainExposure.factor_name, mainExposure.display_name)}${mainExposure.hac_p_value < 0.05 ? "，统计关系较明确" : "，仍需观察"}` : "当前没有可估计的公开因子关联。"],
    ["主要风险来源", mainRisk ? factorDisplayName(mainRisk.factor_name, mainRisk.display_name) : "当前无法拆分单因子风险来源。"],
    ["特征稳定性", drift ? `${factorLabel(drift.name)}的关联变化相对更大` : "暂无关联变化结果。"],
    ["压力环境表现", crisis ? crisis.status === "available" ? `${crisis.observation_count} 期样本，期间收益 ${percent(crisis.conditional_return)}` : `样本 ${crisis.observation_count} 期，暂不判断` : "暂无压力环境结果。"],
  ];
  return <Card size="small" title="风险结论" extra={<Tag color={available === baseline.factor_bundle.factors.length ? "success" : "warning"}>因子覆盖 {available}/{baseline.factor_bundle.factors.length}</Tag>}>
    <List size="small" split={false} dataSource={conclusions} renderItem={([label, value]) => <List.Item style={{ padding: "3px 0" }}><Text strong style={{ minWidth: 92 }}>{label}</Text><Text>{value}</Text></List.Item>} />
  </Card>;
}

export default function CtaAttributionPanel({ product }: { product: KbProduct | undefined }): React.JSX.Element {
  const [result, setResult] = useState<CtaPhaseAResponse | null>(null);
  const [dynamicResult, setDynamicResult] = useState<CtaPhaseBResponse | null>(null);
  const [regimeResult, setRegimeResult] = useState<CtaPhaseCResponse | null>(null);
  const [nonlinearResult, setNonlinearResult] = useState<CtaPhaseDResponse | null>(null);
  const [view, setView] = useState<"static" | "dynamic" | "regime" | "nonlinear">("static");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [applicability, setApplicability] = useState<ModelApplicability | null>(null);

  const ready = Boolean(product && isResearchReady(product) && product.reviewed_nav_count >= 20);

  const load = useCallback(async () => {
    if (!product || !ready) return;
    setLoading(true);
    setError(null);
    try {
      const nextApplicability = await getModelApplicability(product.id);
      setApplicability(nextApplicability);
      if (nextApplicability.status !== "applicable" && nextApplicability.status !== "observe_only") return;
      if (view === "static") {
        setResult(await getCtaPhaseA(product.id));
      } else if (view === "dynamic") {
        setDynamicResult(await getCtaPhaseB(product.id));
      } else {
        if (view === "regime") setRegimeResult(await getCtaPhaseC(product.id));
        else setNonlinearResult(await getCtaPhaseD(product.id));
      }
    } catch (cause) {
      setResult(null);
      setDynamicResult(null);
      setRegimeResult(null);
      setNonlinearResult(null);
      setError(cause instanceof Error ? cause.message : "CTA 已审核净值归因失败");
    } finally {
      setLoading(false);
    }
  }, [product, ready, view]);

  useEffect(() => { void load(); }, [load]);

  if (!product) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="未选择产品" />;
  const disclosureCard = <StrategyDisclosureCard product={product} />;
  if (!ready) return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    {disclosureCard}
    <Alert type="info" showIcon message="CTA 风险模型样本不足" description={`${product.research_workflow?.blocking_reasons[0] ?? "研究状态未满足"}；已审核净值 ${product.reviewed_nav_count} 条。`} />
  </Space>;
  if (applicability && applicability.status !== "applicable" && applicability.status !== "observe_only") return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    {disclosureCard}
    <ModelApplicabilityCard applicability={applicability} product={product} onUploaded={() => void load()} />
    <Alert type="info" showIcon message="CTA 归因未运行" />
  </Space>;
  if (loading) return <div style={{ padding: 42, textAlign: "center" }}><Spin tip="正在读取已审核净值和公开因子…" /></div>;
  if (error) return <Alert type="error" showIcon message={error} />;
  if (view === "dynamic") {
    return <Space direction="vertical" size={12} style={{ width: "100%" }}>
      {disclosureCard}
      {applicability && <ModelApplicabilityCard applicability={applicability} product={product} onUploaded={() => void load()} />}
      <ViewSelector value={view} onChange={setView} />
      <RiskModelOverview product={product} result={result} />
      <RiskConclusionCard baseline={result} dynamic={dynamicResult} regime={regimeResult} />
      {dynamicResult ? <DynamicBetaView result={dynamicResult} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无因子关联变化结果" />}
    </Space>;
  }
  if (view === "regime") {
    return <Space direction="vertical" size={12} style={{ width: "100%" }}>
      {disclosureCard}
      {applicability && <ModelApplicabilityCard applicability={applicability} product={product} onUploaded={() => void load()} />}
      <ViewSelector value={view} onChange={setView} />
      <RiskModelOverview product={product} result={result} />
      <RiskConclusionCard baseline={result} dynamic={dynamicResult} regime={regimeResult} />
      {regimeResult ? <RegimeView result={regimeResult} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无状态归因结果" />}
    </Space>;
  }
  if (view === "nonlinear") {
    return <Space direction="vertical" size={12} style={{ width: "100%" }}>
      {disclosureCard}
      {applicability && <ModelApplicabilityCard applicability={applicability} product={product} onUploaded={() => void load()} />}
      <ViewSelector value={view} onChange={setView} />
      <RiskModelOverview product={product} result={result} />
      <RiskConclusionCard baseline={result} dynamic={dynamicResult} regime={regimeResult} />
      {nonlinearResult ? <NonlinearView result={nonlinearResult} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无非线性增量结果" />}
    </Space>;
  }
  if (!result) return <Space direction="vertical" size={12} style={{ width: "100%" }}>{disclosureCard}{applicability && <ModelApplicabilityCard applicability={applicability} product={product} onUploaded={() => void load()} />}<ViewSelector value={view} onChange={setView} /><RiskModelOverview product={product} result={result} /><RiskConclusionCard baseline={result} dynamic={dynamicResult} regime={regimeResult} /></Space>;

  const returnsByFactor = new Map(result.attribution.return_contribution.map((row) => [row.factor_name, row]));
  const risksByFactor = new Map(result.attribution.euler_risk_contribution.map((row) => [row.factor_name, row]));
  const factors = result.attribution.factor_exposure.map((row) => ({ ...row, contribution: returnsByFactor.get(row.factor_name)?.mean_return_contribution, risk: risksByFactor.get(row.factor_name)?.component_risk_contribution }));
  const hasUsableFit = result.baseline.r_squared >= 0.1 && result.baseline.validation?.status !== "not_applicable";
  const reconciliation = result.attribution.return_reconciliation;

  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    {disclosureCard}
    {applicability && <ModelApplicabilityCard applicability={applicability} product={product} onUploaded={() => void load()} />}
    <ViewSelector value={view} onChange={setView} />
    <RiskModelOverview product={product} result={result} />
    <RiskConclusionCard baseline={result} dynamic={dynamicResult} regime={regimeResult} />
    <Alert type="success" showIcon message="已审核净值 · 只读统计归因" description={`${result.data_contract.reviewed_nav_count} 个${result.frequency}净值点，${result.performance_path.start_date} 至 ${result.performance_path.end_date}。未触发资料解析。`} />
    <Descriptions size="small" bordered column={{ xs: 1, sm: 2, lg: 3 }}>
      <Descriptions.Item label="CAGR">{percent(result.performance_path.cagr)}</Descriptions.Item>
      <Descriptions.Item label="最大回撤">{percent(result.performance_path.maximum_drawdown)}</Descriptions.Item>
      <Descriptions.Item label="Calmar">{result.performance_path.calmar?.toFixed(2) ?? "—"}</Descriptions.Item>
      <Descriptions.Item label="最差 5%平均亏损">{percent(result.performance_path.expected_shortfall_5pct)}</Descriptions.Item>
      <Descriptions.Item label="最长回撤">{result.performance_path.drawdown_duration_max_periods} 期</Descriptions.Item>
      <Descriptions.Item label="恢复状态"><Tag color={result.performance_path.recovery_completed ? "success" : "warning"}>{result.performance_path.recovery_completed ? "已恢复" : "尚未恢复"}</Tag></Descriptions.Item>
    </Descriptions>
    <Alert type={result.baseline.validation?.status === "applicable" ? "success" : "info"} showIcon message={result.baseline.validation?.status === "applicable" ? "归因结论可供参考" : "归因结论仅供观察"} description={result.baseline.validation?.status === "applicable" ? "样本外检验通过，可将市场特征作为辅助判断。" : "当前数据尚不足以形成强化归因结论。"} />
    <Collapse size="small" items={[{
      key: "research-details",
      label: "查看数据与方法明细",
      children: <Space direction="vertical" size={12} style={{ width: "100%" }}>
        <Descriptions size="small" bordered column={{ xs: 1, sm: 2, lg: 4 }}>
          <Descriptions.Item label="样本内解释度">{result.baseline.r_squared.toFixed(3)}</Descriptions.Item>
          <Descriptions.Item label="样本外解释度">{result.baseline.out_of_sample.r_squared?.toFixed(3) ?? "—"}</Descriptions.Item>
          <Descriptions.Item label="未解释收益">{hasUsableFit ? percent(result.baseline.annualized_alpha_candidate) : "不作结论"}</Descriptions.Item>
          <Descriptions.Item label="独立样本外区间">{result.baseline.out_of_sample.segments?.length ?? "—"}</Descriptions.Item>
        </Descriptions>
        <Table size="small" rowKey="factor_name" pagination={false} dataSource={factors} scroll={{ x: 620 }} columns={[
          { title: "市场特征", dataIndex: "display_name", render: (value: string, row: CtaPhaseAResponse["attribution"]["factor_exposure"][number]) => <FactorLabel name={row.factor_name} label={factorDisplayName(row.factor_name, value)} /> },
          { title: "关联方向", dataIndex: "beta", render: (value: number) => <Tag color={value >= 0 ? "blue" : "orange"}>{value >= 0 ? "同向" : "反向"} {Math.abs(value).toFixed(2)}</Tag> },
          { title: "统计可信度", dataIndex: "hac_p_value", render: (value: number) => value < 0.05 ? <Tag color="green">p {value < 0.001 ? "<0.001" : `=${value.toFixed(3)}`}（&lt;0.05，显著）</Tag> : <Tag>p ={value.toFixed(3)}（≥0.05，不显著）</Tag> },
          ...(hasUsableFit ? [
            { title: "平均期收益影响", dataIndex: "contribution", render: (value: number | undefined) => value == null ? "—" : percent(value) },
          ] : []),
        ]} />
        <CandidateFactorGateCard gate={result.candidate_factor_gate} />
        {reconciliation && <Descriptions size="small" bordered column={4} title="平均期收益对账">
          <Descriptions.Item label="产品平均收益">{percent(reconciliation.mean_product_return)}</Descriptions.Item>
          <Descriptions.Item label="因子解释收益">{percent(reconciliation.mean_factor_explained_return)}</Descriptions.Item>
          <Descriptions.Item label="截距收益">{percent(reconciliation.mean_intercept_return)}</Descriptions.Item>
          <Descriptions.Item label="对账残差">{percent(reconciliation.mean_residual_return)}</Descriptions.Item>
        </Descriptions>}
      </Space>,
    }]} />
  </Space>;
}

function ViewSelector({ value, onChange }: { value: "static" | "dynamic" | "regime" | "nonlinear"; onChange: (value: "static" | "dynamic" | "regime" | "nonlinear") => void }): React.JSX.Element {
  return <Segmented
    size="small"
    value={value}
    onChange={(next) => onChange(next as "static" | "dynamic" | "regime" | "nonlinear")}
    options={[{ label: "基础归因", value: "static" }, { label: "关联变化", value: "dynamic" }, { label: "市场环境表现", value: "regime" }, { label: "情景验证", value: "nonlinear" }]}
  />;
}

function DynamicBetaView({ result }: { result: CtaPhaseBResponse }): React.JSX.Element {
  const labels: Record<string, string> = {
    trend: "趋势",
    term_structure_carry: "真实 Carry / 期限结构",
    short_term_trend_20: "短趋势",
    mean_reversion_5d: "反转",
    volatility_state: "波动率状态",
  };
  const latest = result.kalman_beta.paths.at(-1);
  const latestRows = result.kalman_beta.factor_names.map((name) => ({
    key: name,
    factor_name: name,
    display_name: labels[name] ?? name,
    long_run_beta: result.kalman_beta.long_run_style_beta[name],
    latest_beta: latest?.betas[name],
    standard_error: latest?.beta_standard_errors[name],
    beta_range: result.kalman_beta.beta_range[name],
  }));
  const latestPath = result.kalman_beta.paths.slice(-80);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const chartWidth = 720;
  const chartHeight = 170;
  const plotLeft = 48;
  const plotRight = 12;
  const plotTop = 8;
  const plotBottom = 18;
  const plotWidth = chartWidth - plotLeft - plotRight;
  const plotHeight = chartHeight - plotTop - plotBottom;
  const values = latestPath.flatMap((point) => result.kalman_beta.factor_names.map((name) => point.betas[name] ?? 0));
  const rawMinValue = values.length ? Math.min(...values, 0) : -1;
  const rawMaxValue = values.length ? Math.max(...values, 0) : 1;
  const valuePadding = Math.max((rawMaxValue - rawMinValue) * 0.08, 0.05);
  const minValue = rawMinValue - valuePadding;
  const maxValue = rawMaxValue + valuePadding;
  const span = Math.max(maxValue - minValue, 1e-8);
  const ticks = Array.from({ length: 5 }, (_, index) => minValue + span * index / 4);
  const xPosition = (index: number) => latestPath.length <= 1 ? plotLeft : plotLeft + index / (latestPath.length - 1) * plotWidth;
  const yPosition = (value: number) => plotTop + plotHeight - (value - minValue) / span * plotHeight;
  const formatCoefficient = (value: number | null | undefined): string => (
    value == null || !Number.isFinite(value) ? "—" : value.toPrecision(3)
  );
  const updateHoveredPoint = (event: React.MouseEvent<SVGRectElement>) => {
    const bounds = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
    if (!bounds) return;
    const x = (event.clientX - bounds.left) / bounds.width * chartWidth;
    const index = Math.round((x - plotLeft) / plotWidth * (latestPath.length - 1));
    setHoveredIndex(Math.max(0, Math.min(latestPath.length - 1, index)));
  };
  const hoveredPoint = hoveredIndex == null ? null : latestPath[hoveredIndex];
  const stabilitySummary = (window: CtaPhaseBResponse["rolling_beta"]["windows"][number]): string => {
    if (window.status === "insufficient") return `近 ${window.window} 期：样本不足`;
    const factors = Object.entries(window.summary ?? {}).map(([name, summary]) =>
      `${labels[name] ?? name} ${summary.sign_consistency == null ? "—" : `${(summary.sign_consistency * 100).toFixed(0)}%`}`,
    );
    return `近 ${window.window} 期：${factors.join("，")}`;
  };

  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <Alert type="success" showIcon message="因子关联变化" description={`${result.alignment.aligned_observation_count} 个已对齐观测；逐期估计只使用当时可得数据，不使用未来数据。`} />
    <Descriptions size="small" bordered column={{ xs: 1, sm: 2, lg: 3 }}>
      <Descriptions.Item label="对齐样本">{result.alignment.aligned_observation_count}</Descriptions.Item>
      <Descriptions.Item label="起始日期">{result.alignment.start_date}</Descriptions.Item>
      <Descriptions.Item label="结束日期">{result.alignment.end_date}</Descriptions.Item>
      <Descriptions.Item label="关联变化累计影响">{result.kalman_beta.dynamic_timing_contribution_total.toFixed(6)}</Descriptions.Item>
      <Descriptions.Item label="估计波动度">{result.kalman_beta.innovation_root_mean_square.toFixed(6)}</Descriptions.Item>
      <Descriptions.Item label="未来数据"><Tag color="green">未使用</Tag></Descriptions.Item>
    </Descriptions>
    <Card size="small" title="因子关联系数变化（最近 80 个观测）">
      {latestPath.length > 1 ? <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} width="100%" height="180" role="img" aria-label="因子关联系数变化，纵轴为关联系数">
          {ticks.map((tick) => <g key={tick}>
            <line x1={plotLeft} y1={yPosition(tick)} x2={chartWidth - plotRight} y2={yPosition(tick)} stroke={Math.abs(tick) < span / 100 ? "#8c8c8c" : "#f0f0f0"} strokeDasharray={Math.abs(tick) < span / 100 ? "4 4" : undefined} />
            <text x={plotLeft - 7} y={yPosition(tick) + 4} textAnchor="end" fill="#8c8c8c" fontSize="11">{tick.toFixed(2)}</text>
          </g>)}
          <line x1={plotLeft} y1={plotTop} x2={plotLeft} y2={plotTop + plotHeight} stroke="#bfbfbf" />
          {result.kalman_beta.factor_names.map((name, factorIndex) => {
            const points = latestPath.map((point, index) => `${xPosition(index)},${yPosition(point.betas[name] ?? 0)}`).join(" ");
            const color = ["#1677ff", "#13c2c2", "#722ed1", "#fa8c16"][factorIndex % 4];
            return <g key={name}>
              <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
            </g>;
          })}
          {hoveredPoint && hoveredIndex != null && <g pointerEvents="none">
            <line x1={xPosition(hoveredIndex)} y1={plotTop} x2={xPosition(hoveredIndex)} y2={plotTop + plotHeight} stroke="#595959" strokeDasharray="3 3" />
            {result.kalman_beta.factor_names.map((name, factorIndex) => <circle key={name} cx={xPosition(hoveredIndex)} cy={yPosition(hoveredPoint.betas[name] ?? 0)} r="3" fill={["#1677ff", "#13c2c2", "#722ed1", "#fa8c16"][factorIndex % 4]} />)}
          </g>}
          <rect
            x={plotLeft}
            y={plotTop}
            width={plotWidth}
            height={plotHeight}
            fill="transparent"
            style={{ cursor: "crosshair" }}
            onMouseMove={updateHoveredPoint}
            onMouseLeave={() => setHoveredIndex(null)}
          />
        </svg>
        <div aria-label="图例" style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "6px 22px", marginTop: 4, fontSize: 12, color: "#1f1f1f" }}>
          {result.kalman_beta.factor_names.map((name, factorIndex) => <span key={name} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span aria-hidden="true" style={{ width: 18, borderTop: `2px solid ${["#1677ff", "#13c2c2", "#722ed1", "#fa8c16"][factorIndex % 4]}` }} />
            {labels[name] ?? name}
          </span>)}
        </div>
        {hoveredPoint && hoveredIndex != null && <div
          role="status"
          style={{
            position: "absolute", top: 10, zIndex: 1, minWidth: 172, padding: "8px 10px",
            borderRadius: 6, background: "rgba(255, 255, 255, 0.96)", border: "1px solid #d9d9d9",
            boxShadow: "0 3px 10px rgba(0, 0, 0, 0.12)", fontSize: 12, lineHeight: 1.7, pointerEvents: "none",
            ...(hoveredIndex / (latestPath.length - 1) > 0.72 ? { right: 8 } : { left: `${xPosition(hoveredIndex) / chartWidth * 100}%` }),
          }}
        >
          <Text strong style={{ fontSize: 12 }}>{hoveredPoint.date}</Text>
          {result.kalman_beta.factor_names.map((name) => <div key={name}>{labels[name] ?? name}：{formatCoefficient(hoveredPoint.betas[name])}</div>)}
        </div>}
      </div> : <Text type="secondary">动态路径不足以绘图。</Text>}
    </Card>
    <Table
      size="small" rowKey="key" pagination={false} dataSource={latestRows} scroll={{ x: 720 }}
      columns={[
        { title: "因子", dataIndex: "display_name" },
        { title: "长期关联系数", dataIndex: "long_run_beta", render: (value: number) => value?.toFixed(4) ?? "—" },
        { title: "最新关联系数", dataIndex: "latest_beta", render: (value: number) => value?.toFixed(4) ?? "—" },
        { title: "最新标准误", dataIndex: "standard_error", render: (value: number) => value?.toFixed(4) ?? "—" },
        { title: "变化范围", dataIndex: "beta_range", render: (value: number) => value?.toFixed(4) ?? "—" },
      ]}
    />
    <Card size="small" title="关联稳定性">
      <Space direction="vertical" size={4} style={{ width: "100%" }}>
        <Text type="secondary">方向稳定比例越高，表示该因子的统计关联越稳定。</Text>
        {result.rolling_beta.windows.map((window) => <Text key={window.window}>{stabilitySummary(window)}</Text>)}
      </Space>
    </Card>
  </Space>;
}

function RegimeView({ result }: { result: CtaPhaseCResponse }): React.JSX.Element {
  const stateLabels: Record<string, string> = { normal: "平稳环境", crisis: "压力环境", whipsaw: "震荡环境" };
  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <Alert type="info" showIcon message="不同市场环境下的表现" description={`这里的市场环境是依据${factorLabel(result.state_proxy.market_proxy_factor)}的历史涨跌和波动划分的统计情景，不是具体交易市场，也不代表产品仓位。`} />
    <Descriptions size="small" bordered column={4}>
      <Descriptions.Item label="平稳环境">{result.regime_rules.state_counts.normal ?? 0} 期</Descriptions.Item>
      <Descriptions.Item label="压力环境">{result.regime_rules.state_counts.crisis ?? 0} 期</Descriptions.Item>
      <Descriptions.Item label="震荡环境">{result.regime_rules.state_counts.whipsaw ?? 0} 期</Descriptions.Item>
      <Descriptions.Item label="划分方式">历史市场表现规则</Descriptions.Item>
    </Descriptions>
    <Table
      size="small" rowKey="state" pagination={false} scroll={{ x: 980 }} dataSource={result.regime_attribution.regimes}
      columns={[
        { title: "市场环境", dataIndex: "state", render: (value: string) => <Tag color={value === "crisis" ? "red" : value === "whipsaw" ? "orange" : "blue"}>{stateLabels[value] ?? value}</Tag> },
        { title: "样本数", dataIndex: "observation_count" },
        { title: "条件收益", dataIndex: "conditional_return", render: (value: number | null) => percent(value) },
        { title: "最差 5%平均亏损", dataIndex: "expected_shortfall_5pct", width: 130, render: (value: number | null) => percent(value) },
        { title: "条件相关", dataIndex: "conditional_correlation", render: (value: number | null) => value == null ? "—" : value.toFixed(3) },
        { title: "压力期未解释收益", dataIndex: "crisis_alpha_candidate", render: (value: number | null) => percent(value) },
        { title: "估计状态", dataIndex: "status", render: (value: string) => <Tag color={value === "available" ? "success" : "warning"}>{value === "available" ? "可估计" : "样本不足"}</Tag> },
      ]}
    />
    {result.regime_attribution.regimes.filter((item) => item.status === "available").map((item) => <Card key={item.state} size="small" title={`${stateLabels[item.state]}下的因子关联`}>
      <Space wrap>{Object.entries(item.conditional_beta ?? {}).map(([factor, value]) => <Tag key={factor}>{factorLabel(factor)}：统计系数 {value.toFixed(4)}</Tag>)}</Space>
    </Card>)}
    {result.warnings.length > 0 && <Text type="secondary">部分环境样本不足时，不展示对应统计结果。</Text>}
  </Space>;
}

function NonlinearView({ result }: { result: CtaPhaseDResponse }): React.JSX.Element {
  const increment = result.nonlinear_increment;
  const checks = [
    ...(result.scenario_checks ?? []),
    {
      key: "momentum_volatility",
      title: "趋势与波动率",
      description: "检验不同波动水平下，趋势特征是否出现稳定变化。",
      status: increment.status,
      stable: increment.summary.stable_improvement,
      conclusion: increment.summary.conclusion,
      evaluated_segments: increment.summary.evaluated_delta_count,
      positive_fraction: increment.summary.positive_delta_fraction,
      mean_r2_delta: increment.summary.mean_r2_delta,
      details: increment,
    },
  ];
  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <Card size="small" title="CTA 情景验证">
      <Text type="secondary">分别验证产品在不同趋势、期限结构、波动与自身回撤情景下，是否呈现可重复的特征。结论只采用未参与拟合的后续数据。</Text>
      <List
        style={{ marginTop: 8 }}
        split={false}
        dataSource={checks}
        renderItem={(check) => {
          const usable = check.status === "available";
          const improved = check.positive_fraction == null ? "—" : `${(check.positive_fraction * 100).toFixed(0)}%`;
          return <List.Item>
            <Card
              size="small"
              type="inner"
              title={check.title}
              style={{ width: "100%" }}
              extra={<Tag color={!usable ? "warning" : check.stable ? "success" : "default"}>{!usable ? "样本不足" : check.stable ? "存在稳定证据" : "未发现稳定证据"}</Tag>}
            >
              <Space direction="vertical" size={4} style={{ width: "100%" }}>
                <Text>{check.description}</Text>
                <Text strong>{check.conclusion}</Text>
                {usable && <Text type="secondary">{check.evaluated_segments} 个独立测试区间中，{improved} 的区间有所改善。</Text>}
                {check.details && <Collapse
                  size="small"
                  ghost
                  items={[{
                    key: "details",
                    label: "查看研究明细",
                    children: <Table
                      size="small"
                      rowKey="segment"
                      pagination={false}
                      dataSource={check.details.segments}
                      scroll={{ x: 620 }}
                      columns={[
                        { title: "测试区间", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => `${row.test_start_date} 至 ${row.test_end_date}` },
                        { title: "基础解释度", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => row.linear.oos_r2 == null ? "—" : row.linear.oos_r2.toFixed(3) },
                        { title: "加入规则后", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => row.nonlinear.oos_r2 == null ? "—" : row.nonlinear.oos_r2.toFixed(3) },
                        { title: "变化", render: (_: unknown, row: CtaPhaseDResponse["nonlinear_increment"]["segments"][number]) => row.r2_delta == null ? "—" : row.r2_delta.toFixed(3) },
                      ]}
                    />,
                  }]}
                />}
              </Space>
            </Card>
          </List.Item>;
        }}
      />
    </Card>
  </Space>;
}
