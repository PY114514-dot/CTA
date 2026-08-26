/** One-off, read-only marginal-risk comparison for an explicit portfolio. */

import React, { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Descriptions, Empty, InputNumber, Select, Space, Spin, Table, Tag, Typography, message } from "antd";
import { getCtaAllocationScore, kbGetNavSeries, kbListProducts, type CtaAllocationScore, type CtaProductScoreRequestPayload, type DataFrequency, type KbProduct } from "../api";
import { getCtaPhaseA } from "./ctaAttributionApi";
import { isResearchReady } from "./productWorkflow";

const { Text } = Typography;

function frequencyOf(value: string | null): DataFrequency {
  return value === "daily" || value === "monthly" ? value : "weekly";
}

function percent(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}

export default function PortfolioRiskMapPanel({ selectedProductIds }: { selectedProductIds: string[] }): React.JSX.Element {
  const [products, setProducts] = useState<KbProduct[]>([]);
  const [candidateId, setCandidateId] = useState<string>();
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [candidateWeight, setCandidateWeight] = useState(0.1);
  const [result, setResult] = useState<CtaAllocationScore | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    kbListProducts({ limit: 1000 }).then((items) => {
      if (!active) return;
      const selected = items.filter((item) => selectedProductIds.includes(item.id) && isResearchReady(item));
      setProducts(selected);
      setCandidateId((current) => selected.some((item) => item.id === current) ? current : selected.at(-1)?.id);
      setWeights((current) => Object.fromEntries(selected.filter((item) => item.id !== selected.at(-1)?.id).map((item) => [item.id, current[item.id] ?? 0])));
    }).catch(() => { if (active) setProducts([]); });
    return () => { active = false; };
  }, [selectedProductIds]);

  const candidateFrequency = frequencyOf(products.find((item) => item.id === candidateId)?.nav_frequency ?? null);
  const compatibleProducts = useMemo(() => products.filter((item) => frequencyOf(item.nav_frequency) === candidateFrequency), [candidateFrequency, products]);
  const currentProducts = useMemo(() => compatibleProducts.filter((item) => item.id !== candidateId), [candidateId, compatibleProducts]);
  const currentWeightTotal = currentProducts.reduce((sum, item) => sum + (weights[item.id] ?? 0), 0);
  const canEvaluate = Boolean(candidateId && currentProducts.length > 0 && Math.abs(currentWeightTotal - 1) < 1e-6);

  const evaluate = async (): Promise<void> => {
    if (!candidateId || !canEvaluate) return;
    setLoading(true);
    try {
      const series = await Promise.all(compatibleProducts.map(async (product) => {
        const [nav, phaseA] = await Promise.all([
          kbGetNavSeries(product.id, true),
          getCtaPhaseA(product.id).catch(() => null),
        ]);
        return {
        product,
        nav,
        factorExposures: Object.fromEntries(phaseA?.attribution.factor_exposure.map((factor) => [factor.factor_name, factor.beta]) ?? []),
      }; }));
      const request: CtaProductScoreRequestPayload = {
        products: series.map(({ product, nav, factorExposures }) => ({
          product_id: product.id,
          product_name: product.standard_name,
          frequency: frequencyOf(product.nav_frequency),
          strategy: product.strategy,
          factor_exposures: factorExposures,
          nav_points: nav.filter((point) => point.review_status === "reviewed").map((point) => ({ observation_date: point.observation_date, net_asset_value: point.nav })),
        })),
        current_portfolio: currentProducts.map((product) => ({ product_id: product.id, weight: weights[product.id] ?? 0 })),
        candidate_product_id: candidateId,
        candidate_weight: candidateWeight,
      };
      const response = await getCtaAllocationScore(request);
      setResult(response.allocation);
    } catch (cause) {
      message.error(cause instanceof Error ? cause.message : "组合风险试算失败");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  if (products.length < 2) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="先在产品列表中选择至少两只可研究产品，再进行组合风险试算" />;
  return <Space direction="vertical" size={12} style={{ width: "100%" }}>
    <Alert type="info" showIcon message="组合风险地图" description="只比较指定候选产品加入当前组合后的边际风险变化；权重仅用于本次试算，不会自动保存、推荐或写入配置。" />
    <Card size="small" title="本次试算输入" extra={<Button type="primary" size="small" loading={loading} disabled={!canEvaluate} onClick={() => void evaluate()}>计算边际风险</Button>}>
        <Space direction="vertical" size={8} style={{ width: "100%" }}>
        <Space wrap><Text>候选产品</Text><Select value={candidateId} style={{ minWidth: 240 }} options={products.map((item) => ({ value: item.id, label: item.standard_name }))} onChange={(value) => { setCandidateId(value); setResult(null); }} /><Text>加入权重</Text><InputNumber min={0.01} max={0.99} step={0.01} value={candidateWeight} onChange={(value) => setCandidateWeight(Number(value ?? 0))} /></Space>
        {currentProducts.map((product) => <Space key={product.id} wrap><Text style={{ minWidth: 220 }}>{product.standard_name}</Text><InputNumber min={0} max={1} step={0.01} value={weights[product.id] ?? 0} onChange={(value) => { setWeights((current) => ({ ...current, [product.id]: Number(value ?? 0) })); setResult(null); }} /><Text type="secondary">当前组合权重</Text></Space>)}
        {compatibleProducts.length < products.length && <Text type="secondary">已排除 {products.length - compatibleProducts.length} 只不同频率产品；组合风险地图仅比较 {candidateFrequency === "monthly" ? "月频" : candidateFrequency === "daily" ? "日频" : "周频"}净值。</Text>}
        <Text type={Math.abs(currentWeightTotal - 1) < 1e-6 ? "secondary" : "danger"}>当前组合权重合计：{(currentWeightTotal * 100).toFixed(0)}%{Math.abs(currentWeightTotal - 1) < 1e-6 ? "" : "（需为 100% 才能计算）"}</Text>
      </Space>
    </Card>
    {result && <RiskMapResult result={result} products={products} candidateId={candidateId} />}
  </Space>;
}

function RiskMapResult({ result, products, candidateId }: { result: CtaAllocationScore; products: KbProduct[]; candidateId: string | undefined }): React.JSX.Element {
  if (result.status !== "available") return <Alert type="warning" showIcon message={result.reason ?? "共同样本不足，无法生成组合风险地图"} />;
  const candidate = products.find((item) => item.id === candidateId)?.standard_name ?? result.candidate_product_id;
  const diversification = result.metrics.full_sample_correlation == null ? "未覆盖" : Math.abs(result.metrics.full_sample_correlation) < 0.2 ? "较强" : Math.abs(result.metrics.full_sample_correlation) < 0.5 ? "中等" : "较弱";
  const volatility = result.metrics.marginal_volatility_change == null ? "未覆盖" : result.metrics.marginal_volatility_change < -0.0002 ? "改善" : result.metrics.marginal_volatility_change > 0.0002 ? "恶化" : "基本不变";
  const drawdown = result.metrics.marginal_maximum_drawdown_change == null ? "未覆盖" : result.metrics.marginal_maximum_drawdown_change < -0.0002 ? "改善" : result.metrics.marginal_maximum_drawdown_change > 0.0002 ? "恶化" : "基本不变";
  const strength = result.metrics.factor_exposure_overlap == null ? "中低（因子重合未覆盖）" : (result.components.confidence ?? 0) >= 75 ? "中高" : "中";
  return <Card size="small" title={`组合风险地图 · ${candidate}`}>
    <Alert type={drawdown === "恶化" ? "warning" : "info"} showIcon message={`分散化 ${diversification} · 波动 ${volatility} · 回撤 ${drawdown}`} description={`结论强度：${strength}。配置分仅作为展开明细，不代替这些边际风险判断。`} style={{ marginBottom: 10 }} />
    <Descriptions size="small" bordered column={{ xs: 1, sm: 2, lg: 3 }}>
      <Descriptions.Item label="全样本相关性">{result.metrics.full_sample_correlation?.toFixed(3) ?? "—"}</Descriptions.Item>
      <Descriptions.Item label="加入后波动变化">{percent(result.metrics.marginal_volatility_change)}</Descriptions.Item>
      <Descriptions.Item label="加入后最大回撤变化">{percent(result.metrics.marginal_maximum_drawdown_change)}</Descriptions.Item>
      <Descriptions.Item label="危机期互补性">{percent(result.metrics.crisis_complementarity)}</Descriptions.Item>
      <Descriptions.Item label="震荡期互补性">{percent(result.metrics.whipsaw_complementarity)}</Descriptions.Item>
      <Descriptions.Item label="因子暴露重合度">{result.metrics.factor_exposure_overlap == null ? "未覆盖" : `${(result.metrics.factor_exposure_overlap * 100).toFixed(1)}%`}</Descriptions.Item>
    </Descriptions>
    <Table size="small" pagination={false} rowKey="name" style={{ marginTop: 10 }} dataSource={Object.entries(result.components).map(([name, score]) => ({ name, score, weight: result.component_weights?.[name] }))} columns={[
      { title: "边际维度", dataIndex: "name" },
      { title: "权重", dataIndex: "weight", render: (value: number | undefined) => value == null ? "—" : `${(value * 100).toFixed(0)}%` },
      { title: "试算分", dataIndex: "score", render: (value: number) => value.toFixed(1) },
    ]} />
    <Text type="secondary" style={{ display: "block", marginTop: 8, fontSize: 12 }}>这是条件化的边际试算：结果只对输入的权重、共同样本和公开因子暴露成立。</Text>
  </Card>;
}
