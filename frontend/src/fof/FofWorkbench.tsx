/** FOF workspace: compact product desk | primary Agent conversation. */

import React, { createContext, lazy, Suspense, useCallback, useContext, useEffect, useRef, useState } from "react";
import { Button, Card, Drawer, Layout, Space, Spin, Tag, Tabs, Typography, message } from "antd";
import ReactECharts from "echarts-for-react";
import type { CalibrationQueueItem } from "./ProductLibraryPanel";
import AgentConversation from "./AgentConversation";
import { analyzeNav, getLatestCtaProductScoreItem, getProductPeers, kbGetNavSeries, kbGetProduct, type CtaProductScoreItem, type DataFrequency, type KbProduct } from "../api";
import { strategyLabel } from "./productDisplay";
import { productWorkflowStatus } from "./productWorkflow";
import type { CropRegion } from "./MultiProductReviewModal";

const { Content } = Layout;
const { Text } = Typography;

const CtaRankingPanel = lazy(() => import("./CtaRankingPanel"));
const CtaScorePanel = lazy(() => import("./CtaScorePanel"));
const CtaAttributionPanel = lazy(() => import("./CtaAttributionPanel"));
const PortfolioRiskMapPanel = lazy(() => import("./PortfolioRiskMapPanel"));
type WorkspacePanel = "agent" | "attribution" | "ranking" | "portfolio" | "scores" | "archive";

function initialWorkspacePanel(): WorkspacePanel {
  const value = new URLSearchParams(window.location.search).get("panel");
  return value === "attribution" || value === "ranking" || value === "portfolio" || value === "scores" ? value : "agent";
}

function initialResearchProductIds(): string[] {
  const value = new URLSearchParams(window.location.search).get("products");
  return value ? [...new Set(value.split(",").filter(Boolean))] : [];
}

function DeferredPanel({ children }: { children: React.ReactNode }): React.JSX.Element {
  return <Suspense fallback={<Card size="small"><Spin tip="正在加载分析模块" /></Card>}>{children}</Suspense>;
}

/** Workbench-level product actions shared by the library lane.
 *
 * Previously these four callbacks were drilled from FofWorkbench through
 * ProductLibraryPanel down to ProductLibraryItem (onFocusProduct and
 * onOpenResearch were pure passthroughs at the panel level).  The context
 * lets both the panel and the item cards consume them directly.
 */
export interface WorkbenchActions {
  onFocusProduct: (product: KbProduct) => void;
  onOpenResearch: (product: KbProduct) => void;
  onOpenSelection: (productIds: string[], panel: "agent" | "ranking" | "portfolio" | "scores") => void;
  onCalibrateProduct: (product: KbProduct, sourceFileId?: string, sourceRegion?: CropRegion, sourceFragmentId?: string) => void;
  onStartCalibrationQueue: (items: CalibrationQueueItem[]) => void;
}

export const WorkbenchActionsContext = createContext<WorkbenchActions | null>(null);

export function useWorkbenchActions(): WorkbenchActions {
  const actions = useContext(WorkbenchActionsContext);
  if (!actions) throw new Error("useWorkbenchActions 必须在产品工作区操作上下文中使用");
  return actions;
}

interface SnapshotMetrics {
  closeDate: string | null;
  totalReturn: number | null;
  maxDrawdown: number | null;
}

function ProductSnapshotCard({ product, compact = false }: { product: KbProduct; compact?: boolean }): React.JSX.Element {
  const [metrics, setMetrics] = useState<SnapshotMetrics | null>(null);
  const [curve, setCurve] = useState<Array<{ date: string; nav: number }>>([]);

  useEffect(() => {
    let active = true;
    kbGetNavSeries(product.id)
      .then(async (nav) => {
        if (!active) return;
        const ordered = [...nav].sort((a, b) => a.observation_date.localeCompare(b.observation_date));
        // Use the same reviewed unit-NAV series as chat recommendation and
        // peer comparison. Accumulated NAV can follow a different convention.
        const values = ordered
          .filter((item) => item.review_status === "reviewed")
          .map((item) => ({ observation_date: item.observation_date, net_asset_value: item.nav }))
          .filter((item) => Number.isFinite(item.net_asset_value) && item.net_asset_value > 0);
        setCurve(values.map((item) => ({ date: item.observation_date, nav: item.net_asset_value })));
        const frequency: DataFrequency = product.nav_frequency === "daily" || product.nav_frequency === "monthly"
          ? product.nav_frequency
          : "weekly";
        const analysis = values.length >= 2 ? await analyzeNav(values, frequency, 0.015) : null;
        if (!active) return;
        setMetrics({
          closeDate: product.nav_end ?? ordered.at(-1)?.observation_date ?? null,
          totalReturn: analysis?.metrics.cumulative_return ?? null,
          maxDrawdown: analysis?.metrics.maximum_drawdown ?? null,
        });
      })
      .catch(() => { if (active) setMetrics({ closeDate: null, totalReturn: null, maxDrawdown: null }); });
    return () => { active = false; };
  }, [product.id]);

  const formatPercent = (value: number | null) => value == null ? "暂无数据" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
  return <div style={{ width: "100%", padding: compact ? "0 0 8px" : 12, border: compact ? "none" : "1px solid var(--serif-border)", borderRadius: 8, background: "var(--serif-card)" }}>
    <div style={{ display: "flex", justifyContent: "space-between", gap: 6, marginBottom: 8 }}>
      <Text strong ellipsis style={{ maxWidth: 145 }}>{product.standard_name}</Text>
      <Tag color="blue" style={{ margin: 0, fontSize: 10 }}>当前产品</Tag>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 12px", fontSize: 11 }}>
      <Info label="公司" value={product.manager_name} />
      <Info label="策略" value={strategyLabel(product.strategy)} />
      <Info label="发行日期" value={product.inception_date} />
      <Info label="截止日期" value={metrics?.closeDate ?? "加载中"} />
      <Info label="区间收益" value={formatPercent(metrics?.totalReturn ?? null)} />
      <Info label="最大回撤" value={formatPercent(metrics?.maxDrawdown ?? null)} />
    </div>
    {curve.length >= 2 && <ReactECharts
      option={{
        grid: { left: 42, right: 12, top: 18, bottom: 30 },
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: curve.map((item) => item.date), axisLabel: { hideOverlap: true } },
        yAxis: { type: "value", scale: true },
        series: [{ type: "line", data: curve.map((item) => item.nav), showSymbol: false, smooth: true, lineStyle: { color: "#1677ff" }, areaStyle: { color: "rgba(22,119,255,.10)" } }],
      }}
      style={{ height: 190, marginTop: 12 }}
    />}
  </div>;
}

function ProductPeerCard({ product }: { product: KbProduct }): React.JSX.Element {
  const [peers, setPeers] = useState<Awaited<ReturnType<typeof getProductPeers>> | null>(null);
  useEffect(() => { void getProductPeers(product.id).then(setPeers).catch(() => setPeers({ similar: [], diversifiers: [] })); }, [product.id]);
  const rows = (title: string, items: NonNullable<typeof peers>["similar"]) => <div style={{ marginTop: 8 }}><Text strong style={{ fontSize: 12 }}>{title}</Text>{items.length ? items.map((item) => <div key={item.product_id} style={{ fontSize: 12, marginTop: 3 }}>{item.name} · 相关性 {item.correlation.toFixed(2)} · 共同期数 {item.overlap}</div>) : <Text type="secondary" style={{ display: "block", fontSize: 12 }}>暂无足够共同历史的同类产品</Text>}</div>;
  return <Card size="small" title="同类与分散化" style={{ marginTop: 10 }}>{peers ? <>{rows("最相似产品", peers.similar)}{rows("低相关替代品", peers.diversifiers)}</> : <Spin size="small" />}</Card>;
}

function ProductSummaryBar({ product, selected, onToggleSelect, onOpenDetails, onCalibrateProduct, onOpenResearch }: {
  product: KbProduct;
  selected: boolean;
  onToggleSelect: () => void;
  onOpenDetails: () => void;
  onCalibrateProduct: () => void;
  onOpenResearch: () => void;
}): React.JSX.Element {
  const workflow = productWorkflowStatus(product);
  const [score, setScore] = useState<{ asOfDate: string; item: CtaProductScoreItem } | null>(null);
  useEffect(() => {
    let active = true;
    void getLatestCtaProductScoreItem(product.id)
      .then((result) => { if (active) setScore({ asOfDate: result.as_of_date, item: result.item }); })
      .catch(() => { if (active) setScore(null); });
    return () => { active = false; };
  }, [product.id]);
  const reviewPercent = product.nav_count > 0 ? Math.round(product.reviewed_nav_count / product.nav_count * 100) : 0;
  const sourceConfidence = product.nav_confidence == null ? "未评估" : `${Math.round(product.nav_confidence * 100)}%`;
  return <Card size="small" style={{ marginBottom: 10 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
      <div>
        <Space size={8} wrap><Text strong style={{ fontSize: 16 }}>{product.standard_name}</Text><Tag color={workflow.color}>{workflow.label}</Tag></Space>
        <Text type="secondary" style={{ display: "block", marginTop: 4, fontSize: 12 }}>{product.manager_name || "管理人未披露"} · 净值 {product.nav_count} 条 · 人工复核 {reviewPercent}% · {product.nav_frequency === "weekly" ? "周频" : product.nav_frequency === "monthly" ? "月频" : "日频"} · 截止 {product.nav_end || "未披露"}</Text>
        <Text type="secondary" style={{ display: "block", marginTop: 2, fontSize: 12 }}>资料识别 {sourceConfidence}（仅影响资料解析可信度，不覆盖人工复核结果）</Text>
        {score && <Space wrap size={6} style={{ marginTop: 6 }}>
          <Tag color="blue">质量分 {score.item.summary.quality_score?.toFixed(1) ?? "—"}</Tag>
          <Tag>置信分 {score.item.summary.confidence_score.toFixed(1)}</Tag>
          <Tag>R² {score.item.summary.headline.r_squared?.toFixed(2) ?? "—"}</Tag>
          <Tag>样本外 R² {score.item.summary.headline.oos_r_squared?.toFixed(2) ?? "—"}</Tag>
          <Text type="secondary" style={{ fontSize: 11 }}>评分截至 {score.asOfDate}</Text>
        </Space>}
      </div>
      <Space wrap>
        <Button size="small" onClick={onToggleSelect}>{selected ? "已加入对话" : "加入对话"}</Button>
        <Button size="small" onClick={onCalibrateProduct}>复核净值</Button>
        <Button size="small" type="primary" onClick={onOpenResearch}>查看研究</Button>
        <Button size="small" onClick={onOpenDetails}>详情</Button>
      </Space>
    </div>
  </Card>;
}

function Info({ label, value }: { label: string; value: string | null }): React.JSX.Element {
  return <div><div style={{ color: "var(--serif-muted-foreground)" }}>{label}</div><div style={{ marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{value || "未披露"}</div></div>;
}

export default function FofWorkbench({ onCalibrateProduct, onStartCalibrationQueue, compareIds, compareNames, onCompareChange, onToggleCompare, onOpenCompare, onOpenProductLibrary }: {
  onCalibrateProduct: (product: KbProduct, sourceFileId?: string, sourceRegion?: CropRegion, sourceFragmentId?: string) => void;
  onStartCalibrationQueue: (items: CalibrationQueueItem[]) => void;
  compareIds: string[];
  compareNames: Record<string, string>;
  onCompareChange: (ids: string[], names: Record<string, string>) => void;
  onToggleCompare: (productId: string, name?: string) => void;
  onOpenCompare: () => void;
  onOpenProductLibrary: () => void;
}): React.JSX.Element {
  const [selectedProductIds, setSelectedProductIds] = useState<string[]>(initialResearchProductIds);
  const [focusedProduct, setFocusedProduct] = useState<KbProduct | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [activePanel, setActivePanel] = useState<WorkspacePanel>(initialWorkspacePanel);
  const restoredUrlState = useRef(false);
  useEffect(() => {
    if (!restoredUrlState.current) {
      restoredUrlState.current = true;
      setActivePanel(initialWorkspacePanel());
      setSelectedProductIds(initialResearchProductIds());
      return;
    }
    const url = new URL(window.location.href);
    if (activePanel === "agent") url.searchParams.delete("panel");
    else url.searchParams.set("panel", activePanel);
    if (selectedProductIds.length === 0) url.searchParams.delete("products");
    else url.searchParams.set("products", selectedProductIds.join(","));
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  }, [activePanel, selectedProductIds]);
  useEffect(() => {
    const restoreFromUrl = () => {
      setActivePanel(initialWorkspacePanel());
      setSelectedProductIds(initialResearchProductIds());
    };
    window.addEventListener("popstate", restoreFromUrl);
    return () => window.removeEventListener("popstate", restoreFromUrl);
  }, []);
  const handleToggleProduct = useCallback((productId: string) => {
    setSelectedProductIds((prev) => prev.includes(productId) ? prev.filter((id) => id !== productId) : [...prev, productId]);
  }, []);
  const handleFocusProduct = useCallback((product: KbProduct) => {
    setFocusedProduct(product);
    setSelectedProductIds((prev) => prev.includes(product.id) ? prev : [...prev, product.id]);
  }, []);
  const handleOpenResearch = useCallback((product: KbProduct) => {
    handleFocusProduct(product);
    setActivePanel("attribution");
  }, [handleFocusProduct]);
  useEffect(() => {
    const openHandler = (event: Event) => {
      const product = (event as CustomEvent<KbProduct>).detail;
      if (product?.id) handleOpenResearch(product);
    };
    const focusHandler = (event: Event) => {
      const product = (event as CustomEvent<KbProduct>).detail;
      if (product?.id) handleFocusProduct(product);
    };
    const toggleHandler = (event: Event) => {
      const detail = (event as CustomEvent<{ productId: string; selected: boolean }>).detail;
      if (!detail?.productId) return;
      setSelectedProductIds((current) => detail.selected
        ? (current.includes(detail.productId) ? current : [...current, detail.productId])
        : current.filter((id) => id !== detail.productId));
    };
    const selectionHandler = (event: Event) => {
      const detail = (event as CustomEvent<{ productIds: string[]; panel: "agent" | "ranking" | "portfolio" | "scores" }>).detail;
      if (!detail?.productIds?.length) return;
      setSelectedProductIds(detail.productIds);
      setActivePanel(detail.panel);
    };
    window.addEventListener("kb:open-product", openHandler);
    window.addEventListener("kb:focus-product", focusHandler);
    window.addEventListener("kb:toggle-product", toggleHandler);
    window.addEventListener("kb:open-selection", selectionHandler);
    return () => {
      window.removeEventListener("kb:open-product", openHandler);
      window.removeEventListener("kb:focus-product", focusHandler);
      window.removeEventListener("kb:toggle-product", toggleHandler);
      window.removeEventListener("kb:open-selection", selectionHandler);
    };
  }, [handleFocusProduct, handleOpenResearch]);
  const handleDropProduct = useCallback((productId: string) => {
    setSelectedProductIds((prev) => prev.includes(productId) ? prev : [...prev, productId]);
  }, []);

  const handleRemoveProduct = useCallback((productId: string) => {
    setSelectedProductIds((prev) => prev.filter((id) => id !== productId));
  }, []);
  const handleReviewProduct = useCallback((productId: string) => {
    void kbGetProduct(productId)
      .then((product) => onCalibrateProduct(product))
      .catch((error) => message.error(error instanceof Error ? error.message : "打开产品复核失败"));
  }, [onCalibrateProduct]);
  return <Layout style={{ width: "100%", minWidth: 0, height: "calc(100vh - 64px)", minHeight: 580, background: "transparent", overflow: "hidden" }}>
    <div style={{ padding: "12px 20px 0", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
      <div>
        <Text strong>FOF 工作台</Text>
      </div>
    </div>
    <Layout style={{ width: "100%", minWidth: 0, flex: 1, minHeight: 0, background: "transparent", padding: "14px 20px 20px" }}>
      <Content style={{ width: "100%", minWidth: 0, display: "flex", minHeight: 650, flexDirection: "column", gap: 10 }}>
        {focusedProduct && <ProductSummaryBar product={focusedProduct} selected={selectedProductIds.includes(focusedProduct.id)} onToggleSelect={() => handleToggleProduct(focusedProduct.id)} onOpenDetails={() => setDetailsOpen(true)} onCalibrateProduct={() => onCalibrateProduct(focusedProduct)} onOpenResearch={() => handleOpenResearch(focusedProduct)} />}
        {selectedProductIds.length > 0 && <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, padding: "7px 12px", border: "1px solid var(--serif-border)", borderRadius: 8, background: "var(--serif-card)", flexWrap: "wrap" }}>
          <div>
            <Text strong>研究对象：{selectedProductIds.length} 个产品</Text>
            <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>用于单品研究、复核与对比；筛选、排名和 FOF 配置会检索产品库中全部已复核产品。</Text>
          </div>
          <Space size={6}>
            {activePanel !== "agent" && <Button size="small" onClick={() => setActivePanel("agent")}>查看 Agent 对话</Button>}
            <Button size="small" type="link" onClick={() => setSelectedProductIds([])}>清空研究对象</Button>
          </Space>
        </div>}
        <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
          <Tabs activeKey={activePanel} onChange={(key) => setActivePanel(key as typeof activePanel)} items={[
            { key: "agent", label: "Agent 对话", children: <AgentConversation selectedProductIds={selectedProductIds} onAddProduct={handleDropProduct} onRemoveProduct={handleRemoveProduct} onReviewProduct={handleReviewProduct} onOpenProductLibrary={onOpenProductLibrary} onCompareProducts={(productIds, productNames) => { onCompareChange(productIds, productNames); onOpenCompare(); }} onMaterialAnalyzed={() => undefined} /> },
            { key: "attribution", label: "归因与风险模型", children: <DeferredPanel><Card size="small"><CtaAttributionPanel product={focusedProduct ?? undefined} /></Card></DeferredPanel> },
            { key: "portfolio", label: "组合风险地图", children: <DeferredPanel><Card size="small"><PortfolioRiskMapPanel selectedProductIds={selectedProductIds} /></Card></DeferredPanel> },
            { key: "ranking", label: "周度排名", children: <DeferredPanel><CtaRankingPanel selectedProductIds={selectedProductIds} /></DeferredPanel> },
            { key: "scores", label: "产品评分与画像", children: <DeferredPanel><CtaScorePanel selectedProductIds={selectedProductIds} compareIds={compareIds} onCompareChange={onCompareChange} onOpenCompare={onOpenCompare} /></DeferredPanel> },
          ]} />
        </div>
        {compareIds.length > 0 && (
          <div style={{ flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "8px 12px", border: "1px solid var(--serif-border)", borderRadius: 8, background: "var(--serif-card)" }}>
            <Space size={8} wrap>
              <Text strong>对比</Text>
              {compareIds.map((id) => (
                <Tag key={id} closable onClose={(event) => { event.preventDefault(); onToggleCompare(id); }}>{compareNames[id] ?? "未命名产品"}</Tag>
              ))}
              {compareIds.length === 1 && <Text type="secondary">再勾选一只产品</Text>}
            </Space>
            <Button size="small" type="primary" disabled={compareIds.length < 2} onClick={onOpenCompare}>查看对比</Button>
          </div>
        )}
      </Content>
    </Layout>
    <Drawer title="产品详情" placement="right" width={380} open={detailsOpen} onClose={() => setDetailsOpen(false)}>
      {focusedProduct ? <>
        <Button size="small" block style={{ marginBottom: 10 }} onClick={() => onToggleCompare(focusedProduct.id, focusedProduct.standard_name)}>
          {compareIds.includes(focusedProduct.id) ? "移出对比" : "加入对比"}
        </Button>
        <ProductSnapshotCard product={focusedProduct} />
        <ProductPeerCard product={focusedProduct} />
      </> : <Text type="secondary">请先从产品列表选择产品。</Text>}
    </Drawer>
  </Layout>;
}
