/** FOF workspace: compact product desk | primary Agent conversation. */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Button, Card, Dropdown, Layout, Tag, Typography } from "antd";
import { AppstoreOutlined, CloseOutlined, DownOutlined, UpOutlined } from "@ant-design/icons";
import ProductLibraryPanel from "./ProductLibraryPanel";
import CtaRankingPanel from "./CtaRankingPanel";
import CtaAttributionPanel from "./CtaAttributionPanel";
import AgentConversation from "./AgentConversation";
import { analyzeNav, kbGetNavSeries, type DataFrequency, type KbProduct } from "../api";
import { strategyLabel } from "./productDisplay";

const { Content } = Layout;
const { Text } = Typography;

interface SnapshotMetrics {
  closeDate: string | null;
  totalReturn: number | null;
  maxDrawdown: number | null;
}

function ProductSnapshotCard({ product, selected, onClose }: { product: KbProduct; selected: boolean; onClose: () => void }): React.JSX.Element {
  const [metrics, setMetrics] = useState<SnapshotMetrics | null>(null);

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
  return <div
    draggable
    onDragStart={(event) => {
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("application/x-fof-product", JSON.stringify({ id: product.id, name: product.standard_name }));
    }}
    style={{ width: "100%", padding: 12, border: "1px solid var(--serif-border)", borderRadius: 8, background: "var(--serif-card)", cursor: "grab" }}
    title="拖拽到右侧 Agent 对话区即可发起产品分析"
  >
    <div style={{ display: "flex", justifyContent: "space-between", gap: 6, marginBottom: 8 }}>
      <Text strong ellipsis style={{ maxWidth: 145 }}>{product.standard_name}</Text>
      <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
        <Tag color={selected ? "gold" : "default"} style={{ margin: 0, fontSize: 10 }}>{selected ? "已选" : "查看"}</Tag>
        <Button type="text" size="small" icon={<CloseOutlined />} aria-label={`关闭 ${product.standard_name} 信息卡`} onClick={(event) => { event.stopPropagation(); onClose(); }} />
      </div>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 12px", fontSize: 11 }}>
      <Info label="公司" value={product.manager_name} />
      <Info label="策略" value={strategyLabel(product.strategy)} />
      <Info label="发行日期" value={product.inception_date} />
      <Info label="截止日期" value={metrics?.closeDate ?? "加载中"} />
      <Info label="区间收益" value={formatPercent(metrics?.totalReturn ?? null)} />
      <Info label="最大回撤" value={formatPercent(metrics?.maxDrawdown ?? null)} />
    </div>
    <div style={{ marginTop: 10, color: "var(--serif-accent)", fontSize: 11 }}>拖入右侧以对话分析</div>
  </div>;
}

function Info({ label, value }: { label: string; value: string | null }): React.JSX.Element {
  return <div><div style={{ color: "var(--serif-muted-foreground)" }}>{label}</div><div style={{ marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{value || "未披露"}</div></div>;
}

export default function FofWorkbench({ onCalibrateProduct }: { onCalibrateProduct: (product: KbProduct, sourceFileId?: string) => void }): React.JSX.Element {
  const [selectedProductIds, setSelectedProductIds] = useState<string[]>([]);
  const [pinnedProducts, setPinnedProducts] = useState<KbProduct[]>([]);
  const [infoCollapsed, setInfoCollapsed] = useState(false);
  const [productRefreshKey, setProductRefreshKey] = useState(0);
 const [libraryWidth, setLibraryWidth] = useState(320);
  const [rankingOpen, setRankingOpen] = useState(false);
  const [rightPanel, setRightPanel] = useState<"agent" | "attribution">("agent");
 const resizeCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => () => resizeCleanupRef.current?.(), []);

  const beginLibraryResize = useCallback((startX: number) => {
    resizeCleanupRef.current?.();
    const initialWidth = libraryWidth;
    const onMove = (event: PointerEvent) => {
      // Keep both work areas usable while allowing the analyst to prioritize
      // either the product list or the conversation.
      setLibraryWidth(Math.max(220, Math.min(540, initialWidth + event.clientX - startX)));
    };
    const onEnd = () => resizeCleanupRef.current?.();
    const cleanup = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
      resizeCleanupRef.current = null;
    };
    resizeCleanupRef.current = cleanup;
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onEnd, { once: true });
    window.addEventListener("pointercancel", onEnd, { once: true });
  }, [libraryWidth]);
  const handleToggleProduct = useCallback((productId: string) => {
    setSelectedProductIds((prev) => prev.includes(productId) ? prev.filter((id) => id !== productId) : [...prev, productId]);
  }, []);
  const handleFocusProduct = useCallback((product: KbProduct) => {
    setPinnedProducts((prev) => prev.some((item) => item.id === product.id) ? prev : [...prev, product].slice(-4));
    setInfoCollapsed(false);
  }, []);
  const handleDropProduct = useCallback((productId: string) => {
    setSelectedProductIds((prev) => prev.includes(productId) ? prev : [...prev, productId]);
  }, []);

  const handleCloseProductInfo = useCallback((productId: string) => {
    setPinnedProducts((prev) => prev.filter((product) => product.id !== productId));
  }, []);

  return <Layout style={{ width: "100%", minWidth: 0, height: "calc(100vh - 64px)", minHeight: 580, background: "transparent", overflow: "hidden" }}>
    <div style={{ padding: "12px 20px 0", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
      <div>
        <Text strong>FOF 工作台</Text>
        <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>上传资料，处理待办，然后直接和 Agent 对话</Text>
      </div>
      <Dropdown trigger={["click"]} menu={{ items: [
        { key: "agent", label: "Agent 对话", onClick: () => setRightPanel("agent") },
        { key: "attribution", label: "动态归因", onClick: () => setRightPanel("attribution") },
        { key: "ranking", label: rankingOpen ? "收起周度排名" : "查看周度排名", onClick: () => setRankingOpen((value) => !value) },
      ] }}>
        <Button size="small" icon={<AppstoreOutlined />}>分析工具</Button>
      </Dropdown>
    </div>
    {rankingOpen && <div style={{ padding: "12px 20px 0" }}><CtaRankingPanel selectedProductIds={selectedProductIds} /></div>}
    <Layout style={{ width: "100%", minWidth: 0, flex: 1, minHeight: 0, background: "transparent", padding: "14px 20px 20px", gap: 14, flexDirection: "row", alignItems: "stretch" }}>
      <aside style={{ flex: `0 0 ${libraryWidth}px`, minWidth: 220, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, minHeight: 0, padding: 12, border: "1px solid var(--serif-border)", borderRadius: 10, background: "var(--serif-card)", display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ flex: 1, minHeight: 0 }}>
          <ProductLibraryPanel refreshKey={productRefreshKey} selectedIds={selectedProductIds} onToggleSelect={handleToggleProduct} onFocusProduct={handleFocusProduct} onCalibrateProduct={onCalibrateProduct} />
          </div>
        </div>
      </aside>
      <div
        role="separator"
        aria-label="调整产品库与 Agent 对话宽度"
        aria-orientation="vertical"
        tabIndex={0}
        onPointerDown={(event) => {
          event.preventDefault();
          beginLibraryResize(event.clientX);
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            setLibraryWidth((width) => Math.max(220, width - 24));
          }
          if (event.key === "ArrowRight") {
            event.preventDefault();
            setLibraryWidth((width) => Math.min(540, width + 24));
          }
        }}
        title="拖动调整产品库与 Agent 对话宽度（方向键也可调整）"
        style={{
          flex: "0 0 12px",
          margin: "0 -6px",
          cursor: "col-resize",
          touchAction: "none",
          position: "relative",
          zIndex: 2,
          outlineOffset: -2,
          background: "linear-gradient(90deg, transparent 4px, var(--serif-border) 5px, var(--serif-border) 7px, transparent 8px)",
        }}
      />
      <Content style={{ flex: "1 1 0", width: "auto", minWidth: 0, display: "flex", minHeight: 650, flexDirection: "column", gap: 10 }}>
        {rightPanel === "attribution" && <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>动态归因只读取已确认、已审核净值</Text>
          <Button type="link" size="small" onClick={() => setRightPanel("agent")}>返回对话</Button>
        </div>}
        <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
          {rightPanel === "agent"
            ? <AgentConversation selectedProductIds={selectedProductIds} onDropProduct={handleDropProduct} onMaterialAnalyzed={() => setProductRefreshKey((key) => key + 1)} />
            : <Card size="small" title="CTA 动态归因 · Phase A"><CtaAttributionPanel product={pinnedProducts.at(-1)} /></Card>}
        </div>
      </Content>
      {pinnedProducts.length > 0 && (
        <aside style={{ flex: "0 0 288px", minWidth: 0, display: "flex", flexDirection: "column" }}>
          <Card
            size="small"
            style={{ height: "100%", overflow: "hidden" }}
            title="产品信息"
            extra={<Button size="small" type="text" icon={infoCollapsed ? <DownOutlined /> : <UpOutlined />} aria-label={infoCollapsed ? "展开产品信息" : "收起产品信息"} onClick={() => setInfoCollapsed((v) => !v)} />}
            styles={{ body: infoCollapsed ? { display: "none" } : { height: "calc(100% - 40px)", padding: 10, overflowY: "auto" } }}
          >
            <Text type="secondary" style={{ display: "block", fontSize: 11, marginBottom: 10 }}>点击左侧产品可固定在此处；可直接拖入对话。</Text>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {pinnedProducts.map((product) => <ProductSnapshotCard key={product.id} product={product} selected={selectedProductIds.includes(product.id)} onClose={() => handleCloseProductInfo(product.id)} />)}
            </div>
          </Card>
        </aside>
      )}
    </Layout>
  </Layout>;
}
