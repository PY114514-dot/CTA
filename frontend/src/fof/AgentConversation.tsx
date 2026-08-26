/** Center panel: Agent conversation with real backend, tool trace and citations. */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { Button, Card, Collapse, Empty, Input, Popover, Space, Tag, Tooltip, Typography, message } from "antd";
import {
  AppstoreOutlined,
  CheckCircleFilled,
  ClockCircleOutlined,
  DeleteOutlined,
  FileTextOutlined,
  HistoryOutlined,
  ProfileOutlined,
  PlusOutlined,
  RobotOutlined,
  SendOutlined,
  ToolOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { kbChat, kbGetProduct, kbSaveAllocationDraft, kbSubmitAllocationDraft, kbUploadFile, type ChatAllocationDraft, type ChatCitation, type ChatMessage, type ChatToolCall, type KbProductDetail } from "../api";

const { Text, Paragraph } = Typography;
const { TextArea } = Input;

interface DisplayMessage {
  role: "user" | "agent";
  content: string;
  timestamp: string;
  citations?: ChatCitation[];
  toolCalls?: ChatToolCall[];
  productsReferenced?: Array<{ id: string; name: string; category: string }>;
  dataContext?: ChatMessage["data_context"];
  methodProvenance?: ChatMessage["method_provenance"];
  allocationDraft?: ChatAllocationDraft | null;
}

interface ConversationSession {
  id: string;
  title: string;
  productIds: string[];
  messages: DisplayMessage[];
  updatedAt: number;
}

type AgentTaskType = "allocation" | "screening" | "research";

interface AgentRunProgress {
  startedAt: number;
  taskType: AgentTaskType;
}

const SESSION_STORAGE_KEY = "fof-agent-conversations-v1";

function newSession(productIds: string[] = []): ConversationSession {
  return { id: `chat-${crypto.randomUUID()}`, title: "新研究", productIds, messages: [], updatedAt: Date.now() };
}

function loadSessions(): ConversationSession[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) ?? "[]") as ConversationSession[];
    return parsed.length ? parsed : [newSession()];
  } catch {
    return [newSession()];
  }
}

function inferTaskType(query: string): AgentTaskType {
  if (/(配置|组合|fof|权重|回撤|波动|夏普)/i.test(query)) return "allocation";
  if (/(筛选|排名|候选|挑选)/.test(query)) return "screening";
  return "research";
}

interface Props {
  selectedProductIds: string[];
  onAddProduct: (productId: string) => void;
  onRemoveProduct: (productId: string) => void;
  onReviewProduct: (productId: string) => void;
  onOpenProductLibrary: () => void;
  onCompareProducts: (productIds: string[], productNames: Record<string, string>) => void;
  onMaterialAnalyzed?: () => void;
}

export default function AgentConversation({ selectedProductIds, onAddProduct, onRemoveProduct, onReviewProduct, onOpenProductLibrary, onCompareProducts, onMaterialAnalyzed }: Props): React.JSX.Element {
  const [sessions, setSessions] = useState<ConversationSession[]>(loadSessions);
  // The active ID must come from the same initial list. Calling loadSessions()
  // twice creates two different fallback sessions, so messages would be written
  // to an ID that is not present in the visible history.
  const [activeSessionId, setActiveSessionId] = useState(() => sessions[0]!.id);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [runProgress, setRunProgress] = useState<AgentRunProgress | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [contextProducts, setContextProducts] = useState<KbProductDetail[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const materialInputRef = useRef<HTMLInputElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  // deleteSession guarantees sessions is never empty, so [0] is always defined.
  const activeSession = useMemo(() => sessions.find((item) => item.id === activeSessionId) ?? sessions[0]!, [activeSessionId, sessions]);
  const messages = activeSession?.messages ?? [];

  useEffect(() => {
    if (!sessions.some((session) => session.id === activeSessionId)) {
      setActiveSessionId(sessions[0]!.id);
    }
  }, [activeSessionId, sessions]);

  useEffect(() => {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    let active = true;
    void Promise.all(selectedProductIds.map((productId) => kbGetProduct(productId).catch(() => null))).then((products) => {
      if (active) setContextProducts(products.filter((product): product is KbProductDetail => product != null));
    });
    return () => { active = false; };
  }, [selectedProductIds]);

  const updateSession = useCallback((sessionId: string, update: (session: ConversationSession) => ConversationSession) => {
    setSessions((previous) => previous.map((session) => session.id === sessionId ? update(session) : session));
  }, []);

  const updateActiveSession = useCallback((update: (session: ConversationSession) => ConversationSession) => {
    updateSession(activeSession.id, update);
  }, [activeSession.id, updateSession]);

  const appendMessages = useCallback((items: DisplayMessage[]) => {
    updateActiveSession((session) => ({ ...session, messages: [...session.messages, ...items], updatedAt: Date.now() }));
  }, [updateActiveSession]);

  const createSession = useCallback(() => {
    const created = newSession(selectedProductIds);
    setSessions((previous) => [created, ...previous]);
    setActiveSessionId(created.id);
    setInput("");
  }, [selectedProductIds]);

  const deleteSession = useCallback((sessionId: string) => {
    setSessions((previous) => {
      const remaining = previous.filter((session) => session.id !== sessionId);
      const next = remaining.length ? remaining : [newSession(selectedProductIds)];
      if (sessionId === activeSessionId) setActiveSessionId(next[0]!.id);
      return next;
    });
  }, [activeSessionId, selectedProductIds]);

  const historyList = <div style={{ width: 250, maxHeight: 360, overflow: "auto" }}>
    <Text type="secondary" style={{ fontSize: 11 }}>研究历史</Text>
    <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 7 }}>
      {[...sessions].sort((left, right) => right.updatedAt - left.updatedAt).map((session) => {
        const active = session.id === activeSession.id;
        return <div key={session.id} style={{ display: "flex", gap: 2, alignItems: "center", padding: "6px 3px 6px 8px", borderRadius: 6, background: active ? "var(--serif-muted)" : "transparent" }}>
          <button type="button" onClick={() => { setActiveSessionId(session.id); setHistoryOpen(false); }} title={session.title} style={{ border: 0, background: "transparent", cursor: "pointer", padding: 0, flex: 1, minWidth: 0, textAlign: "left", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--serif-foreground)", fontSize: 12 }}>{session.title}</button>
          <Tooltip title="删除该对话"><Button type="text" size="small" icon={<DeleteOutlined />} aria-label={`删除 ${session.title}`} onClick={() => deleteSession(session.id)} /></Tooltip>
        </div>;
      })}
    </div>
  </div>;

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });
  }, []);

  useEffect(() => {
    if (!runProgress) return undefined;
    const intervalId = window.setInterval(scrollToBottom, 1000);
    return () => window.clearInterval(intervalId);
  }, [runProgress, scrollToBottom]);

  const handleSend = useCallback(async (presetQuery?: string) => {
    const query = presetQuery ?? input.trim();
    if (!query || loading) return;
    const sessionId = activeSession.id;
    const appendToSession = (items: DisplayMessage[]) => {
      updateSession(sessionId, (session) => ({ ...session, messages: [...session.messages, ...items], updatedAt: Date.now() }));
    };
    const now = new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    const productIds = selectedProductIds.length ? selectedProductIds : activeSession.productIds;
    updateSession(sessionId, (session) => ({ ...session, productIds, title: session.messages.length === 0 ? query.slice(0, 22) : session.title, messages: [...session.messages, { role: "user", content: query, timestamp: now }], updatedAt: Date.now() }));
    setInput("");
    setLoading(true);
    setRunProgress({ startedAt: Date.now(), taskType: inferTaskType(query) });
    scrollToBottom();
    const requestController = new AbortController();
    activeRequestRef.current = requestController;

    try {
      const response: ChatMessage = await kbChat({
        query,
        product_ids: productIds,
        session_id: activeSession.id,
      }, requestController.signal);
      appendToSession([
        {
          role: "agent",
          content: response.content,
          timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
          citations: response.citations,
          toolCalls: response.tool_calls,
          productsReferenced: response.products_referenced,
          dataContext: response.data_context,
          methodProvenance: response.method_provenance,
          allocationDraft: response.allocation_draft,
        },
      ]);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        message.info("已停止等待本次回复，您可以继续发送新问题。");
        return;
      }
      message.error(err instanceof Error ? err.message : "Agent 对话失败");
      appendToSession([
        {
          role: "agent",
          content: "抱歉，处理您的问题时出现错误。请检查后端服务是否正常运行。",
          timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      if (activeRequestRef.current === requestController) activeRequestRef.current = null;
      setLoading(false);
      setRunProgress(null);
      scrollToBottom();
    }
  }, [activeSession, input, loading, selectedProductIds, scrollToBottom, updateSession]);

  const stopWaiting = useCallback(() => activeRequestRef.current?.abort(), []);

  const handleMaterialUpload = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      message.warning("单图研究目前支持 PNG、JPG 或 WEBP 图片。");
      return;
    }
    if (loading) return;
    const now = new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    appendMessages([{ role: "user", content: `请研究我上传的素材《${file.name}》`, timestamp: now }]);
    setLoading(true);
    setRunProgress({ startedAt: Date.now(), taskType: "research" });
    scrollToBottom();
    try {
      const result = await kbUploadFile(file);
      appendMessages([{
        role: "agent",
        content: `已将素材《${file.name}》送入产品库解析。解析完成后，它会直接出现在左侧「复核队列」中；如识别出净值曲线，请点击产品卡的「复核或校准净值」完成确认。`,
        timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
        toolCalls: [{ name: "register_product_material", status: "ok", duration_ms: null, summary: `资料已登记，当前状态：${result.parsing_status}` }],
      }]);
      onMaterialAnalyzed?.();
    } catch (err) {
      const detail = err instanceof Error ? err.message : "单图研究失败";
      message.error(detail);
      appendMessages([{ role: "agent", content: `未能完成该图片研究：${detail}`, timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) }]);
    } finally {
      setLoading(false);
      setRunProgress(null);
      scrollToBottom();
    }
  }, [appendMessages, loading, onMaterialAnalyzed, scrollToBottom]);

  const startProductTask = (query: string): void => {
    if (selectedProductIds.length === 0 && activeSession.productIds.length === 0) {
      onOpenProductLibrary();
      return;
    }
    void handleSend(query);
  };

  const openDeterministicCompare = (): void => {
    const productIds = selectedProductIds.length ? selectedProductIds : activeSession.productIds;
    if (productIds.length !== 2) {
      message.info("请先选择两只产品，再进行对比。");
      return;
    }
    onCompareProducts(productIds, Object.fromEntries(contextProducts.map((product) => [product.id, product.standard_name])));
  };

  return (
    <Card
      size="small"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        const raw = event.dataTransfer.getData("application/x-fof-product");
        try {
          const product = JSON.parse(raw) as { id: string; name: string };
          if (product.id && product.name) {
            onAddProduct(product.id);
            void handleSend(
              `请基于产品「${product.name}」的已导入资料，概述策略、收益风险特征，并提示需要补充的数据。`,
            );
          }
        } catch {
          // Ignore drops that do not originate from a product information card.
        }
      }}
      style={{ width: "100%", height: "calc(100vh - 220px)", minHeight: 560, display: "flex", flexDirection: "column", overflow: "hidden" }}
      styles={{ body: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", padding: "12px 16px" } }}
      title={
        <Space size={8}>
          <RobotOutlined style={{ color: "var(--serif-accent, #1677ff)" }} />
          <span>Agent 对话</span>
          {selectedProductIds.length > 0 && (
            <Tag style={{ fontSize: 10 }}>{selectedProductIds.length} 个产品已选</Tag>
          )}
        </Space>
      }
      extra={<Space size={6}>
        {selectedProductIds.length === 0 && <Button size="small" type="primary" icon={<AppstoreOutlined />} onClick={onOpenProductLibrary}>选择产品</Button>}
        <Popover content={historyList} trigger="click" placement="bottomRight" open={historyOpen} onOpenChange={setHistoryOpen}><Button size="small" icon={<HistoryOutlined />}>历史 {sessions.length}</Button></Popover>
        <Button size="small" icon={<PlusOutlined />} onClick={createSession}>新建对话</Button>
      </Space>}
    >
      <input
        ref={materialInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) void handleMaterialUpload(file);
        }}
      />
      {/* Messages area */}
      <div ref={scrollRef} style={{ flex: 1, overflow: "auto", minWidth: 0, marginBottom: 12 }}>
        {messages.length === 0 ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<Space direction="vertical" size={8}>
            <Text strong>先选择产品，再开始研究</Text>
            <Text type="secondary" style={{ fontSize: 11 }}>从产品列表勾选产品后，可在这里进行复核、对比或 FOF 配置。</Text>
            <Button type="primary" icon={<AppstoreOutlined />} onClick={onOpenProductLibrary}>去产品列表选择产品</Button>
            <Text type="secondary" style={{ fontSize: 11 }}>也可以不选产品，直接描述筛选目标。</Text>
            <Space wrap>
              <Button size="small" onClick={() => startProductTask("请复核已选产品：先给出是否值得继续研究的结论，再列出收益风险特征、数据缺口与下一步。")}>产品复核</Button>
              <Button size="small" onClick={openDeterministicCompare}>产品对比</Button>
              <Button size="small" onClick={() => void handleSend("请按商品 CTA 的适配度筛选产品：先给出候选结论，再说明主要风险和数据缺口。")}>筛选与排名</Button>
              <Button size="small" onClick={() => startProductTask("我想控制回撤在 10% 内，请给出初始 FOF 配置、产品角色和待补充尽调项。")}>FOF 配置</Button>
            </Space>
          </Space>} style={{ marginTop: 42 }} />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {messages.map((msg, index) => (
              <MessageBubble
                key={index}
                msg={msg}
                onOpenProduct={(productId) => window.dispatchEvent(new CustomEvent("kb:open-selection", { detail: { productIds: [productId], panel: "scores" } }))}
                onReviewProduct={onReviewProduct}
              />
            ))}
            {loading && runProgress && <AgentRunProgressCard progress={runProgress} />}
          </div>
        )}
        {messages.length === 0 && loading && runProgress && <div style={{ display: "flex", justifyContent: "center", marginTop: 16 }}><AgentRunProgressCard progress={runProgress} /></div>}
      </div>

      {/* Input area */}
      <div style={{ flexShrink: 0 }}>
        {contextProducts.length > 0 && <Space wrap size={[4, 4]} style={{ marginBottom: 7 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>单品研究对象</Text>
          {contextProducts.map((product) => <Tag key={product.id} closable onClose={(event) => { event.preventDefault(); onRemoveProduct(product.id); }}>{product.standard_name}</Tag>)}
        </Space>}
        {contextProducts.length === 0 && <Space size={6} style={{ display: "flex", marginBottom: 7 }}><Text type="secondary" style={{ fontSize: 11 }}>未选择产品：可直接筛选或生成 FOF 配置，也可先选择产品做单品研究。</Text><Button type="link" size="small" style={{ padding: 0, height: "auto" }} onClick={onOpenProductLibrary}>打开产品列表</Button></Space>}
        <form onSubmit={(event) => { event.preventDefault(); void handleSend(); }} style={{ display: "flex", gap: 8 }}>
        <TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="输入研究问题或配置需求..."
          autoSize={{ minRows: 1, maxRows: 4 }}
          onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); void handleSend(); } }}
          style={{ flex: 1 }}
        />
        <Button htmlType="button" title="添加一张周报图片" icon={<UploadOutlined />} aria-label="添加周报图片" onClick={() => materialInputRef.current?.click()} disabled={loading} style={{ alignSelf: "flex-end" }} />
        {loading && <Button htmlType="button" onClick={stopWaiting} style={{ alignSelf: "flex-end" }}>停止等待</Button>}
        <Button
          htmlType="submit"
          type="primary"
          icon={<SendOutlined />}
          disabled={!input.trim() || loading}
          style={{ alignSelf: "flex-end" }}
        />
        </form>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function AgentRunProgressCard({ progress }: { progress: AgentRunProgress }): React.JSX.Element {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const updateElapsed = () => setElapsedSeconds(Math.max(0, Math.floor((Date.now() - progress.startedAt) / 1000)));
    updateElapsed();
    const intervalId = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(intervalId);
  }, [progress.startedAt]);

  const steps = progress.taskType === "allocation"
    ? [
      ["提交任务", "已收到配置需求"],
      ["检索候选产品", "读取产品库中已确认、净值已复核的产品"],
      ["规则筛选与配置优化", "按约束筛选并计算初始权重"],
      ["组合历史表现", "计算组合净值、收益、波动、夏普与回撤"],
      ["生成研究解读", "仅解释本地计算结果，不参与计算"],
    ]
    : progress.taskType === "screening"
      ? [
        ["提交任务", "已收到筛选需求"],
        ["检索候选产品", "读取产品库与已复核净值"],
        ["规则筛选", "按策略和风险条件过滤候选产品"],
        ["生成研究解读", "仅解释本地计算结果，不参与计算"],
      ]
      : [
        ["提交任务", "已收到研究问题"],
        ["检索资料", "读取已选产品、净值和相关证据"],
        ["计算指标", "执行净值、风险或归因的本地计算"],
        ["生成研究解读", "仅解释本地计算结果，不参与计算"],
      ];
  const activeStep = Math.min(steps.length - 1, Math.floor(elapsedSeconds / 2));

  return (
    <div style={{ alignSelf: "flex-start", width: "min(520px, 88%)", padding: "10px 12px", borderRadius: 8, background: "var(--serif-card-bg, #fff)", border: "1px solid color-mix(in srgb, var(--serif-accent, #2563EB) 24%, var(--serif-border, #e2e8f0))" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 7 }}>
        <Space size={6}>
          <ToolOutlined style={{ color: "var(--serif-accent, #2563EB)" }} />
          <Text strong style={{ fontSize: 12 }}>执行进度</Text>
        </Space>
        <Text type="secondary" style={{ fontSize: 11 }}>已用时 {elapsedSeconds} 秒</Text>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {steps.map(([title, detail], index) => {
          const completed = index < activeStep;
          const active = index === activeStep;
          return <div key={title} style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0, fontSize: 11, color: completed || active ? "var(--serif-text-secondary, #64748b)" : "#94a3b8" }}>
            {completed ? <CheckCircleFilled style={{ color: "#16a34a", fontSize: 12 }} /> : active ? <ClockCircleOutlined style={{ color: "var(--serif-accent, #2563EB)", fontSize: 12 }} /> : <span style={{ width: 12, height: 12, borderRadius: "50%", border: "1px solid currentColor", boxSizing: "border-box" }} />}
            <Tag color={active ? "blue" : undefined} style={{ margin: 0, fontSize: 10, lineHeight: "16px" }}>{title}</Tag>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{completed ? "完成" : active ? "进行中" : "等待执行"} · {detail}</span>
          </div>;
        })}
      </div>
      {elapsedSeconds >= 15 && <Text type="secondary" style={{ display: "block", marginTop: 7, fontSize: 11 }}>任务仍在运行。完成后会在下方列出实际使用的工具、耗时和结果。</Text>}
    </div>
  );
}

function MessageBubble({ msg, onOpenProduct, onReviewProduct }: { msg: DisplayMessage; onOpenProduct: (productId: string) => void; onReviewProduct: (productId: string) => void }): React.JSX.Element {
  const isUser = msg.role === "user";
  const [summary, ...detailParts] = msg.content.trim().split(/\n\s*\n/);
  const detail = detailParts.join("\n\n");
  return (
    <div
      style={{
        alignSelf: isUser ? "flex-end" : "flex-start",
        maxWidth: "88%",
        padding: "8px 12px",
        borderRadius: 8,
        background: isUser ? "color-mix(in srgb, var(--serif-accent-secondary, #d4a84b) 22%, white)" : "var(--serif-card-bg, #fafafa)",
        color: isUser ? "var(--serif-foreground, #1a1a1a)" : undefined,
        border: isUser ? "1px solid color-mix(in srgb, var(--serif-accent-secondary, #d4a84b) 38%, white)" : "1px solid var(--serif-border, #f0f0f0)",
      }}
    >
      <div style={{ fontSize: 13, whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
        {!isUser && <Text strong style={{ display: "block", marginBottom: 3 }}>研究结论</Text>}
        {summary}
      </div>
      {!isUser && detail && <Collapse size="small" ghost style={{ marginTop: 5 }} items={[{ key: "detail", label: <span style={{ fontSize: 11 }}>查看完整说明</span>, children: <div style={{ whiteSpace: "pre-wrap", fontSize: 13, lineHeight: 1.6 }}>{detail}</div> }]} />}

      {/* Tool calls */}
      {msg.allocationDraft && <AllocationDraftCard draft={msg.allocationDraft} />}

      {!isUser && <AgentActions msg={msg} onOpenProduct={onOpenProduct} onReviewProduct={onReviewProduct} />}

      {/* Tool calls */}
      {msg.toolCalls && msg.toolCalls.length > 0 && (
        <div style={{ marginTop: 8, paddingTop: 6, borderTop: "1px dashed var(--serif-border, #e8e8e8)" }}>
          {msg.toolCalls.map((tc, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--serif-text-secondary, #888)", padding: "1px 0" }}>
              <ToolOutlined style={{ fontSize: 10 }} />
              <Tag style={{ fontSize: 10, lineHeight: "16px", margin: 0 }}>{toolLabel(tc.name)}</Tag>
              <span>{tc.status === "ok" ? "完成" : "失败"}</span>
              {tc.duration_ms != null && <span>· {tc.duration_ms.toFixed(0)}ms</span>}
              {tc.summary && <span>· {tc.summary}</span>}
            </div>
          ))}
        </div>
      )}

      {/* Citations */}
      {msg.citations && msg.citations.length > 0 && (
        <Collapse
          size="small"
          ghost
          style={{ marginTop: 6 }}
          items={[{
            key: "citations",
            label: <span style={{ fontSize: 11, color: "var(--serif-text-secondary, #888)" }}><FileTextOutlined /> 原始证据 {msg.citations.length} 条</span>,
            children: (
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {msg.citations.map((c, i) => (
                  <CitationItem key={i} citation={c} />
                ))}
              </div>
            ),
          }]}
        />
      )}

      {/* Products referenced */}
      {msg.productsReferenced && msg.productsReferenced.length > 0 && (
        <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 4 }}>
          {msg.productsReferenced.map((p) => (
            <Tag key={p.id} style={{ fontSize: 10, margin: 0 }}>{p.name}</Tag>
          ))}
        </div>
      )}

      <div style={{ fontSize: 10, marginTop: 4, opacity: 0.6 }}>{msg.timestamp}</div>
    </div>
  );
}

function AgentActions({ msg, onOpenProduct, onReviewProduct }: { msg: DisplayMessage; onOpenProduct: (productId: string) => void; onReviewProduct: (productId: string) => void }): React.JSX.Element | null {
  const products = msg.productsReferenced ?? [];
  const allocationProducts = msg.allocationDraft?.allocations ?? [];
  const requiresNavReview = msg.toolCalls?.some((tool) => tool.name === "validate_nav_quality" && tool.status !== "ok") || msg.content.includes("人工复核");
  if (products.length === 0 && allocationProducts.length === 0) return null;
  return <Space wrap size={[6, 6]} style={{ marginTop: 8 }}>
    {requiresNavReview && products.slice(0, 3).map((product) => <Button key={`review-${product.id}`} size="small" type="primary" onClick={() => onReviewProduct(product.id)}>复核净值</Button>)}
    {products.slice(0, 3).map((product) => <Button key={product.id} size="small" icon={<ProfileOutlined />} onClick={() => onOpenProduct(product.id)}>查看 {product.name}</Button>)}
  </Space>;
}

function AllocationDraftCard({ draft }: { draft: ChatAllocationDraft }): React.JSX.Element {
  const portfolio = draft.portfolio;
  const [saved, setSaved] = useState(false);
  const [decisionId, setDecisionId] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [saving, setSaving] = useState(false);
  const save = async (): Promise<void> => {
    setSaving(true);
    try {
      const result = await kbSaveAllocationDraft(draft.draft_id);
      setDecisionId(result.id);
      setSaved(true);
      message.success("配置草案已保存，尚未提交或执行。");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "保存配置草案失败");
    } finally {
      setSaving(false);
    }
  };
  const submit = async (): Promise<void> => {
    if (!decisionId) return;
    setSaving(true);
    try {
      await kbSubmitAllocationDraft(decisionId);
      setSubmitted(true);
      message.success("已提交人工确认；通过后才会计算评分、冻结版本并开启跟踪。");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "提交人工确认失败");
    } finally {
      setSaving(false);
    }
  };
  return (
    <div style={{ marginTop: 10, padding: 10, borderRadius: 6, border: "1px solid var(--serif-border)", background: "var(--serif-card-bg)" }}>
      <div style={{ fontSize: 12, fontWeight: 600 }}>{draft.goal}</div>
      {draft.allocations.length > 0 && (
        <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
          {draft.allocations.map((item) => (
            <div key={item.product_id} style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: 8, fontSize: 12 }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.product_name}<span style={{ color: "var(--serif-text-secondary)", fontSize: 11 }}> · {item.rationale}</span></span>
              <strong>{(item.weight * 100).toFixed(1)}%</strong>
            </div>
          ))}
        </div>
      )}
      {portfolio && <div style={{ marginTop: 10 }}>
        <Space wrap size={[12, 4]} style={{ fontSize: 12 }}>
          <span>年化收益 {(portfolio.metrics.annualized_return * 100).toFixed(1)}%</span>
          <span>年化波动 {(portfolio.metrics.annualized_volatility * 100).toFixed(1)}%</span>
          <span>夏普 {portfolio.metrics.sharpe_ratio?.toFixed(2) ?? "—"}</span>
          <span>最大回撤 {(portfolio.metrics.maximum_drawdown * 100).toFixed(1)}%</span>
        </Space>
        <ReactECharts
          option={{
            grid: { left: 38, right: 12, top: 16, bottom: 24 },
            tooltip: { trigger: "axis", valueFormatter: (value: number) => value.toFixed(3) },
            xAxis: { type: "category", data: portfolio.nav.map((point) => point.date), axisLabel: { fontSize: 10, hideOverlap: true } },
            yAxis: { type: "value", scale: true, axisLabel: { formatter: (value: number) => value.toFixed(2) } },
            series: [{ type: "line", data: portfolio.nav.map((point) => point.nav), showSymbol: false, lineStyle: { color: "#2563EB", width: 2 }, areaStyle: { color: "rgba(37, 99, 235, 0.08)" } }],
          }}
          style={{ height: 180, marginTop: 6 }}
        />
      </div>}
      {draft.risk_warnings.length > 0 && <div style={{ marginTop: 7, fontSize: 11 }}>风险提示：{draft.risk_warnings.join("；")}</div>}
      <Space size={6} style={{ marginTop: 10 }}>
        <Button size="small" type="primary" loading={saving} disabled={saved || !draft.draft_id} onClick={() => void save()}>{saved ? "已保存为草案" : "保存配置草案"}</Button>
        {saved && <Button size="small" loading={saving} disabled={submitted} onClick={() => void submit()}>{submitted ? "已提交人工确认" : "提交人工确认"}</Button>}
      </Space>
    </div>
  );
}

function toolLabel(name: string): string {
  const labels: Record<string, string> = {
    search_products: "检索资料",
    calculate_nav_metrics: "计算净值指标",
    analyze_attribution_risk: "归因与风险评价",
    llm_comparison_interpretation: "差异化对比解读",
    screen_products: "筛选产品",
    compare_products: "对比产品",
    optimize_fof_allocation: "配置优化",
    interpret_allocation_constraints: "理解配置约束",
    validate_agent_plan: "校验研究计划",
    run_allocation_agent: "执行配置 Agent",
    inspect_source_images: "读取原始图片",
    analyze_single_material: "单图证据研究",
    llm_research_synthesis: "LLM 研究解读",
  };
  return labels[name] ?? name;
}

function CitationItem({ citation }: { citation: ChatCitation }): React.JSX.Element {
  return (
    <div style={{ fontSize: 11, padding: "3px 6px", borderRadius: 4, background: "var(--serif-card-bg, #fff)", border: "1px solid var(--serif-border, #f0f0f0)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <FileTextOutlined style={{ fontSize: 10, color: "var(--serif-accent, #1677ff)" }} />
        <Text style={{ fontSize: 11 }} ellipsis>
          原始证据 · {citation.filename || "未知文件"}
          {citation.page_number != null && ` · 第 ${citation.page_number} 页`}
          {citation.fragment_type && ` · ${citation.fragment_type}`}
        </Text>
        {citation.confidence != null && (
          <Tag style={{ fontSize: 9, margin: 0, lineHeight: "14px" }}>{(citation.confidence * 100).toFixed(0)}%</Tag>
        )}
      </div>
      {citation.snippet && (
        <div style={{ marginTop: 2, color: "var(--serif-text-secondary, #888)", fontSize: 10 }}>
          {citation.snippet}
        </div>
      )}
    </div>
  );
}
