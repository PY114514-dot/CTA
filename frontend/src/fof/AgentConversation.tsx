/** Center panel: Agent conversation with real backend, tool trace and citations. */

import React, { useCallback, useRef, useState } from "react";
import { Button, Card, Collapse, Empty, Input, Space, Tag, Typography, message } from "antd";
import {
  FileTextOutlined,
  RobotOutlined,
  SendOutlined,
  ToolOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { kbChat, kbUploadFile, type ChatAllocationDraft, type ChatCitation, type ChatMessage, type ChatToolCall } from "../api";

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

interface Props {
  selectedProductIds: string[];
  onDropProduct: (productId: string) => void;
  onMaterialAnalyzed?: () => void;
}

export default function AgentConversation({ selectedProductIds, onDropProduct, onMaterialAnalyzed }: Props): React.JSX.Element {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const materialInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });
  }, []);

  const handleSend = useCallback(async (presetQuery?: string) => {
    const query = presetQuery ?? input.trim();
    if (!query || loading) return;
    const now = new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    setMessages((prev) => [...prev, { role: "user", content: query, timestamp: now }]);
    setInput("");
    setLoading(true);
    scrollToBottom();

    try {
      const response: ChatMessage = await kbChat({
        query,
        product_ids: selectedProductIds,
      });
      setMessages((prev) => [
        ...prev,
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
      message.error(err instanceof Error ? err.message : "Agent 对话失败");
      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          content: "抱歉，处理您的问题时出现错误。请检查后端服务是否正常运行。",
          timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  }, [input, loading, selectedProductIds, scrollToBottom]);

  const handleMaterialUpload = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      message.warning("单图研究目前支持 PNG、JPG 或 WEBP 图片。");
      return;
    }
    if (loading) return;
    const now = new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    setMessages((prev) => [...prev, { role: "user", content: `请研究我上传的素材《${file.name}》`, timestamp: now }]);
    setLoading(true);
    scrollToBottom();
    try {
      const result = await kbUploadFile(file);
      setMessages((prev) => [...prev, {
        role: "agent",
        content: `已将素材《${file.name}》送入产品库解析。解析完成后，它会直接出现在左侧“产品库”中；如识别出净值曲线，请点击产品卡的“复核 / 校准净值”完成确认。`,
        timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
        toolCalls: [{ name: "register_product_material", status: "ok", duration_ms: null, summary: `资料已登记，当前状态：${result.parsing_status}` }],
      }]);
      onMaterialAnalyzed?.();
    } catch (err) {
      const detail = err instanceof Error ? err.message : "单图研究失败";
      message.error(detail);
      setMessages((prev) => [...prev, { role: "agent", content: `未能完成该图片研究：${detail}`, timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) }]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  }, [loading, onMaterialAnalyzed, scrollToBottom]);

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
            onDropProduct(product.id);
            void handleSend(
              `请基于产品「${product.name}」的已导入资料，概述策略、收益风险特征，并提示需要补充的数据。`,
            );
          }
        } catch {
          // Ignore drops that do not originate from a product information card.
        }
      }}
      style={{ width: "100%", flex: 1, height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}
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
      <div ref={scrollRef} style={{ flex: 1, overflow: "auto", minHeight: 0, marginBottom: 12 }}>
        {messages.length === 0 ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<Space direction="vertical" size={8}>
            <Text type="secondary">直接描述你的筛选目标</Text>
            <Text type="secondary" style={{ fontSize: 11 }}>例如：“我想控制回撤在 10% 内，并看好商品 CTA”；或“比较已选产品”。</Text>
            <Space wrap>
              <Button size="small" onClick={() => void handleSend("我想控制回撤在 10% 内，请给出初始 FOF 配置和各产品占比。")}>控制回撤</Button>
              <Button size="small" onClick={() => void handleSend("我看好商品 CTA，请筛选适合的产品并给出初始配置占比。")}>看好商品 CTA</Button>
              <Button size="small" onClick={() => void handleSend("请比较我已选的产品：收益、回撤、相关性和适合的配置角色。")}>比较已选产品</Button>
            </Space>
          </Space>} style={{ marginTop: 42 }} />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {messages.map((msg, index) => (
              <MessageBubble key={index} msg={msg} />
            ))}
            {loading && (
              <div style={{ alignSelf: "flex-start", padding: "8px 12px", borderRadius: 8, background: "var(--serif-card-bg, #fafafa)", border: "1px solid var(--serif-border, #f0f0f0)" }}>
                <Text type="secondary" style={{ fontSize: 12 }}>Agent 正在检索与分析...</Text>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input area */}
      <div style={{ flexShrink: 0, display: "flex", gap: 8 }}>
        <TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="输入研究问题或配置需求..."
          autoSize={{ minRows: 1, maxRows: 4 }}
          onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); void handleSend(); } }}
          style={{ flex: 1 }}
        />
        <Button title="添加一张周报图片" icon={<UploadOutlined />} aria-label="添加周报图片" onClick={() => materialInputRef.current?.click()} disabled={loading} style={{ alignSelf: "flex-end" }} />
        <Button
          icon={<SendOutlined />}
          onClick={() => void handleSend()}
          disabled={!input.trim() || loading}
          style={{ alignSelf: "flex-end" }}
        />
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------


function MessageBubble({ msg }: { msg: DisplayMessage }): React.JSX.Element {
  const isUser = msg.role === "user";
  return (
    <div
      style={{
        alignSelf: isUser ? "flex-end" : "flex-start",
        maxWidth: "88%",
        padding: "8px 12px",
        borderRadius: 8,
        background: isUser ? "var(--serif-accent, #1677ff)" : "var(--serif-card-bg, #fafafa)",
        color: isUser ? "var(--serif-on-accent, #fff)" : undefined,
        border: isUser ? undefined : "1px solid var(--serif-border, #f0f0f0)",
      }}
    >
      {/* Content */}
      <div style={{ fontSize: 13, whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{msg.content}</div>

      {/* Tool calls */}
      {msg.allocationDraft && <AllocationDraftCard draft={msg.allocationDraft} />}

      {msg.dataContext && msg.dataContext.length > 0 && (
        <div style={{ marginTop: 8, padding: "6px 8px", borderRadius: 6, background: "var(--serif-muted, #f7f8fa)", border: "1px solid var(--serif-border, #e8e8e8)" }}>
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 3 }}>本次实际使用的数据</div>
          {msg.dataContext.map((item) => (
            <div key={item.product_id} style={{ fontSize: 11, lineHeight: 1.55, color: "var(--serif-text-secondary, #666)" }}>
              {item.name} · {item.nav_count} 点{item.frequency ? ` · ${item.frequency}` : ""}
              {item.start_date && item.end_date ? ` · ${item.start_date} 至 ${item.end_date}` : ""}
              {` · ${item.review_status}`}{item.source_files.length > 0 ? ` · 来源：${item.source_files.join("、")}` : " · 未关联来源文件"}
            </div>
          ))}
        </div>
      )}

      {msg.methodProvenance && Object.keys(msg.methodProvenance).length > 0 && (
        <Collapse
          size="small"
          ghost
          style={{ marginTop: 6 }}
          items={[{
            key: "methods",
            label: <span style={{ fontSize: 11, color: "var(--serif-text-secondary, #888)" }}>本次回答的方法来源</span>,
            children: (
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                {Object.entries(msg.methodProvenance).map(([key, item]) => (
                  <div key={key} style={{ fontSize: 11, color: "var(--serif-text-secondary, #666)" }}>
                    {key}：{item.method ?? "未记录"}{item.model ? `（${item.model}）` : ""}
                    {item.detail ? ` · ${item.detail}` : ""}
                  </div>
                ))}
              </div>
            ),
          }]}
        />
      )}

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
            label: <span style={{ fontSize: 11, color: "var(--serif-text-secondary, #888)" }}><FileTextOutlined /> {msg.citations.length} 条证据引用</span>,
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

function AllocationDraftCard({ draft }: { draft: ChatAllocationDraft }): React.JSX.Element {
  return (
    <div style={{ marginTop: 10, padding: 10, borderRadius: 6, border: "1px solid var(--serif-border)", background: "var(--serif-card-bg)" }}>
      <div style={{ fontSize: 12, fontWeight: 600 }}>{draft.goal}</div>
      <div style={{ marginTop: 5, fontSize: 11, color: "var(--serif-text-secondary)" }}>硬约束：{draft.hard_constraints.join("；")}</div>
      {draft.soft_preferences.length > 0 && <div style={{ marginTop: 3, fontSize: 11, color: "var(--serif-text-secondary)" }}>软偏好：{draft.soft_preferences.join("；")}</div>}
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
      {draft.exclusions.length > 0 && <div style={{ marginTop: 7, fontSize: 11, color: "var(--serif-text-secondary)" }}>未纳入：{draft.exclusions.map((item) => `${item.product_name}（${item.reasons.join("；")}）`).join("；")}</div>}
      {draft.risk_warnings.length > 0 && <div style={{ marginTop: 7, fontSize: 11 }}>风险提示：{draft.risk_warnings.join("；")}</div>}
      {draft.due_diligence_gaps.length > 0 && <div style={{ marginTop: 5, fontSize: 11, color: "var(--serif-text-secondary)" }}>待补尽调：{draft.due_diligence_gaps.join("；")}</div>}
      {draft.snapshot_id && <div style={{ marginTop: 6, fontSize: 10, color: "var(--serif-text-secondary)" }}>已绑定研究快照 {draft.snapshot_id}</div>}
    </div>
  );
}

function toolLabel(name: string): string {
  const labels: Record<string, string> = {
    search_products: "检索资料",
    calculate_nav_metrics: "计算净值指标",
    screen_products: "筛选产品",
    compare_products: "对比产品",
    optimize_fof_allocation: "配置优化",
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
          {citation.filename || "未知文件"}
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
