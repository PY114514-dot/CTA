/** One product card and its direct workflow actions. */

import React from "react";
import { Button, Checkbox, Dropdown, Tag, Typography, message } from "antd";
import { DeleteOutlined, EditOutlined, MoreOutlined } from "@ant-design/icons";
import type { KbProduct } from "../api";
import { needsManualNaming, productNameCategories, strategyLabel } from "./productDisplay";
import { isResearchReady, productWorkflowAction, productWorkflowReason, productWorkflowStatus } from "./productWorkflow";
import { useWorkbenchActions } from "./FofWorkbench";

const { Text } = Typography;

interface Props {
  product: KbProduct;
  selected: boolean;
  filenameById: Map<string, string>;
  onToggleSelect: (id: string) => void;
  onEdit: (product: KbProduct, confirmAfter?: boolean) => void;
  onConfirm: (id: string) => void;
  onReviewCurves?: (product: KbProduct) => void;
  onReject: (id: string) => void;
  onDelete: (product: KbProduct) => void;
}

export default function ProductLibraryItem({ product, selected, filenameById, onToggleSelect, onEdit, onConfirm, onReviewCurves, onReject, onDelete }: Props): React.JSX.Element {
  const { onFocusProduct, onOpenResearch, onCalibrateProduct } = useWorkbenchActions();
  const requiresNaming = needsManualNaming(product);
  const canResearch = isResearchReady(product);
  const reviewConfidence = product.nav_count > 0 ? Math.round(product.reviewed_nav_count / product.nav_count * 100) : 0;
  const extractionConfidence = product.nav_confidence;
  const confidenceBand = product.nav_confidence_band ?? (extractionConfidence == null ? "unknown" : extractionConfidence >= 0.85 ? "high" : extractionConfidence >= 0.60 ? "medium" : "low");
  const confidenceLabel = confidenceBand === "high" ? "高" : confidenceBand === "medium" ? "中" : confidenceBand === "low" ? "低" : "未评估";
  const displayStatus = productWorkflowStatus(product);
  const workflowAction = productWorkflowAction(product, {
    editIdentity: () => onEdit(product, true),
    calibrateNav: () => onCalibrateProduct(product),
    confirmIdentity: () => onConfirm(product.id),
  });
  const primaryAction = onReviewCurves
    ? { label: "复核多条曲线", run: () => onReviewCurves(product) }
    : workflowAction;
  const sourceNames = product.source_files.length > 0
    ? product.source_files
    : (product.source_file_ids ?? []).map((id) => filenameById.get(id)).filter((name): name is string => Boolean(name));
  const nameCategories = productNameCategories(product.standard_name, product.strategy);

  return <div
    style={{ padding: "5px 10px", borderRadius: 6, border: `1px solid ${selected ? "var(--serif-accent, #1677ff)" : "var(--serif-border, #f0f0f0)"}`, marginBottom: 4, background: selected ? "var(--serif-accent-bg, #f0f5ff)" : "var(--serif-card-bg, #fff)", cursor: "pointer", transition: "border-color 0.2s" }}
    onClick={() => {
      if (requiresNaming) return void message.info("该记录尚未形成可靠产品名称，请右键选择“编辑信息”后再加入候选池。");
      if (product.confirmation_status === "rejected") return void message.info("该记录已标为不是产品，不会进入 Agent 研究上下文。");
      if (canResearch) { onFocusProduct(product); onToggleSelect(product.id); return; }
      // Clicking a not-yet-ready product must still take the user somewhere:
      // open its next workflow step (extract/review NAV, confirm identity…)
      // instead of a dead-end toast.  Only fall back to the reason when no
      // actionable step exists.
      if (primaryAction) { onFocusProduct(product); primaryAction.run(); return; }
      void message.info(productWorkflowReason(product));
    }}
  >
    <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
      <Checkbox checked={selected} disabled={!canResearch} style={{ marginTop: 2 }} onClick={(event) => event.stopPropagation()} onChange={() => { onFocusProduct(product); onToggleSelect(product.id); }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Text strong ellipsis style={{ flex: 1, fontSize: 13 }}>{product.standard_name}</Text>
          <Tag color={displayStatus.color} title={canResearch ? "打开研究档案" : "研究状态由产品身份与人工审核净值决定"} onClick={canResearch ? (event) => { event.stopPropagation(); onOpenResearch(product); } : undefined} style={{ margin: 0, fontSize: 10, lineHeight: "16px", cursor: canResearch ? "pointer" : undefined }}>{canResearch ? "查看研究" : displayStatus.label}</Tag>
          <Dropdown trigger={["click"]} menu={{ items: [
            { key: "edit", icon: <EditOutlined />, label: "编辑信息", onClick: () => onEdit(product) },
            ...(product.confirmation_status !== "rejected" ? [{ key: "reject", label: "标记为非产品", danger: true, onClick: () => onReject(product.id) }] : []),
            { key: "delete", icon: <DeleteOutlined />, label: "删除产品", danger: true, onClick: () => onDelete(product) },
          ] }}>
            <Button type="text" size="small" icon={<MoreOutlined />} aria-label={`${product.standard_name} 更多操作`} style={{ width: 24, height: 24, minWidth: 24 }} onClick={(event) => event.stopPropagation()} />
          </Dropdown>
        </div>
        <div style={{ marginTop: 2, display: "flex", gap: 8, fontSize: 11, color: "var(--serif-text-secondary, #888)" }}>
          {product.manager_name && <span>{product.manager_name}</span>}
          {strategyLabel(product.strategy) && <span>{strategyLabel(product.strategy)}</span>}
          {product.nav_methods?.[0] && <span title={product.nav_methods[0]}>来源：{product.nav_methods[0]}</span>}
        </div>
        {nameCategories.length > 0 && <div style={{ marginTop: 3 }} title="根据产品名称和已有策略生成的初步标签，不替代人工确认">
          {nameCategories.map((category) => <Tag key={category} color={category === "CTA" ? "blue" : category === "股票" ? "green" : category === "波动率" ? "purple" : "default"} style={{ marginInlineEnd: 4, fontSize: 10, lineHeight: "16px" }}>{category}</Tag>)}
        </div>}
        {requiresNaming ? <Text type="secondary" style={{ display: "block", marginTop: 2, fontSize: 11 }}>识别文本已保留，需核对并编辑正式名称</Text> : <>
          <Text type="secondary" style={{ display: "block", marginTop: 2, fontSize: 11 }}>净值 {product.nav_count} 条 · 人工复核 {reviewConfidence}% · 证据 {product.fact_count} 条</Text>
          <Text type="secondary" style={{ display: "block", marginTop: 2, fontSize: 11 }}>{product.machine_nav_review?.status === "direct_source" ? "直接周频净值来源 · 不进行图片曲线识别校验" : `资料识别 ${extractionConfidence == null ? "未评估" : `${(extractionConfidence * 100).toFixed(0)}%（${confidenceLabel}）`} · 仅反映图表/资料解析质量`}</Text>
          {product.machine_nav_review && product.machine_nav_review.status !== "direct_source" && <Text type={product.machine_nav_review.machine_reviewed ? "success" : "warning"} style={{ display: "block", marginTop: 2, fontSize: 11 }}>{product.machine_nav_review.machine_reviewed ? "机器复核通过：可进入初步研究（非人工事实核验）" : `需人工处理：${product.machine_nav_review.reasons[0] ?? "机器复核未通过"}`}</Text>}
          {confidenceBand === "low" && reviewConfidence < 100 && <Text type="warning" style={{ display: "block", marginTop: 2, fontSize: 11 }}>资料识别置信度较低；仍需完成净值人工复核</Text>}
          {product.confirmation_status === "pending" && <div style={{ marginTop: 3, fontSize: 11, lineHeight: 1.5, color: "var(--serif-text-secondary, #888)" }}><div>识别范围：{product.nav_start && product.nav_end ? `${product.nav_start} 至 ${product.nav_end}` : "未识别日期"} · {product.reviewed_nav_count}/{product.nav_count} 已复核</div><div>来源：{sourceNames.length > 0 ? sourceNames.join("、") : "未关联原始文件"}</div></div>}
          {primaryAction && <Button type="primary" size="small" block style={{ marginTop: 7 }} onClick={(event) => { event.stopPropagation(); primaryAction.run(); }}>{primaryAction.label}</Button>}
        </>}
      </div>
    </div>
  </div>;
}
