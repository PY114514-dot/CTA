/** Compact material intake; review happens in the dedicated product-review workspace. */

import React from "react";
import { Button, Card, List, Progress, Select, Space, Tag, Typography, Upload } from "antd";
import { FileOutlined, InboxOutlined, LoadingOutlined, UploadOutlined } from "@ant-design/icons";
import type { KbFile } from "../api";

const { Text } = Typography;
const { Dragger } = Upload;

export interface UploadBatchSummary {
  total: number;
  uploaded: number;
  completed: number;
  failed: number;
}

function taskState(file: KbFile, activeFileIds: ReadonlySet<string>): { label: string; detail: string; color: "processing" | "warning" | "error" | "success" | "default" } {
  if (file.parsing_status === "processing") {
    return activeFileIds.has(file.id)
      ? { label: "后台解析中", detail: "正在提取文本、图表和候选曲线", color: "processing" }
      : { label: "等待解析", detail: "已上传，正在等待后台处理槽位", color: "warning" };
  }
  if (file.parsing_status === "completed") {
    if ((file.nav_count ?? 0) >= 2) {
      return { label: "等待复核", detail: `已提取 ${file.nav_count} 条候选净值，确认后才会进入研究`, color: "success" };
    }
    if ((file.unbound_curve_count ?? 0) > 0) {
      return { label: "确认曲线归属", detail: `${file.unbound_curve_count} 条曲线尚未绑定到产品`, color: "warning" };
    }
    if (file.workflow?.next_action === "confirm_product_binding") {
      return { label: "确认产品", detail: "已完成解析，等待确认资料对应的产品", color: "warning" };
    }
    return { label: "已完成解析", detail: "资料已保留，未提取到可复核的净值", color: "default" };
  }
  if (file.parsing_status === "completed_no_nav") {
    return { label: "需要校准", detail: file.parsing_error || "未找到可直接使用的曲线，可在原图中校准", color: "warning" };
  }
  if (file.parsing_status === "failed") {
    return { label: "解析失败", detail: file.parsing_error || "请重试或换一份清晰资料", color: "error" };
  }
  return { label: "等待提交", detail: "资料正在上传到工作区", color: "default" };
}

interface Props {
  hasQueue: boolean;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  summary: string;
  nextTaskLabel: string;
  onNextTask: () => void;
  uploadCount: number;
  onStageUpload: (file: File) => void;
  materialNature: string | undefined;
  onMaterialNatureChange: (value: string | undefined) => void;
  batch: UploadBatchSummary | null;
  batchPercent: number;
  batchActiveCount: number;
  batchNavExtractedCount: number;
  /** Files whose status is ``processing``, so the user sees names, not a number. */
  processingNames?: string[];
  /** Backend parse-slot budget from /files/ingestion-queue, if available. */
  maxConcurrency?: number;
  queuedCount?: number;
  queueFiles: KbFile[];
  activeFileIds?: string[];
  onClassifyFile: (fileId: string, materialNature: "CTA策略介绍" | "CTA研究/市场资料" | "忽略") => void;
}

export default function MaterialUploadCard({ hasQueue, expanded, onExpandedChange, summary, nextTaskLabel, onNextTask, uploadCount, onStageUpload, materialNature, onMaterialNatureChange, batch, batchPercent, batchActiveCount, batchNavExtractedCount, processingNames = [], maxConcurrency, queuedCount, queueFiles, activeFileIds = [], onClassifyFile }: Props): React.JSX.Element {
  const terminalCount = batch ? batch.completed + batch.failed : 0;
  const parsing = processingNames.length > 0;
  const activeIds = new Set(activeFileIds);
  const capacityText = maxConcurrency != null
    ? `（后台异步，最多 ${maxConcurrency} 份并行${queuedCount ? `，排队 ${queuedCount}` : ""}）`
    : "（后台异步识别）";
  return <Card size="small" style={{ flexShrink: 0 }} styles={{ body: { padding: hasQueue && !expanded ? "6px 8px" : 10 } }}>
    {hasQueue && !expanded ? <div style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 32, flexWrap: "wrap" }}>
      <Space size={8} style={{ flex: "1 1 260px", minWidth: 0 }}>
        {parsing ? <LoadingOutlined style={{ color: "var(--serif-accent, #1677ff)" }} /> : <FileOutlined style={{ color: "var(--serif-accent, #1677ff)" }} />}
        <div style={{ minWidth: 0 }}>
          <Text strong style={{ display: "block", fontSize: 12 }}>资料处理</Text>
          <Text type="secondary" ellipsis style={{ display: "block", fontSize: 11 }}>{summary}</Text>
        </div>
      </Space>
      <Space size={6} wrap>
        <Button size="small" onClick={() => onExpandedChange(true)}>查看队列</Button>
        <Button size="small" type="primary" onClick={onNextTask}>{nextTaskLabel}</Button>
        <Button size="small" icon={<UploadOutlined />} onClick={() => onExpandedChange(true)}>上传资料</Button>
      </Space>
    </div> : <>
      <Space direction="vertical" size={4} style={{ width: "100%", marginBottom: 8 }}>
        <Text strong style={{ fontSize: 12 }}>资料用途</Text>
        <Select
          size="small"
          value={materialNature ?? "nav"}
          onChange={(value) => onMaterialNatureChange(value === "nav" ? undefined : value)}
          options={[
            { value: "nav", label: "产品净值资料（识别曲线 / 表格）" },
            { value: "CTA策略介绍", label: "产品 / 策略介绍（仅保留研究证据）" },
            { value: "CTA研究/市场资料", label: "市场 / 行业资料（仅保留研究证据）" },
          ]}
        />
        <Text type="secondary" style={{ fontSize: 11 }}>介绍和市场资料不会进入净值校准队列。</Text>
      </Space>
      <Dragger multiple showUploadList={false} beforeUpload={(file) => { onStageUpload(file); return false; }} disabled={uploadCount > 0} style={{ padding: "6px 0" }}>
        <p className="ant-upload-drag-icon" style={{ marginBottom: 2 }}><InboxOutlined style={{ color: "var(--serif-accent, #1677ff)" }} /></p>
        <p style={{ margin: 0, fontSize: 12, color: "var(--serif-text-secondary, #666)" }}>{uploadCount > 0 ? `上传中 (${uploadCount})...` : "拖拽或点击上传 PDF / 图片 / XLSX / CSV / DOCX / PPTX"}</p>
      </Dragger>
      {batch && <div style={{ marginTop: 8 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, fontSize: 11 }}>
          <Text style={{ fontSize: 11 }}>本批 {batch.total} 份：已上传 {batch.uploaded}/{batch.total}{batchActiveCount > 0 ? ` · 识别中 ${batchActiveCount}` : ""}{batch.completed > 0 ? ` · 已完成 ${batch.completed}` : ""}{batch.failed > 0 ? ` · 失败 ${batch.failed}` : ""}{terminalCount > 0 ? ` · 净值已提取 ${batchNavExtractedCount}` : ""}</Text>
          {terminalCount === batch.total && <Tag color={batch.failed ? "warning" : "success"} style={{ margin: 0, fontSize: 10 }}>{batch.failed ? "解析完成，含失败" : "解析完成"}</Tag>}
        </div>
        <Progress percent={batchPercent} size="small" status={batch.failed > 0 ? "exception" : terminalCount === batch.total ? "success" : "active"} showInfo={false} style={{ margin: "3px 0 0" }} />
      </div>}
      {parsing && <div style={{ marginTop: 6, display: "flex", alignItems: "flex-start", gap: 6, fontSize: 11, color: "var(--serif-text-secondary, #888)" }}>
        <LoadingOutlined style={{ color: "var(--serif-accent, #1677ff)", marginTop: 2 }} />
        <span>识别中{capacityText}：{processingNames.slice(0, 2).join("、")}{processingNames.length > 2 ? ` 等 ${processingNames.length} 份` : ""}，完成后自动进入复核队列</span>
      </div>}
      {hasQueue && <Card size="small" type="inner" title="后台处理队列" style={{ marginTop: 10 }} extra={maxConcurrency != null ? <Text type="secondary" style={{ fontSize: 11 }}>运行 {activeFileIds.length}/{maxConcurrency}{queuedCount ? ` · 排队 ${queuedCount}` : ""}</Text> : null}>
        <Text type="secondary" style={{ display: "block", fontSize: 11, marginBottom: 6 }}>上传后会自动解析；只有人工复核通过的净值才会进入研究。</Text>
        <List
          size="small"
          dataSource={queueFiles}
          locale={{ emptyText: "暂无待处理资料" }}
          style={{ maxHeight: 190, overflowY: "auto" }}
          renderItem={(file) => {
            const state = taskState(file, activeIds);
            return <List.Item key={file.id} style={{ padding: "7px 0" }}>
              <div style={{ minWidth: 0, width: "100%" }}>
                <Space size={6} wrap={false} style={{ maxWidth: "100%" }}>
                  <Text ellipsis style={{ maxWidth: 210, fontSize: 12 }}>{file.filename}</Text>
                  <Tag color={state.color} style={{ margin: 0, flexShrink: 0 }}>{state.label}</Tag>
                </Space>
                <Text type="secondary" ellipsis style={{ display: "block", fontSize: 11, marginTop: 2 }}>{state.detail}</Text>
                {file.parsing_status !== "processing" && <Space size={4} style={{ marginTop: 4 }}>
                  <Button size="small" type="link" onClick={() => onClassifyFile(file.id, "CTA策略介绍")}>仅作研究证据</Button>
                  <Button size="small" type="link" danger onClick={() => onClassifyFile(file.id, "忽略")}>移出待办</Button>
                </Space>}
              </div>
            </List.Item>;
          }}
        />
      </Card>}
      {hasQueue && <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginTop: 8 }}>
        <Text type="secondary" ellipsis style={{ flex: 1, fontSize: 12 }}>{summary}</Text>
        <Space size={4}><Button size="small" type="primary" onClick={onNextTask}>{nextTaskLabel}</Button><Button size="small" type="text" onClick={() => onExpandedChange(false)}>收起</Button></Space>
      </div>}
    </>}
  </Card>;
}
