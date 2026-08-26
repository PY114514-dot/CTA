import { Alert, Button, Modal, Space, Typography } from "antd";
import { AnchorMarkButton, ANCHOR_LABELS } from "../StepSection";
import ChartImageCanvas, { type ChartAnchors } from "./ChartImageCanvas";
import type { CandidateCurvePoint } from "./CandidateCurveOverlay";
import type { AnchorKind } from "../../hooks/useImageDigitization";

const { Text } = Typography;

/** Cohesive prop bundles replacing the former flat 26-prop surface. */
export interface ChartPreviewCanvasBundle {
  src?: string;
  anchors: ChartAnchors;
  candidateCurve: CandidateCurvePoint[];
  editable: boolean;
  isPicking: boolean;
  onImageClick: React.MouseEventHandler<HTMLImageElement>;
  onPointMove: (index: number, yRatio: number) => void;
}

export interface ChartPreviewZoomBundle {
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  level: number;
  onLevelChange: (value: number) => void;
}

export interface ChartPreviewEditBundle {
  isAutoFitting: boolean;
  onAutoFit: () => void;
  onToggleEdit: () => void;
  autoFitMessage?: string;
  onDismissAutoFit: () => void;
}

export interface ChartPreviewSummaryBundle {
  navCount: number;
  frequency: "daily" | "weekly" | "monthly";
  confidence?: number;
  maximumDrawdown?: string;
}

/** Single canvas mount so the 7 canvas props are forwarded once, not twice. */
function ReviewCanvas({
  src, alt, anchors, candidateCurve, editable, isPicking, onImageClick, onImageDoubleClick, onPointMove, fillWidth = false,
}: {
  src: string; alt: string; anchors: ChartAnchors; candidateCurve: CandidateCurvePoint[];
  editable: boolean; isPicking: boolean; onImageClick: React.MouseEventHandler<HTMLImageElement>;
  onImageDoubleClick?: React.MouseEventHandler<HTMLImageElement>; onPointMove: (index: number, yRatio: number) => void;
  fillWidth?: boolean;
}): React.JSX.Element {
  return (
    <ChartImageCanvas
      src={src}
      alt={alt}
      anchors={anchors}
      candidateCurve={candidateCurve}
      editable={editable}
      isPicking={isPicking}
      onImageClick={onImageClick}
      onImageDoubleClick={onImageDoubleClick}
      onPointMove={onPointMove}
      fillWidth={fillWidth}
    />
  );
}

/** Source-chart review, zoomed inspection and the lightweight three-anchor correction UI. */
export default function ChartPreviewReview({
  canvas, zoom, edit, summary, activeAnchor, onActiveAnchorChange, canMark, onToggleColorPicker, showModal = true,
}: {
  canvas: ChartPreviewCanvasBundle;
  zoom: ChartPreviewZoomBundle;
  edit: ChartPreviewEditBundle;
  summary: ChartPreviewSummaryBundle;
  activeAnchor?: AnchorKind;
  onActiveAnchorChange: (value: AnchorKind | undefined) => void;
  canMark: boolean;
  onToggleColorPicker: () => void;
  showModal?: boolean;
}): React.JSX.Element | null {
  if (!canvas.src) return null;
  const active = Boolean(activeAnchor);
  const editable = canvas.editable && !active;
  const isPicking = active || canvas.isPicking;
  return <>
    <div style={{ display: "inline-block", marginTop: 16, border: "1px solid var(--serif-border)", borderRadius: 6, background: "var(--serif-background)", maxWidth: "100%" }}>
      <ReviewCanvas src={canvas.src} alt="待校准净值图" anchors={canvas.anchors} candidateCurve={canvas.candidateCurve} editable={editable} isPicking={isPicking} onImageClick={canvas.onImageClick} onImageDoubleClick={() => { if (!active && !canvas.isPicking) zoom.onOpen(); }} onPointMove={canvas.onPointMove} />
    </div>
    <div style={{ marginTop: 6, fontSize: 12, color: "var(--serif-muted-foreground)" }}>彩色纵横线标出当前定位图框（调用结果会明确说明来自 VLM 还是 CV）；橙色虚线是仅在该框内追踪的 CV 候选曲线。若未找到候选曲线，只需从产品线上取色，并在必要时标记日期首尾，无需逐点画线。</div>
    {canvas.candidateCurve.length >= 2 && <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
      <Text type="secondary" style={{ fontSize: 12 }}>候选：{summary.navCount} 点 · {summary.frequency === "weekly" ? "周频" : summary.frequency === "monthly" ? "月频" : "日频"}{summary.confidence !== undefined ? ` · 置信度 ${(summary.confidence * 100).toFixed(0)}%` : ""}{summary.maximumDrawdown ? ` · 估算最大回撤 ${summary.maximumDrawdown}` : ""}</Text>
      <Button size="small" loading={edit.isAutoFitting} onClick={edit.onAutoFit}>自动贴合</Button>
      <Button size="small" onClick={edit.onToggleEdit}>{canvas.editable ? "退出锚点微调" : "三个锚点微调"}</Button>
    </div>}
    {edit.autoFitMessage && <Alert type={edit.autoFitMessage.startsWith("已自动") ? "success" : "warning"} showIcon message={edit.autoFitMessage} style={{ marginTop: 10 }} closable onClose={edit.onDismissAutoFit} />}
    {showModal && <Modal title="放大查看与标记" open={zoom.open} onCancel={zoom.onClose} footer={null} width="92vw" centered>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
        <div style={{ display: "flex", gap: 8, flex: 1, minWidth: 320, maxWidth: 560 }}>
          {(["start", "end", "top", "bottom"] as const).map((anchor) => <AnchorMarkButton key={anchor} anchor={anchor} active={activeAnchor === anchor} completed={canvas.anchors[`${anchor === "start" ? "startXRatio" : anchor === "end" ? "endXRatio" : anchor === "top" ? "topYRatio" : "bottomYRatio"}` as keyof ChartAnchors] !== undefined} disabled={!canMark} onClick={() => onActiveAnchorChange(anchor)} />)}
        </div>
        <Button size="small" type={canvas.isPicking ? "primary" : "default"} disabled={!canMark} onClick={onToggleColorPicker}>
          {canvas.isPicking ? "请点击产品曲线取色" : "从图上取色"}
        </Button>
        <Space size={4}><Button size="small" onClick={() => zoom.onLevelChange(Math.max(0.5, zoom.level - 0.25))}>-</Button><span style={{ fontSize: 12, minWidth: 42, textAlign: "center" }}>{Math.round(zoom.level * 100)}%</span><Button size="small" onClick={() => zoom.onLevelChange(Math.min(4, zoom.level + 0.25))}>+</Button><Button size="small" onClick={() => zoom.onLevelChange(1)}>适应宽度</Button></Space>
      </div>
      <div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginBottom: 10 }}>{canvas.isPicking ? "请在放大图中点击产品净值曲线取色。" : activeAnchor ? `在图中点击「${ANCHOR_LABELS[activeAnchor]}」对应的刻度完成标记，OCR 会自动读取标签。` : "点击上排按钮选择要标记的锚线，再点击图中对应刻度；超出屏幕的部分可滚动查看。"}</div>
      <div style={{ maxHeight: "66vh", overflow: "auto", border: "1px solid var(--serif-border)", borderRadius: 6, background: "var(--serif-background)" }}><div style={{ width: `${zoom.level * 100}%` }}><ReviewCanvas src={canvas.src} alt="放大净值图" anchors={canvas.anchors} candidateCurve={canvas.candidateCurve} editable={editable} isPicking={isPicking} onImageClick={canvas.onImageClick} onPointMove={canvas.onPointMove} fillWidth /></div></div>
    </Modal>}
    {showModal && activeAnchor && <div style={{ marginTop: 10, padding: "8px 12px", background: "var(--serif-muted)", border: "1px solid var(--serif-border)", borderRadius: 6, fontSize: 13 }}>请在上方图片中点击「{ANCHOR_LABELS[activeAnchor]}」对应的刻度位置，OCR 将自动读取标签。</div>}
  </>;
}
