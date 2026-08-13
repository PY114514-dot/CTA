import { Button, Col, Input, InputNumber, Row, Select, Typography } from "antd";
import { AnchorMarkButton, ANCHOR_LINE_COLORS, StepSection } from "../StepSection";
import type { AnchorKind } from "../../hooks/useImageDigitization";

const { Paragraph } = Typography;

/** Optional coordinate calibration. Kept separate from automatic extraction controls. */
export default function AdvancedCalibrationPanel({
  visible,
  valueMode,
  onValueModeChange,
  activeAnchor,
  onActiveAnchorChange,
  canMark,
  anchors,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  navMin,
  onNavMinChange,
  navMax,
  onNavMaxChange,
  canStartRecognition,
  isDigitizing,
  onStartRecognition,
  missingFields,
  ocrHints,
}: {
  visible: boolean;
  valueMode: "nav" | "cumulative_return";
  onValueModeChange: (value: "nav" | "cumulative_return") => void;
  activeAnchor?: AnchorKind;
  onActiveAnchorChange: (anchor: AnchorKind) => void;
  canMark: boolean;
  anchors: { start?: number; end?: number; top?: number; bottom?: number };
  startDate: string;
  onStartDateChange: (value: string) => void;
  endDate: string;
  onEndDateChange: (value: string) => void;
  navMin?: number;
  onNavMinChange: (value: number | undefined) => void;
  navMax?: number;
  onNavMaxChange: (value: number | undefined) => void;
  canStartRecognition: boolean;
  isDigitizing: boolean;
  onStartRecognition: () => void;
  missingFields: string[];
  ocrHints: Partial<Record<AnchorKind, string>>;
}): React.JSX.Element {
  const inputStyle = (anchor: AnchorKind) => activeAnchor === anchor ? { borderColor: ANCHOR_LINE_COLORS[anchor] } : undefined;
  return (
    <StepSection step={2} title="高级校准（仅曲线不贴合时使用）" visible={visible}>
      <Select value={valueMode} onChange={onValueModeChange} placeholder="请选择纵轴口径" style={{ width: "100%", maxWidth: 280 }} options={[
        { value: "nav", label: "纵轴：单位净值" }, { value: "cumulative_return", label: "纵轴：累计收益率 (%)" },
      ]} />
      <Paragraph type="secondary" style={{ marginTop: 10, marginBottom: 12, fontSize: 13 }}>
        调整曲线前，请先标记纵轴最大值和最小值；随后可直接拖动橙色点使候选线贴合原图。日期和四个锚点仅在自动结果不可靠时才需要校准。
      </Paragraph>
      <div style={{ display: "flex", gap: 8 }}>
        {(["start", "end", "top", "bottom"] as const).map((anchor) => (
          <AnchorMarkButton key={anchor} anchor={anchor} active={activeAnchor === anchor} completed={anchors[anchor] !== undefined} disabled={!canMark} onClick={() => onActiveAnchorChange(anchor)} />
        ))}
      </div>
      <Row gutter={[12, 12]} style={{ marginTop: 14 }}>
        <Col xs={12} sm={6}>
          <FieldLabel label="首日期"><Input aria-label="图片起始日期" value={startDate} onChange={(event) => onStartDateChange(event.target.value)} placeholder="YYYY-MM-DD" status={missingFields.includes("首日期") ? "error" : undefined} style={inputStyle("start")} /></FieldLabel>
          <OcrHint text={ocrHints.start} />
        </Col>
        <Col xs={12} sm={6}>
          <FieldLabel label="末日期"><Input aria-label="图片结束日期" value={endDate} onChange={(event) => onEndDateChange(event.target.value)} placeholder="YYYY-MM-DD" status={missingFields.includes("末日期") ? "error" : undefined} style={inputStyle("end")} /></FieldLabel>
          <OcrHint text={ocrHints.end} />
        </Col>
        <Col xs={12} sm={6}>
          <FieldLabel label={`最大值${valueMode === "cumulative_return" ? " (%)" : ""}`}><InputNumber aria-label="纵轴最大值" value={navMax} onChange={(value) => onNavMaxChange(value ?? undefined)} step={0.01} status={missingFields.includes("最大值") ? "error" : undefined} style={{ width: "100%", ...inputStyle("top") }} /></FieldLabel>
          <OcrHint text={ocrHints.top} />
        </Col>
        <Col xs={12} sm={6}>
          <FieldLabel label={`最小值${valueMode === "cumulative_return" ? " (%)" : ""}`}><InputNumber aria-label="纵轴最小值" value={navMin} onChange={(value) => onNavMinChange(value ?? undefined)} step={0.01} status={missingFields.includes("最小值") ? "error" : undefined} style={{ width: "100%", ...inputStyle("bottom") }} /></FieldLabel>
          <OcrHint text={ocrHints.bottom} />
        </Col>
      </Row>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 16 }}>
        <Button type="primary" loading={isDigitizing} disabled={!canStartRecognition} onClick={onStartRecognition}>
          按当前校准识别产品净值
        </Button>
        <span style={{ fontSize: 12, color: "var(--serif-muted-foreground)" }}>
          填完首末日期、最大值和最小值后，按此按钮执行手动校准识别。
        </span>
      </div>
      <div style={{ marginTop: 10, fontSize: 12, color: "var(--serif-muted-foreground)" }}>
        {anchors.start !== undefined && anchors.end !== undefined ? "✓ 横轴已校准" : "○ 横轴未校准（使用曲线首尾）"}
        <span style={{ margin: "0 8px" }}>·</span>
        {anchors.top !== undefined && anchors.bottom !== undefined ? "✓ 纵轴已校准" : "○ 纵轴未校准（使用自动图框）"}
      </div>
    </StepSection>
  );
}

function FieldLabel({ label, children }: { label: string; children: React.ReactNode }): React.JSX.Element {
  return <><div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginBottom: 4 }}>{label}</div>{children}</>;
}

function OcrHint({ text }: { text?: string }): React.JSX.Element | null {
  return text ? <div style={{ fontSize: 11, color: "#A8503F", marginTop: 3 }}>{text}</div> : null;
}
