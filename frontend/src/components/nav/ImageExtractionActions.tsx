import { Alert, Button, Col, Input, InputNumber, Row, Select, Tooltip, Typography } from "antd";
import type { ImageExtractionFrequency } from "../../api";

const { Text } = Typography;

/** User-facing controls for automatic/manual chart extraction and evidence text. */
export default function ImageExtractionActions({
  pendingImage,
  isDigitizing,
  isSmartDigitizing,
  imageError,
  importMessage,
  curveLineColor,
  onCurveLineColorChange,
  isPickingCurveColor,
  onToggleColorPicker,
  reportedCumulativeReturn,
  onReportedCumulativeReturnChange,
  imageExtractionFrequency,
  onImageExtractionFrequencyChange,
  onSmartExtract,
  onOpenAdvancedCalibration,
  productName,
  onProductNameChange,
  isRecognizingProduct,
}: {
  pendingImage: boolean;
  isDigitizing: boolean;
  isSmartDigitizing: boolean;
  imageError?: string;
  importMessage?: string;
  curveLineColor: string;
  onCurveLineColorChange: (value: string) => void;
  isPickingCurveColor: boolean;
  onToggleColorPicker: () => void;
  reportedCumulativeReturn?: number;
  onReportedCumulativeReturnChange: (value: number | undefined) => void;
  imageExtractionFrequency: ImageExtractionFrequency;
  onImageExtractionFrequencyChange: (value: ImageExtractionFrequency) => void;
  onSmartExtract: () => void;
  onOpenAdvancedCalibration: () => void;
  productName: string;
  onProductNameChange: (value: string) => void;
  isRecognizingProduct: boolean;
}): React.JSX.Element {
  return (
    <>
      {imageError && <Alert type="error" showIcon message={imageError} style={{ marginBottom: 12 }} closable />}
      {importMessage && <Alert type="success" showIcon message={importMessage} style={{ marginBottom: 12 }} closable />}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
        <Button type="primary" loading={isSmartDigitizing} disabled={!pendingImage || isDigitizing} onClick={onSmartExtract}>
          自动定位并识别产品净值
        </Button>
        <Text type="secondary" style={{ fontSize: 12 }}>自动读取图框、曲线颜色、日期和纵轴范围</Text>
        <Text type="secondary" style={{ fontSize: 12 }}>产品曲线颜色</Text>
        <Select
          aria-label="选择产品曲线颜色"
          value={curveLineColor}
          onChange={onCurveLineColorChange}
          size="small"
          style={{ width: 150 }}
          options={[
            { value: "#C9983E", label: "金黄色产品线" },
            { value: "#E53935", label: "红色产品线" },
            { value: "#2F6FAE", label: "蓝色产品线" },
            { value: "#7A7A7A", label: "灰色对比线" },
          ]}
        />
        <input aria-label="自定义产品曲线颜色" type="color" value={curveLineColor} onChange={(event) => onCurveLineColorChange(event.target.value.toUpperCase())} title="自定义曲线颜色" style={{ width: 30, height: 24, padding: 1, cursor: "pointer" }} />
        <Button size="small" type={isPickingCurveColor ? "primary" : "default"} disabled={!pendingImage} onClick={onToggleColorPicker}>
          {isPickingCurveColor ? "请点击原图曲线取色" : "从图上取色"}
        </Button>
        <Tooltip title="仅当智能识别无法贴合原图时，再手动指定颜色、日期或纵轴范围">
          <Button size="small" onClick={onOpenAdvancedCalibration}>打开高级校准</Button>
        </Tooltip>
        <Text type="secondary" style={{ fontSize: 12 }}>自动失败时，再取色并打开高级校准</Text>
      </div>
      <details style={{ marginBottom: 10 }}>
        <summary style={{ cursor: "pointer", fontSize: 13 }}>识别不贴合时：取色与手动重试</summary>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", margin: "10px 0" }}>
      </div>
      <Row gutter={[12, 12]}>
        <Col xs={24} sm={7}>
          <div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginBottom: 4 }}>披露累计收益 %（可选）</div>
          <InputNumber aria-label="披露累计收益" value={reportedCumulativeReturn} onChange={(value) => onReportedCumulativeReturnChange(value ?? undefined)} min={-100} max={10000} step={0.01} style={{ width: "100%" }} />
        </Col>
        <Col xs={24} sm={5}>
          <div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginBottom: 4 }}>图片提取频率</div>
          <Select<ImageExtractionFrequency> value={imageExtractionFrequency} onChange={onImageExtractionFrequencyChange} style={{ width: "100%" }} options={[
            { value: "weekly", label: "周频" }, { value: "monthly", label: "月频" }, { value: "daily", label: "日频（仅有明确日频披露时）" }, { value: "auto", label: "自动判断" },
          ]} />
        </Col>
      </Row>
      </details>
      <Text type="secondary" style={{ display: "block", marginTop: 6, fontSize: 12 }}>
        自动识别失败时：从产品线上取色，打开高级校准，补齐日期和纵轴范围，再点击“按当前校准识别产品净值”。无需逐点画线。
      </Text>
      <Input aria-label="产品名称" value={productName} onChange={(event) => onProductNameChange(event.target.value)} placeholder="产品名称（可选；文件名或 OCR 候选，可修正）" suffix={isRecognizingProduct ? "识别中…" : undefined} style={{ marginTop: 12 }} />
      <Text type="secondary" style={{ display: "block", marginTop: 6, fontSize: 12 }}>请在同一页核对坐标与候选净值后，再开始识别。</Text>
    </>
  );
}
