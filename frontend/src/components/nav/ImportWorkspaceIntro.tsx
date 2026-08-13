import { Alert, Button, Col, Row, Typography } from "antd";

const { Paragraph, Text } = Typography;

/** Compact entry/exit affordances for the optional import and calibration workspace. */
export function ImportWorkspaceClosed({ canAnalyze, isAnalyzing, error, onOpen, onAnalyze }: { canAnalyze: boolean; isAnalyzing: boolean; error?: string; onOpen: () => void; onAnalyze: () => void }): React.JSX.Element {
  return <div style={{ padding: "8px 0 4px" }}>
    <Paragraph type="secondary" style={{ marginBottom: 16 }}>从工作台选定产品后，在这里导入或校准净值。图片曲线、PDF 和表格都是同一产品资料的不同来源。</Paragraph>
    <Row gutter={[12, 12]}><Col xs={24} sm={12}><Button type="primary" block onClick={onOpen}>导入 / 校准净值</Button></Col><Col xs={24} sm={12}><Button block disabled={!canAnalyze} loading={isAnalyzing} onClick={onAnalyze}>开始因子分析</Button></Col></Row>
    {error && <Alert type="error" showIcon message={error} style={{ marginTop: 16 }} closable />}
  </div>;
}

export function ImportWorkspaceOpenHeader(): React.JSX.Element {
  return <Paragraph type="secondary">上传净值表可直接计算；上传曲线图片则进入 VLM 定位与曲线识别。</Paragraph>;
}
