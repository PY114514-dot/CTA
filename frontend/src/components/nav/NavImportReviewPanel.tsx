import { Alert, Button, Checkbox, Col, Divider, Input, Row, Typography } from "antd";

const { Paragraph, Text } = Typography;

/** Direct NAV-table import and the final human-review gate before research. */
export default function NavImportReviewPanel({
  navText,
  onNavTextChange,
  conflictReasons,
  conflictConfirmed,
  onConflictConfirmedChange,
  isCalibrating,
  isSaving,
  isAnalyzing,
  canAnalyze,
  onSaveReviewed,
  onAnalyze,
  analysisError,
  onDismissAnalysisError,
  navCount,
}: {
  navText: string;
  onNavTextChange: (value: string) => void;
  conflictReasons: string[];
  conflictConfirmed: boolean;
  onConflictConfirmedChange: (value: boolean) => void;
  isCalibrating: boolean;
  isSaving: boolean;
  isAnalyzing: boolean;
  canAnalyze: boolean;
  onSaveReviewed: () => void;
  onAnalyze: () => void;
  analysisError?: string;
  onDismissAnalysisError: () => void;
  navCount: number;
}): React.JSX.Element {
  return (
    <>
      <Divider titlePlacement="left">确认净值数据</Divider>
      <Paragraph type="secondary" style={{ fontSize: 13, marginTop: 0 }}>
        已上传的 CSV/XLSX 会自动填入这里；也可粘贴或复核日期、单位净值后计算。
      </Paragraph>
      <Input.TextArea value={navText} onChange={(event) => onNavTextChange(event.target.value)} rows={12} aria-label="净值数据" placeholder="例如：2024-01-05,1.0000" />
      {conflictReasons.length > 0 && (
        <Alert
          type="warning"
          showIcon
          style={{ marginTop: 12 }}
          message="当前候选曲线与材料披露不一致，不能直接进入因子研究"
          description={<><div>{conflictReasons.join("；")}</div><Checkbox checked={conflictConfirmed} onChange={(event) => onConflictConfirmedChange(event.target.checked)} style={{ marginTop: 8 }}>我已在原图核对关键点，并确认当前净值为人工复核版本</Checkbox></>}
        />
      )}
      <Row gutter={12} style={{ marginTop: 16 }}>
        {isCalibrating && <Col xs={24} sm={12}><Button block type="primary" loading={isSaving} disabled={!canAnalyze || isSaving} onClick={onSaveReviewed}>确认并用于研究</Button></Col>}
        <Col xs={24} sm={isCalibrating ? 12 : 24}><Button block onClick={onAnalyze} loading={isAnalyzing}>计算</Button></Col>
      </Row>
      {analysisError && <Alert type="error" showIcon message={analysisError} style={{ marginTop: 12 }} closable onClose={onDismissAnalysisError} />}
      {isCalibrating && <Text type="secondary" style={{ display: "block", fontSize: 12, marginTop: 10, lineHeight: 1.55 }}>确认后会以当前曲线写入已复核净值，并自动计算最大回撤后进入研究。候选曲线只供比对；存在披露冲突时，必须先在原图上复核并明确确认。</Text>}
      <Paragraph type="secondary" style={{ marginTop: 16, marginBottom: 0 }}>
        当前共 <Text strong>{navCount}</Text> 个净值点。{!canAnalyze && <Text type="warning"> 至少需要 2 条记录才能计算。</Text>}<br />
        <Text type="secondary">提示：在净值框中按 Ctrl + Enter 可快速计算。</Text>
      </Paragraph>
    </>
  );
}
