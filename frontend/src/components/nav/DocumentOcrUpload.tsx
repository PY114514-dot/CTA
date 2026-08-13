import { Alert, Button, Divider, Typography, Upload } from "antd";

const { Paragraph } = Typography;

/** Optional PDF/image OCR intake; extracted claims are rendered by the caller-supplied evidence view. */
export default function DocumentOcrUpload({ accept, loading, error, result, onUpload }: { accept: string; loading: boolean; error?: string; result?: React.ReactNode; onUpload: (file: File) => void }): React.JSX.Element {
  return <>
    <Divider titlePlacement="left" style={{ marginTop: 0 }}>报告 / PDF 识别（可选）</Divider>
    <Paragraph type="secondary" style={{ fontSize: 13 }}>上传周报或产品资料，提取的文字会自动写入下方 OCR 文本框，供净值与策略画像共同使用。</Paragraph>
    <Upload accept={accept} maxCount={1} showUploadList={false} beforeUpload={(file) => { onUpload(file); return false; }}><Button loading={loading}>上传并解析周报 / PDF（PaddleOCR）</Button></Upload>
    {error && <Alert type="error" showIcon message={error} style={{ marginTop: 12 }} closable />}
    {result}
  </>;
}
