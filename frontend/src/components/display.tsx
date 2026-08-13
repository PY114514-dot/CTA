/**
 * Presentational display components extracted from main.tsx.
 *
 * These are pure UI components with no business logic.
 */
import React from "react";
import { Alert, Col, Input, Statistic, Tag, Typography } from "antd";
import type {
  MultiProductReportResponse,
  PaddleOcrDocumentResponse,
  ProductStrategyProfileResponse,
  StrategyEvidence,
} from "../api";
import { percentage } from "../utils/format";

const { Paragraph: TypoParagraph } = Typography;

export function Metric({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <Col xs={12} md={8}>
      <Statistic title={label} value={value} />
    </Col>
  );
}

export function EvidenceGroup({ title, evidence }: { title: string; evidence: StrategyEvidence[] }): React.JSX.Element {
  return (
    <div style={{ marginTop: 14 }}>
      <TypoParagraph strong style={{ marginBottom: 6 }}>
        {title}
      </TypoParagraph>
      {evidence.length ? (
        evidence.map((item) => (
          <div key={item.label} style={{ marginBottom: 6 }}>
            <Tag color="blue">{item.label}</Tag>
            <span style={{ color: "#667085" }}>{item.evidence.join("；")}</span>
          </div>
        ))
      ) : (
        <TypoParagraph type="secondary" style={{ marginBottom: 0 }}>
          未发现足够关键词证据。
        </TypoParagraph>
      )}
    </div>
  );
}

export function StrategyProfileView({ profile }: { profile: ProductStrategyProfileResponse }): React.JSX.Element {
  return (
    <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid #f0f0f0" }}>
      <TypoParagraph strong style={{ marginBottom: 6 }}>
        策略假设：<Tag color="geekblue">{profile.strategy_hypothesis.label}</Tag>
      </TypoParagraph>
      <EvidenceGroup title="因子线索" evidence={profile.factor_hypotheses} />
      <EvidenceGroup title="期货品种线索" evidence={profile.futures_categories} />
      <TypoParagraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
        {profile.disclaimer}
      </TypoParagraph>
    </div>
  );
}

export function ReportExtractionView({ report }: { report: MultiProductReportResponse }): React.JSX.Element | null {
  if (!report.disclosed_metrics.length) return null;
  return (
    <div
      style={{
        marginTop: 16,
        padding: 12,
        background: "#f6ffed",
        border: "1px solid #b7eb8f",
        borderRadius: 6,
      }}
    >
      <TypoParagraph strong style={{ marginBottom: 8 }}>
        报告扫描
      </TypoParagraph>
      {report.disclosed_metrics.map((item, index) => (
        <div key={item.product_id} style={{ padding: "8px 0", borderTop: index ? "1px solid #d9f7be" : undefined }}>
          <Tag color="green">曲线 {index + 1}</Tag>
          {item.product_name && <Tag>{item.product_name}</Tag>}
          <Tag>{item.product_id}</Tag>
          {item.strategy && <Tag color="blue">{item.strategy}</Tag>}
          <span>
            披露累计 {percentage(item.cumulative_return)} · 年化 {percentage(item.annualized_return)} · 最大回撤{" "}
            {item.maximum_drawdown_disclosed === false ? "未披露" : percentage(item.maximum_drawdown)}
          </span>
          <div style={{ color: "#667085", fontSize: 12, marginTop: 3 }}>
            {item.start_date} 至 {item.end_date}；请在下方预览中按同序选择对应曲线校准。
          </div>
        </div>
      ))}
      {report.warnings.length > 0 && (
        <TypoParagraph type="secondary" style={{ margin: "8px 0 0" }}>
          {report.warnings.join(" ")}
        </TypoParagraph>
      )}
    </div>
  );
}

export function PaddleOcrDocumentView({ result }: { result: PaddleOcrDocumentResponse }): React.JSX.Element {
  const markdown = result.pages
    .map((page) => `<!-- 第 ${page.page_number} 页 -->\n${page.markdown}`)
    .join("\n\n");
  return (
    <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid #f0f0f0" }}>
      <TypoParagraph strong style={{ marginBottom: 4 }}>
        PaddleOCR 文档提取 <Tag color="blue">{result.model}</Tag>
      </TypoParagraph>
      <TypoParagraph type="secondary" style={{ fontSize: 12 }}>
        已提取 {result.pages.length} 个页面/版面。以下内容是 OCR 候选，须人工核验后方可作为产品披露或因子真值使用。
      </TypoParagraph>
      <Input.TextArea value={markdown} readOnly rows={10} aria-label="PaddleOCR Markdown 结果" />
      {result.warnings.length > 0 && (
        <Alert type="warning" showIcon message={result.warnings.join(" ")} style={{ marginTop: 8 }} />
      )}
    </div>
  );
}
