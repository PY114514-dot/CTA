import React, { useMemo, useState } from "react";
import { Alert, Button, Card, Empty, Input, Select, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  recommendFof,
  type FofCandidateEvaluation,
  type FofFundInput,
  type FofRecommendationResponse,
  type FofRiskProfile,
} from "./api";

const { Paragraph, Text } = Typography;

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

export default function FofAgentDrawer(): React.JSX.Element {
  const [fundsText, setFundsText] = useState("");
  const [riskProfile, setRiskProfile] = useState<FofRiskProfile>("balanced");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [result, setResult] = useState<FofRecommendationResponse>();

  const evaluationColumns = useMemo<ColumnsType<FofCandidateEvaluation>>(() => [
    { title: "产品", dataIndex: "fund_name", key: "fund_name", width: 150 },
    { title: "评分", dataIndex: "score", key: "score", render: (value: number) => value.toFixed(2) },
    { title: "年化收益", key: "return", render: (_, item) => pct(item.metrics.annualized_return) },
    { title: "夏普", key: "sharpe", render: (_, item) => item.metrics.sharpe_ratio?.toFixed(2) ?? "—" },
    { title: "最大回撤", key: "drawdown", render: (_, item) => pct(item.metrics.maximum_drawdown) },
    { title: "资格", dataIndex: "eligible", key: "eligible", render: (value: boolean) => <Tag color={value ? "success" : "default"}>{value ? "可纳入" : "样本不足"}</Tag> },
  ], []);

  const allocationColumns = useMemo<ColumnsType<FofRecommendationResponse["recommendations"][number]>>(() => [
    { title: "推荐产品", dataIndex: "fund_name", key: "fund_name" },
    { title: "初始权重", dataIndex: "weight", key: "weight", width: 100, render: (value: number) => pct(value) },
    { title: "依据", dataIndex: "rationale", key: "rationale" },
  ], []);

  async function runAgent(): Promise<void> {
    setError(undefined);
    try {
      const parsed: unknown = JSON.parse(fundsText);
      if (!Array.isArray(parsed)) throw new Error("请输入由已复核产品组成的基金数组 JSON。");
      setLoading(true);
      setResult(await recommendFof({ funds: parsed as FofFundInput[], risk_profile: riskProfile, max_single_fund_weight: 0.35 }));
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "FOF Agent 执行失败。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Space direction="vertical" size={18} style={{ width: "100%" }}>
      <Alert type="info" showIcon message="FOF Agent Runtime" description="仅使用已确认、已复核的产品净值；未复核数据不会进入推荐。" />
      <Card title="候选基金与风险预算" className="card-accent-top">
        <label htmlFor="fof-risk-profile"><Text strong>风险偏好</Text></label>
        <Select id="fof-risk-profile" value={riskProfile} onChange={setRiskProfile} style={{ width: "100%", margin: "8px 0 14px" }} options={[
          { value: "conservative", label: "稳健：优先回撤控制" }, { value: "balanced", label: "均衡：收益与风险平衡" }, { value: "growth", label: "进取：更重视长期收益" },
        ]} />
        <label htmlFor="fof-funds-json"><Text strong>候选基金 JSON</Text></label>
        <Paragraph type="secondary" style={{ margin: "6px 0" }}>每只基金需包含 fund_id、fund_name、frequency 和按日期排序的 nav_points；至少两只已复核产品，每只至少 8 个净值点才可纳入。</Paragraph>
        <Input.TextArea id="fof-funds-json" aria-label="候选基金 JSON" value={fundsText} onChange={(event) => setFundsText(event.target.value)} rows={15} spellCheck={false} style={{ fontFamily: "var(--font-mono)", fontSize: 12 }} />
        <Button type="primary" loading={loading} onClick={() => void runAgent()} style={{ marginTop: 14 }}>运行 FOF Agent</Button>
      </Card>
      {error && <Alert type="error" showIcon message="FOF Agent 未能执行" description={error} closable onClose={() => setError(undefined)} />}
      {result ? <>
        {result.method_provenance && Object.keys(result.method_provenance).length > 0 && (
          <Alert
            type="info"
            showIcon
            message="本次结果的方法来源"
            description={
              <Space direction="vertical" size={2}>
                {Object.entries(result.method_provenance).map(([key, item]) => (
                  <Text key={key} style={{ fontSize: 11 }}>
                    {key}：{item.method ?? "未记录"}{item.detail ? ` · ${item.detail}` : ""}
                  </Text>
                ))}
              </Space>
            }
          />
        )}
        <Alert type={result.reflection.passed ? "success" : "warning"} showIcon message={result.reflection.passed ? "反思校验通过" : "反思校验要求人工复核"} description={result.reflection.warnings.join("；") || result.reflection.checks.join("；")} />
        <Card title="FOF 初始建议权重" className="card-accent-top">{result.recommendations.length ? <Table rowKey="fund_id" size="small" pagination={false} columns={allocationColumns} dataSource={result.recommendations} scroll={{ x: 620 }} /> : <Empty description="没有满足样本条件的产品" />}</Card>
        <Card title="候选评分与净值指标"><Table rowKey="fund_id" size="small" pagination={false} columns={evaluationColumns} dataSource={result.evaluations} scroll={{ x: 650 }} /></Card>
        <Card title="Agent 执行轨迹"><Space direction="vertical" size={6} style={{ width: "100%" }}>{result.agent_trace.map((event, index) => <Text key={`${event.iteration}-${index}`}><Tag>{event.phase}</Tag>{event.action}：{event.detail}</Text>)}<Text type="secondary">{result.disclaimer}</Text></Space></Card>
      </> : <Empty description="提交候选净值后，Agent 的工具调用与反思结果会显示在这里。" />}
    </Space>
  );
}
