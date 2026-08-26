import React, { useCallback, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  DatePicker,
  Descriptions,
  Empty,
  Progress,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Upload,
  Typography,
} from "antd";
import dayjs, { type Dayjs } from "dayjs";
import {
  AimOutlined,
  DashboardOutlined,
  LineChartOutlined,
  ThunderboltOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import {
  runFactorAttribution,
  parseWeeklyReport,
  validateFactorAttribution,
  type DataFrequency,
  type FactorAttributionResponse,
  type NavPoint,
  type FactorValidationResponse,
  type WeeklyReportSnapshot,
} from "./api";
import { ACCENT_FALLBACK, cssVar } from "./ui/tokens";

const { Text, Paragraph } = Typography;
const { RangePicker } = DatePicker;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface FactorAttributionPanelProps {
  navPoints: NavPoint[];
  frequency: DataFrequency;
}

// ---------------------------------------------------------------------------
// Chart helpers
// ---------------------------------------------------------------------------

function FactorContributionChart({ factors }: { factors: FactorAttributionResponse["factors"] }) {
  const data = factors.filter((f) => Math.abs(f.contribution_pct) > 0.1);
  if (data.length === 0) return null;

  const option = {
    title: { text: "因子收益贡献分解", left: "center", textStyle: { fontSize: 13 } },
    tooltip: { trigger: "axis" as const, formatter: "{b}: {c}%" },
    grid: { left: 100, right: 30, top: 40, bottom: 30 },
    xAxis: {
      type: "value" as const,
      axisLabel: { formatter: "{value}%" },
      splitLine: { lineStyle: { type: "dashed" as const } },
    },
    yAxis: {
      type: "category" as const,
      data: data.map((f) => f.display_name).reverse(),
      axisLabel: { fontSize: 12 },
    },
    series: [
      {
        type: "bar",
        data: data.map((f) => ({
          value: f.contribution_pct,
          itemStyle: {
            color: f.contribution_pct >= 0
              ? cssVar("--serif-accent", ACCENT_FALLBACK)
              : "#cf1322",
            borderRadius: f.contribution_pct >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4],
          },
        })).reverse(),
        barWidth: 22,
        label: { show: true, position: "right", formatter: "{c}%", fontSize: 11 },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 220 }} />;
}

function RollingR2Chart({ snapshots }: { snapshots: FactorAttributionResponse["rolling_r_squared"] }) {
  if (snapshots.length < 5) return null;

  // Downsample if too many points
  const step = Math.max(1, Math.floor(snapshots.length / 120));
  const sampled = snapshots.filter((_, i) => i % step === 0);

  const factorNames = Object.keys(sampled[0]?.betas ?? {});
  const accent = cssVar("--serif-accent", ACCENT_FALLBACK);

  const option = {
    title: { text: "滚动共同解释度与风格参考", left: "center", textStyle: { fontSize: 13 } },
    tooltip: { trigger: "axis" as const },
    legend: { bottom: 0, data: ["R²", ...factorNames], textStyle: { fontSize: 11 } },
    grid: { left: 50, right: 50, top: 40, bottom: 50 },
    xAxis: {
      type: "category" as const,
      data: sampled.map((s) => s.date),
      axisLabel: { fontSize: 10, rotate: 30 },
    },
    yAxis: [
      { type: "value" as const, name: "R²", min: 0, max: 1, position: "left" as const },
      { type: "value" as const, name: "关联系数", position: "right" as const },
    ],
    series: [
      {
        name: "R²",
        type: "line",
        data: sampled.map((s) => s.r_squared),
        yAxisIndex: 0,
        lineStyle: { width: 2.5, color: accent },
        itemStyle: { color: accent },
        symbol: "none",
        areaStyle: { opacity: 0.08 },
      },
      ...factorNames.map((name) => ({
        name,
        type: "line" as const,
        data: sampled.map((s) => s.betas[name] ?? 0),
        yAxisIndex: 1,
        smooth: true,
        symbol: "none",
        lineStyle: { width: 1.5 },
      })),
    ],
  };
  return <ReactECharts option={option} style={{ height: 300 }} />;
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function FactorAttributionPanel({ navPoints, frequency }: FactorAttributionPanelProps) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FactorAttributionResponse | null>(null);
  const [error, setError] = useState("");
  const [reportSnapshot, setReportSnapshot] = useState<WeeklyReportSnapshot | null>(null);
  const [validation, setValidation] = useState<FactorValidationResponse | null>(null);
  const [validationError, setValidationError] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [validationLoading, setValidationLoading] = useState(false);
  const [reportPeriod, setReportPeriod] = useState<[Dayjs, Dayjs]>([dayjs().subtract(7, "day"), dayjs()]);

  const handleRun = useCallback(async () => {
    if (navPoints.length < 20) {
      setError("定量因子归因至少需要 20 个净值数据点");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 60000);
      const rollingWindow = frequency === "daily" ? 60 : frequency === "weekly" ? 26 : 12;
      const oosTrainWindow = frequency === "daily" ? 60 : frequency === "weekly" ? 26 : 24;
      const res = await runFactorAttribution(navPoints, frequency, rollingWindow, undefined, controller.signal, 1000, oosTrainWindow);
      clearTimeout(timeout);
      setResult(res);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("请求超时（60秒），请确认因子库已构建");
      } else {
        setError(err instanceof Error ? err.message : "定量因子归因失败");
      }
    } finally {
      setLoading(false);
    }
  }, [navPoints, frequency]);

  const hasData = navPoints.length >= 20;

  const handleReportUpload = useCallback(async (file: File): Promise<void> => {
    setReportLoading(true);
    setValidationError("");
    setValidation(null);
    try {
      const snapshot = await parseWeeklyReport(file);
      setReportSnapshot(snapshot);
      const reportDay = dayjs(snapshot.report_date);
      if (reportDay.isValid()) setReportPeriod([reportDay.subtract(6, "day"), reportDay]);
    } catch (err) {
      setValidationError(err instanceof Error ? err.message : "周报解析失败");
    } finally {
      setReportLoading(false);
    }
  }, []);

  const handleValidate = useCallback(async (): Promise<void> => {
    if (!result || !reportSnapshot) return;
    setValidationLoading(true);
    setValidationError("");
    try {
      setValidation(await validateFactorAttribution({
        regressionResult: result,
        reportSnapshot,
        periodStart: reportPeriod[0].format("YYYY-MM-DD"),
        periodEnd: reportPeriod[1].format("YYYY-MM-DD"),
      }));
    } catch (err) {
      setValidationError(err instanceof Error ? err.message : "L1 与周报校验失败");
    } finally {
      setValidationLoading(false);
    }
  }, [result, reportSnapshot, reportPeriod]);

  return (
    <Card
      className="card-accent-top"
      title={
        <Space>
          <DashboardOutlined />
          <span>风格与归因参考（L1 统计画像）</span>
          <Tag color="blue">历史统计参考</Tag>
        </Space>
      }
      style={{ marginTop: 24 }}
    >
      {/* Trigger */}
      {!result && !loading && (
        <div style={{ textAlign: "center", padding: "24px 0" }}>
          <AimOutlined style={{ fontSize: 36, color: "var(--serif-accent)", marginBottom: 12 }} />
          <Paragraph type="secondary">
            将产品净值收益率对因子库做 HAC（Newey–West）多元回归，并用区块 Bootstrap 检查结果稳定性，
            输出历史共同波动、残差诊断和因子组贡献。它不能预测后续净值，也不代表真实持仓、交易品种或管理人能力。
          </Paragraph>
          <Button size="large" onClick={() => void handleRun()} disabled={!hasData} icon={<ThunderboltOutlined />}>
            查看风格参考
          </Button>
          {!hasData && (
            <Paragraph type="warning" style={{ marginTop: 8, fontSize: 12 }}>
              请先录入至少 20 个净值数据点，并确保因子库已构建（POST /api/factor-library/build）
            </Paragraph>
          )}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: "center", padding: "32px 0" }}>
          <Progress type="circle" percent={99.9} status="active" size={64} />
          <Paragraph type="secondary" style={{ marginTop: 16 }}>
            正在对齐产品净值与因子收益序列，计算统计关联、置信区间和分窗口变化…
          </Paragraph>
        </div>
      )}

      {/* Error */}
      {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} closable onClose={() => setError("")} />}

      {/* Results */}
      {result && (
        <div>
          {/* Warnings */}
          {result.warnings.length > 0 && (
            <Alert type="warning" showIcon message={result.warnings.join("；")} style={{ marginBottom: 16 }} closable />
          )}

          {/* Summary statistics */}
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col xs={12} sm={6}>
              <Statistic
                title="R²"
                value={result.r_squared}
                precision={4}
                suffix={<Text type="secondary" style={{ fontSize: 12 }}>Adj {result.adj_r_squared.toFixed(3)}</Text>}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title="未解释年化部分"
                value={result.annualized_alpha * 100}
                precision={2}
                suffix="%"
                valueStyle={{ color: result.annualized_alpha > 0 ? "#3f8600" : "#cf1322" }}
                prefix={result.alpha_t_stat > 2 ? "★" : undefined}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic title="残差年化波动" value={result.residual.annual_vol * 100} precision={2} suffix="%" />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic title="观测值" value={result.n_observations} suffix={<Text type="secondary" style={{ fontSize: 12 }}>{result.frequency}</Text>} />
            </Col>
          </Row>

          {/* Interpretation hint */}
          <Alert
            type={result.r_squared > 0.5 ? "success" : result.r_squared > 0.2 ? "info" : "warning"}
            showIcon
            style={{ marginBottom: 16 }}
            message={
              result.r_squared > 0.5
                ? `样本内因子与产品共同解释了 ${(result.r_squared * 100).toFixed(0)}% 的历史收益波动；这只是一项风格参考，不能外推为预测能力。`
                : result.r_squared > 0.2
                ? `样本内共同解释度为 ${(result.r_squared * 100).toFixed(0)}%，大量历史波动仍未被当前因子覆盖，结果仅供辅助研判。`
                : `共同解释度仅 ${(result.r_squared * 100).toFixed(0)}%，产品策略或数据特征可能超出当前因子范围；不应据此判断未来收益。`
            }
          />

          <Descriptions size="small" column={{ xs: 1, sm: 3 }} bordered style={{ marginBottom: 16 }}>
            <Descriptions.Item label="样本外 R²">
              {result.out_of_sample.r_squared === null ? "—" : result.out_of_sample.r_squared.toFixed(3)}
            </Descriptions.Item>
            <Descriptions.Item label="样本外相关">
              {result.out_of_sample.correlation === null ? "—" : result.out_of_sample.correlation.toFixed(3)}
            </Descriptions.Item>
            <Descriptions.Item label="样本外跟踪误差">
              {result.out_of_sample.tracking_error_annual === null ? "—" : `${(result.out_of_sample.tracking_error_annual * 100).toFixed(2)}%`}
            </Descriptions.Item>
            <Descriptions.Item label="整体显著性 p 值">
              {result.joint_hac.p_value === null ? "—" : result.joint_hac.p_value < 0.001 ? "<0.001" : result.joint_hac.p_value.toFixed(4)}
            </Descriptions.Item>
            <Descriptions.Item label="因子条件数">
              {result.collinearity.condition_number === null ? "—" : result.collinearity.condition_number.toFixed(1)}
            </Descriptions.Item>
            <Descriptions.Item label="样本外训练窗口">
              {result.out_of_sample.train_window} 期
            </Descriptions.Item>
          </Descriptions>
          {result.collinearity.high_correlation_pairs.length > 0 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message="因子共线性提示"
              description={`高相关因子对：${result.collinearity.high_correlation_pairs.join("；")}。单个关联系数不宜过度解读，应结合因子组贡献。`}
            />
          )}

          <Alert
            type={result.diagnostics.residual_autocorrelation_detected || result.diagnostics.non_normality_detected ? "warning" : "info"}
            showIcon
            style={{ marginBottom: 16 }}
              message={`统计检验：${result.inference_method}；最大滞后期数 ${result.hac_max_lags ?? "—"}`}
            description={
              <Space direction="vertical" size={2}>
                <span>
                  Bootstrap：{result.bootstrap.successful_reps}/{result.bootstrap.reps} 次成功，区块长度 {result.bootstrap.block_length ?? "—"}；区间只用于衡量参数不确定性。
                </span>
                <span>
                  残差诊断：{result.diagnostics.residual_autocorrelation_detected ? "存在自相关" : "未发现显著自相关"}；
                  {result.diagnostics.volatility_clustering_detected ? "存在波动聚集" : "未发现明显波动聚集"}；
                  {result.diagnostics.non_normality_detected ? "未通过正态性检验" : "未发现明显非正态证据"}。
                </span>
              </Space>
            }
          />

          {/* Factor beta table + contribution chart */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col xs={24} md={12}>
              <Table
                dataSource={result.factors}
                rowKey="name"
                size="small"
                pagination={false}
                columns={[
                  {
                    title: "因子",
                    dataIndex: "display_name",
                    width: 120,
                    render: (v: string, record) => (
                      <Space size={4}>
                        {v}
                        {record.significant && <Tag color="blue" style={{ fontSize: 10, lineHeight: "16px", padding: "0 4px" }}>显著</Tag>}
                      </Space>
                    ),
                  },
                  { title: "组别", dataIndex: "factor_group", width: 92 },
                  { title: "关联系数", dataIndex: "beta", width: 86, render: (v: number) => v.toFixed(4) },
                  { title: "统计误差", dataIndex: "std_error", width: 78, render: (v: number) => v.toFixed(4) },
                  { title: "t 值", dataIndex: "t_stat", width: 60, render: (v: number) => (
                    <Text strong={Math.abs(v) > 2} type={Math.abs(v) > 2 ? undefined : "secondary"}>{v.toFixed(2)}</Text>
                  )},
                  { title: "p 值", dataIndex: "p_value", width: 60, render: (v: number) => v < 0.001 ? "<0.001" : v.toFixed(3) },
                  { title: "收益贡献%", dataIndex: "contribution_pct", width: 82, render: (v: number) => (
                    <Text style={{ color: v >= 0 ? "#3f8600" : "#cf1322" }}>{v.toFixed(1)}%</Text>
                  )},
                  {
                    title: "统计风险贡献",
                    width: 104,
                    render: (_: unknown, record) => {
                      const value = result.factor_risk_contributions[record.name];
                      return value === null || value === undefined ? "—" : value.toFixed(4);
                    },
                  },
                  {
                    title: "Bootstrap 95% CI",
                    width: 130,
                    render: (_: unknown, record) => record.bootstrap_ci_low === null || record.bootstrap_ci_high === null
                      ? "—"
                      : `[${record.bootstrap_ci_low.toFixed(3)}, ${record.bootstrap_ci_high.toFixed(3)}]`,
                  },
                ]}
              />
              <Alert
                type="info"
                showIcon
                style={{ marginTop: 10 }}
                message="三类输出不可互相替代"
                description="关联系数衡量产品收益与因子收益的统计关系；收益贡献是关联系数 × 因子平均收益相对产品平均收益的解释比例；统计风险贡献来自因子协方差。三者均不代表真实持仓或实际盈亏。"
              />
              {result.lasso_selected.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>LASSO 选择：</Text>
                  {result.lasso_selected.map((name) => {
                    const f = result.factors.find((x) => x.name === name);
                    return <Tag key={name} style={{ fontSize: 11 }}>{f?.display_name ?? name}</Tag>;
                  })}
                </div>
              )}
              {Object.keys(result.factor_group_contributions).length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>经济因子组贡献：</Text>
                  <Space wrap size={[4, 4]} style={{ marginTop: 4 }}>
                    {Object.entries(result.factor_group_contributions).map(([group, value]) => (
                      <Tag key={group} color={value >= 0 ? "green" : "red"}>
                        {group} {value.toFixed(1)}%
                      </Tag>
                    ))}
                  </Space>
                </div>
              )}
            </Col>
            <Col xs={24} md={12}>
              <FactorContributionChart factors={result.factors} />
            </Col>
          </Row>

          {/* Rolling R² and exposures */}
          {result.rolling_r_squared.length > 5 && (
            <Card size="small" title={<Space><LineChartOutlined />滚动回归</Space>} style={{ marginBottom: 16 }}>
              <RollingR2Chart snapshots={result.rolling_r_squared} />
            </Card>
          )}

          {/* Residual diagnostics */}
          <Descriptions size="small" column={{ xs: 2, sm: 4 }} bordered style={{ marginBottom: 16 }}>
            <Descriptions.Item label="未解释收益显著性 p 值">{result.alpha_p_value < 0.001 ? "<0.001" : result.alpha_p_value.toFixed(4)}</Descriptions.Item>
            <Descriptions.Item label="未解释收益 95% 区间">
              {result.annualized_alpha_bootstrap_ci_low === null || result.annualized_alpha_bootstrap_ci_high === null
                ? "—"
                : `[${(result.annualized_alpha_bootstrap_ci_low * 100).toFixed(2)}%, ${(result.annualized_alpha_bootstrap_ci_high * 100).toFixed(2)}%]`}
            </Descriptions.Item>
            <Descriptions.Item label="整体检验统计量">{result.joint_hac.statistic === null ? "—" : result.joint_hac.statistic.toFixed(2)}</Descriptions.Item>
            <Descriptions.Item label="整体显著性 p 值">{result.joint_hac.p_value === null ? "—" : result.joint_hac.p_value < 0.001 ? "<0.001" : result.joint_hac.p_value.toFixed(4)}</Descriptions.Item>
            <Descriptions.Item label="残差偏度">{result.residual.skewness.toFixed(3)}</Descriptions.Item>
            <Descriptions.Item label="残差峰度">{result.residual.kurtosis.toFixed(3)}</Descriptions.Item>
          </Descriptions>

          <Card size="small" title="L1 公开因子 vs L4 周报披露校验" style={{ marginBottom: 16 }}>
            <Paragraph type="secondary" style={{ fontSize: 12 }}>
              上传管理人周报后，系统以“回归关联系数 × 同期公开因子收益”与周报因子贡献比较；不会将关联系数直接和单期盈亏相比。
            </Paragraph>
            <Space wrap>
              <Upload accept="application/pdf,image/png,image/jpeg" showUploadList={false} beforeUpload={(file) => { void handleReportUpload(file); return false; }}>
                <Button icon={<UploadOutlined />} loading={reportLoading}>解析周报</Button>
              </Upload>
              <RangePicker value={reportPeriod} allowClear={false} onChange={(value) => {
                if (value?.[0] && value?.[1]) setReportPeriod([value[0], value[1]]);
              }} />
              <Button type="primary" onClick={() => void handleValidate()} disabled={!reportSnapshot} loading={validationLoading}>校验同报告期</Button>
            </Space>
            {reportSnapshot && <Alert type="info" showIcon style={{ marginTop: 10 }} message={`已解析：${reportSnapshot.product_name || "未识别产品"} · 报告日 ${reportSnapshot.report_date} · 解析置信度 ${(reportSnapshot.parse_confidence * 100).toFixed(0)}%`} />}
            {validationError && <Alert type="error" showIcon style={{ marginTop: 10 }} message={validationError} closable onClose={() => setValidationError("")} />}
            {validation && (
              <div style={{ marginTop: 12 }}>
                <Alert type={validation.direction_agreement_pct >= 60 ? "success" : "warning"} showIcon message={validation.summary} description={validation.r_gap_interpretation} style={{ marginBottom: 10 }} />
                {validation.warnings.length > 0 && <Alert type="warning" showIcon message={validation.warnings.join("；")} style={{ marginBottom: 10 }} />}
                <Table
                  size="small"
                  rowKey="factor_name"
                  pagination={false}
                  dataSource={validation.factor_comparisons}
                  columns={[
                    { title: "因子", dataIndex: "display_name" },
                    { title: "关联系数", dataIndex: "l1_beta", render: (value: number) => value.toFixed(3) },
                    { title: "公开因子同期收益", dataIndex: "l1_factor_period_return", render: (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(2)}%` },
                    { title: "L1 预测贡献", dataIndex: "l1_predicted_contribution", render: (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(2)}%` },
                    { title: "周报披露贡献", dataIndex: "l4_contribution", render: (value: number) => `${(value * 100).toFixed(2)}%` },
                    { title: "方向", dataIndex: "direction_match", render: (value: boolean | null) => value === null ? <Tag>不可比</Tag> : value ? <Tag color="green">一致</Tag> : <Tag color="red">不一致</Tag> },
                  ]}
                />
              </div>
            )}
          </Card>

          {/* Footer */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              样本区间 {result.start_date} ~ {result.end_date} · {result.inference_method} · 基于未风控基准因子回归；结果仅解释历史共同波动，非持仓披露或净值预测
            </Text>
            <Button size="small" onClick={() => void handleRun()} loading={loading}>
              重新归因
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}
