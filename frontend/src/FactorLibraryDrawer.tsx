import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  DatePicker,
  Drawer,
  Modal,
  Progress,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import type { TableProps } from "antd";
import dayjs, { type Dayjs } from "dayjs";
import { KatexBlock } from "./components/KatexMath";
import "katex/dist/katex.min.css";
import {
  exportFactorLibrary,
  getBuildJob,
  getFactorDetail,
  getFactorLibraryStatus,
  getFactorPerformance,
  getSectorBreakdown,
  listFactors,
  startFactorBuild,
  type BuildJobStatus,
  type FactorDetail,
  type FactorMetaItem,
  type FactorPerformanceItem,
  type FactorSectorBreakdown,
  type RiskOverlayMeta,
  type SectorBreakdownResponse,
} from "./api";
import { FONT_DISPLAY } from "./theme";
import ExternalFactorLibraryPanel from "./ExternalFactorLibraryPanel";
import { SectionTitle } from "./ui/Section";
import { pct, SignedValue } from "./ui/SignedValue";

const { Paragraph, Text } = Typography;
const { RangePicker } = DatePicker;

// ---------------------------------------------------------------------------
// Presentational helpers (aligned with SettingsDrawer style)
// ---------------------------------------------------------------------------

/** Labeled section block used inside the factor detail modal. */
function DetailSection({ title, children }: { title: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <div style={{ marginBottom: 18 }}>
      <div
        style={{
          fontFamily: FONT_DISPLAY,
          fontSize: 13,
          fontWeight: 600,
          color: "var(--serif-foreground)",
          marginBottom: 8,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FactorLibraryStatus {
  cached_factors: Record<string, { built_at: string; rows: number; start: string; end: string }>;
  providers: { akshare_available: boolean; csv_symbols: string[] };
  core_varieties: string[];
}

interface FactorRow {
  key: string;
  name: string;
  display_name: string;
  category: string;
  description: string;
  rows: number | null;
  start: string | null;
  end: string | null;
  built_at: string | null;
  cached: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export interface FactorLibraryDrawerProps {
  open: boolean;
  onClose: () => void;
}

export default function FactorLibraryDrawer({ open, onClose }: FactorLibraryDrawerProps): React.JSX.Element {
  const [loading, setLoading] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);
  const [factors, setFactors] = useState<FactorMetaItem[]>([]);
  const [status, setStatus] = useState<FactorLibraryStatus | null>(null);
  const [performance, setPerformance] = useState<FactorPerformanceItem[]>([]);
  const [riskProfile, setRiskProfile] = useState<RiskOverlayMeta["profile"]>("baseline");
  const [riskOverlay, setRiskOverlay] = useState<RiskOverlayMeta | null>(null);
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs]>([dayjs("2020-01-01"), dayjs()]);
  const [useCache, setUseCache] = useState(true);

  const [job, setJob] = useState<BuildJobStatus | null>(null);
  const [buildError, setBuildError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const logBoxRef = useRef<HTMLDivElement | null>(null);

  // Factor "公式与代码" detail dialog
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detail, setDetail] = useState<FactorDetail | null>(null);

  // Sector breakdown (板块盈亏归因)
  const [sectorRange, setSectorRange] = useState<[Dayjs, Dayjs]>([dayjs().subtract(7, "day"), dayjs()]);
  const [sectorLoading, setSectorLoading] = useState(false);
  const [sectorError, setSectorError] = useState<string | null>(null);
  const [sectorData, setSectorData] = useState<SectorBreakdownResponse | null>(null);

  const jobRunning = job?.status === "running";

  // ----- data loading -------------------------------------------------------

  const refreshData = useCallback(async (): Promise<void> => {
    setDataError(null);
    try {
      const [f, s, p] = await Promise.all([listFactors(), getFactorLibraryStatus(), getFactorPerformance(riskProfile)]);
      setFactors(f.factors);
      setStatus(s);
      setPerformance(p.performance);
      setRiskOverlay(p.risk_overlay);
    } catch (err) {
      setDataError(err instanceof Error ? err.message : String(err));
    }
  }, [riskProfile]);

  useEffect(() => {
    if (open) {
      setLoading(true);
      void refreshData().finally(() => setLoading(false));
    }
  }, [open, refreshData]);

  // ----- factor detail dialog -------------------------------------------------

  const openDetail = useCallback(async (name: string): Promise<void> => {
    setDetailOpen(true);
    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);
    try {
      setDetail(await getFactorDetail(name));
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : String(err));
    } finally {
      setDetailLoading(false);
    }
  }, []);

  // ----- sector breakdown -------------------------------------------------------

  const handleSectorAnalysis = useCallback(async (): Promise<void> => {
    setSectorLoading(true);
    setSectorError(null);
    try {
      const result = await getSectorBreakdown(
        sectorRange[0].format("YYYY-MM-DD"),
        sectorRange[1].format("YYYY-MM-DD"),
      );
      setSectorData(result);
    } catch (err) {
      setSectorError(err instanceof Error ? err.message : String(err));
    } finally {
      setSectorLoading(false);
    }
  }, [sectorRange]);

  // ----- export (CSV / Markdown) ------------------------------------------------

  const [exporting, setExporting] = useState(false);

  const handleExport = useCallback(async (format: "csv" | "markdown"): Promise<void> => {
    setExporting(true);
    try {
      const { filename, content, content_type } = await exportFactorLibrary(format);
      const blob = new Blob([content], {
        type: content_type ?? (format === "csv" ? "text/csv;charset=utf-8" : "text/markdown;charset=utf-8"),
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // silent — export is non-critical
    } finally {
      setExporting(false);
    }
  }, []);

  // ----- build job polling ----------------------------------------------------

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const pollJob = useCallback(
    async (jobId: string) => {
      try {
        const j = await getBuildJob(jobId);
        setJob(j);
        if (j.status !== "running") {
          stopPolling();
          void refreshData();
        }
      } catch (err) {
        stopPolling();
        setBuildError(err instanceof Error ? err.message : String(err));
      }
    },
    [refreshData, stopPolling],
  );

  const handleBuild = async (): Promise<void> => {
    setBuildError(null);
    setJob(null);
    try {
      const { job_id } = await startFactorBuild({
        start: dateRange[0].format("YYYY-MM-DD"),
        end: dateRange[1].format("YYYY-MM-DD"),
        useCache,
      });
      const initial = await getBuildJob(job_id);
      setJob(initial);
      stopPolling();
      pollRef.current = window.setInterval(() => void pollJob(job_id), 1500);
    } catch (err) {
      setBuildError(err instanceof Error ? err.message : String(err));
    }
  };

  // Auto-scroll log box to bottom
  useEffect(() => {
    const el = logBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [job?.logs.length]);

  // ----- derived data ---------------------------------------------------------

  const factorRows: FactorRow[] = useMemo(
    () =>
      factors.map((f) => {
        const info = status?.cached_factors[f.name];
        return {
          key: f.name,
          name: f.name,
          display_name: f.display_name,
          category: f.category,
          description: f.description,
          rows: info?.rows ?? null,
          start: info?.start ?? null,
          end: info?.end ?? null,
          built_at: info?.built_at ?? null,
          cached: Boolean(info),
        };
      }),
    [factors, status],
  );

  const buildSummary = useMemo(() => {
    if (!job || job.status === "running" || !job.result) return null;
    const entries = Object.values(job.result.results);
    const built = entries.filter((r) => r.status === "built").length;
    const cached = entries.filter((r) => r.status === "cached").length;
    const skippedEntries = entries.filter((r) => r.status === "skipped");
    const skipped = skippedEntries.length;
    const failed = Object.keys(job.result.errors).length;
    return { built, cached, skipped, failed, skippedNotes: skippedEntries.map((r) => r.note).filter(Boolean) as string[] };
  }, [job]);

  // ----- table columns ----------------------------------------------------------

  const factorColumns: TableProps<FactorRow>["columns"] = [
    {
      title: "因子",
      dataIndex: "display_name",
      key: "display_name",
      render: (_: unknown, row: FactorRow) => (
        <Tooltip title={`${row.description} · 点击查看公式与代码`}>
          <div
            onClick={() => void openDetail(row.name)}
            style={{ cursor: "pointer" }}
          >
            <div
              style={{
                fontWeight: 600,
                color: "var(--serif-foreground)",
                textDecoration: "underline",
                textDecorationColor: "var(--serif-border)",
                textUnderlineOffset: 3,
              }}
            >
              {row.display_name}
            </div>
            <div style={{ fontSize: 11, color: "var(--serif-muted-foreground)", fontFamily: "var(--font-mono)" }}>
              {row.name}
            </div>
          </div>
        </Tooltip>
      ),
    },
    {
      title: "类别",
      dataIndex: "category",
      key: "category",
      width: 72,
      render: (v: string) => <Tag>{v}</Tag>,
    },
    {
      title: "行数",
      dataIndex: "rows",
      key: "rows",
      width: 64,
      align: "right",
      render: (v: number | null) => (v === null ? <Text type="secondary">—</Text> : v),
    },
    {
      title: "覆盖区间",
      key: "range",
      width: 170,
      render: (_: unknown, row: FactorRow) =>
        row.start && row.end ? (
          <span style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
            {row.start} ~ {row.end}
          </span>
        ) : (
          <Text type="secondary">—</Text>
        ),
    },
    {
      title: "构建时间",
      dataIndex: "built_at",
      key: "built_at",
      width: 130,
      render: (v: string | null) =>
        v ? (
          <span style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>{v.replace("T", " ")}</span>
        ) : (
          <Text type="secondary">—</Text>
        ),
    },
    {
      title: "状态",
      key: "state",
      width: 72,
      render: (_: unknown, row: FactorRow) =>
        row.cached ? <Tag color="green">已缓存</Tag> : <Tag color="default">未构建</Tag>,
    },
  ];

  const perfColumns: TableProps<FactorPerformanceItem>["columns"] = [
    {
      title: "因子",
      dataIndex: "display_name",
      key: "display_name",
      render: (v: string) => <span style={{ fontWeight: 600 }}>{v}</span>,
    },
    {
      title: "年化收益",
      dataIndex: "annualized_return",
      key: "annualized_return",
      align: "right",
      render: (v: number) => <SignedValue value={v} format={pct} />,
    },
    {
      title: "年化波动",
      dataIndex: "annualized_vol",
      key: "annualized_vol",
      align: "right",
      render: (v: number) => <span style={{ fontVariantNumeric: "tabular-nums" }}>{pct(v)}</span>,
    },
    {
      title: "夏普",
      dataIndex: "sharpe",
      key: "sharpe",
      align: "right",
      render: (v: number) => <SignedValue value={v} format={(x) => x.toFixed(2)} />,
    },
    {
      title: "最大回撤",
      dataIndex: "max_drawdown",
      key: "max_drawdown",
      align: "right",
      render: (v: number) => <SignedValue value={v} format={pct} />,
    },
    {
      title: "胜率",
      dataIndex: "win_rate",
      key: "win_rate",
      align: "right",
      render: (v: number) => <span style={{ fontVariantNumeric: "tabular-nums" }}>{pct(v)}</span>,
    },
    {
      title: "盈亏比",
      dataIndex: "payoff_ratio",
      key: "payoff_ratio",
      align: "right",
      render: (v: number) => <span style={{ fontVariantNumeric: "tabular-nums" }}>{v.toFixed(2)}</span>,
    },
    {
      title: "最差月",
      dataIndex: "worst_month",
      key: "worst_month",
      align: "right",
      render: (v: number) => <SignedValue value={v} format={pct} />,
    },
  ];

  // ----- render -----------------------------------------------------------------

  return (
    <Drawer title="因子库管理" placement="right" width={680} open={open} onClose={onClose}>
      {/* Data source status -------------------------------------------------- */}
      <SectionTitle first>数据源状态</SectionTitle>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
          padding: "10px 12px",
          border: "1px solid var(--serif-border)",
          borderRadius: 6,
          background: "var(--serif-card)",
        }}
      >
        {status ? (
          <>
            {status.providers.akshare_available ? (
              <Tag color="green">AKShare 在线</Tag>
            ) : (
              <Tag color="red">AKShare 不可用</Tag>
            )}
            <Tooltip title={status.providers.csv_symbols.join(", ") || "无上传数据"}>
              <Tag color={status.providers.csv_symbols.length > 0 ? "blue" : "default"}>
                CSV 行情 {status.providers.csv_symbols.length} 个品种
              </Tag>
            </Tooltip>
            <Tag>核心品种 {status.core_varieties.length} 个</Tag>
          </>
        ) : (
          <Tag>{loading ? "探测中…" : "未连接"}</Tag>
        )}
        <Button size="small" style={{ marginLeft: "auto" }} onClick={() => void refreshData()} loading={loading}>
          刷新
        </Button>
      </div>
      {dataError && (
        <Alert type="error" showIcon closable message={dataError} style={{ marginTop: 10 }} onClose={() => setDataError(null)} />
      )}
      <Paragraph type="secondary" style={{ fontSize: 12, margin: "10px 0 0" }}>
        因子仅用于产品风格参考、归因解释和异常提示，不构成收益预测或交易建议。
      </Paragraph>

      {/* One-click build -------------------------------------------------------- */}
      <SectionTitle>一键构建</SectionTitle>
      <div
        style={{
          padding: "12px",
          border: "1px solid var(--serif-border)",
          borderRadius: 6,
          background: "var(--serif-card)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <RangePicker
            value={dateRange}
            allowClear={false}
            disabled={jobRunning}
            onChange={(vals) => {
              if (vals && vals[0] && vals[1]) setDateRange([vals[0], vals[1]]);
            }}
          />
          <Checkbox checked={useCache} disabled={jobRunning} onChange={(e) => setUseCache(e.target.checked)}>
            跳过已缓存
          </Checkbox>
          <Button onClick={() => void handleBuild()} loading={jobRunning} disabled={loading}>
            {jobRunning ? "构建中…" : "开始构建"}
          </Button>
        </div>
        <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
          构建全部 10 个因子（趋势、量价相关性、截面动量、基差 Carry、盘面利润、短期动量、偏度、短期反转、仓单、库存），视数据源情况约需 1–5 分钟。
        </Paragraph>

        {job && (
          <div style={{ marginTop: 12 }}>
            <Progress
              percent={Math.round(job.progress)}
              size="small"
              status={job.status === "error" ? "exception" : job.status === "done" ? "success" : "active"}
            />
          </div>
        )}

        {buildError && (
          <Alert type="error" showIcon closable message={buildError} style={{ marginTop: 10 }} onClose={() => setBuildError(null)} />
        )}

        {buildSummary && job && (
          <Alert
            type={buildSummary.failed > 0 ? "warning" : "success"}
            showIcon
            style={{ marginTop: 10 }}
            message={`构建完成：新建 ${buildSummary.built} 个，缓存命中 ${buildSummary.cached} 个${
              buildSummary.skipped > 0 ? `，跳过 ${buildSummary.skipped} 个` : ""
            }${buildSummary.failed > 0 ? `，失败 ${buildSummary.failed} 个` : ""}`}
            description={
              job.result &&
              (Object.keys(job.result.errors).length > 0 || buildSummary.skippedNotes.length > 0) ? (
                <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
                  {Object.entries(job.result.errors).map(([name, msg]) => (
                    <div key={name} style={{ color: "#cf1322" }}>
                      {name}: {msg}
                    </div>
                  ))}
                  {buildSummary.skippedNotes.map((note, idx) => (
                    <div key={`skip-${idx}`} style={{ color: "var(--serif-muted-foreground)" }}>
                      跳过：{note}
                    </div>
                  ))}
                </div>
              ) : undefined
            }
          />
        )}
      </div>

      {/* Build logs -------------------------------------------------------------- */}
      {job && job.logs.length > 0 && (
        <>
          <SectionTitle>构建日志</SectionTitle>
          <div
            ref={logBoxRef}
            style={{
              maxHeight: 200,
              overflowY: "auto",
              padding: "10px 12px",
              border: "1px solid var(--serif-border)",
              borderRadius: 6,
              background: "var(--serif-muted)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              lineHeight: 1.9,
            }}
          >
            {job.logs.map((entry, idx) => {
              const isError = /失败|不足|异常|终止/.test(entry.message);
              return (
                <div key={idx} style={{ color: isError ? "#cf1322" : "var(--serif-foreground)" }}>
                  <span style={{ color: "var(--serif-muted-foreground)", marginRight: 8 }}>{entry.time}</span>
                  {entry.message}
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Factor list ---------------------------------------------------------------- */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <SectionTitle>因子列表</SectionTitle>
        <Space size={6}>
          <Button size="small" loading={exporting} onClick={() => void handleExport("csv")}>
            导出 CSV
          </Button>
          <Button size="small" loading={exporting} onClick={() => void handleExport("markdown")}>
            导出 Markdown
          </Button>
        </Space>
      </div>
      <Table<FactorRow>
        columns={factorColumns}
        dataSource={factorRows}
        size="small"
        pagination={false}
        loading={loading}
        locale={{ emptyText: "暂无已注册因子" }}
      />

      {/* Factor performance ------------------------------------------------------------ */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <SectionTitle>因子表现概览</SectionTitle>
        <Select
          value={riskProfile}
          style={{ width: 210 }}
          onChange={(value: RiskOverlayMeta["profile"]) => setRiskProfile(value)}
          options={[
            { value: "baseline", label: "基准因子（归因口径）" },
            { value: "vol_target", label: "波动率目标（10%）" },
            { value: "drawdown_control", label: "波动率目标 + 回撤降仓" },
          ]}
        />
      </div>
      {riskOverlay && (
        <Alert
          type={riskProfile === "baseline" ? "info" : "warning"}
          showIcon
          style={{ marginBottom: 12 }}
          message={riskOverlay.display_name}
          description={`${riskOverlay.description} 产品净值归因始终使用“基准因子”，不会使用此处的风控覆盖收益。`}
        />
      )}
      {performance.length > 0 ? (
        <Table<FactorPerformanceItem>
          columns={perfColumns}
          dataSource={performance.map((p) => ({ ...p, key: p.name }))}
          size="small"
          pagination={false}
          loading={loading}
        />
      ) : (
        <Paragraph type="secondary" style={{ fontSize: 12 }}>
          尚无已构建的因子缓存，请先执行构建。
        </Paragraph>
      )}

      {/* Sector breakdown (板块盈亏归因) ------------------------------------------------ */}
      <SectionTitle>板块盈亏归因</SectionTitle>
      <div
        style={{
          padding: "12px",
          border: "1px solid var(--serif-border)",
          borderRadius: 6,
          background: "var(--serif-card)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <RangePicker
            value={sectorRange}
            allowClear={false}
            onChange={(vals) => {
              if (vals && vals[0] && vals[1]) setSectorRange([vals[0], vals[1]]);
            }}
          />
          <Button onClick={() => void handleSectorAnalysis()} loading={sectorLoading}>
            分析板块盈亏
          </Button>
        </div>
        <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
          基于因子贡献矩阵，按板块（黑色 / 有色 / 贵金属 / 能化 / 农产品）聚合区间盈亏。需先构建因子。
        </Paragraph>

        {sectorError && (
          <Alert type="error" showIcon closable message={sectorError} style={{ marginTop: 10 }} onClose={() => setSectorError(null)} />
        )}

        {sectorData && sectorData.factors.length === 0 && !sectorError && (
          <Alert type="info" showIcon message="该区间内无可用贡献数据，请先构建因子或调整日期范围。" style={{ marginTop: 10 }} />
        )}

        {sectorData && sectorData.factors.length > 0 && (
          <div style={{ marginTop: 14 }}>
            {sectorData.factors.map((f: FactorSectorBreakdown) => (
              <div
                key={f.factor_name}
                style={{
                  marginBottom: 14,
                  padding: "10px 12px",
                  border: "1px solid var(--serif-border)",
                  borderRadius: 6,
                  background: "var(--serif-background)",
                }}
              >
                <div style={{ marginBottom: 6 }}>
                  <Text strong style={{ color: "var(--serif-foreground)" }}>{f.display_name}</Text>
                  <Text
                    style={{
                      marginLeft: 10,
                      fontVariantNumeric: "tabular-nums",
                      color: f.period_return >= 0 ? "#389e0d" : "#cf1322",
                      fontWeight: 600,
                    }}
                  >
                    {f.period_return >= 0 ? "+" : ""}{f.period_return.toFixed(2)}%
                  </Text>
                </div>
                <Paragraph style={{ fontSize: 13, marginBottom: 8, color: "var(--serif-foreground)" }}>
                  {f.prose}
                </Paragraph>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {f.sectors.map((s) => (
                    <Tag
                      key={s.sector}
                      color={s.direction === "盈利" ? "green" : s.direction === "亏损" ? "red" : "default"}
                      style={{ fontVariantNumeric: "tabular-nums" }}
                    >
                      {s.display_sector} {s.contribution >= 0 ? "+" : ""}{s.contribution.toFixed(2)}%
                    </Tag>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <SectionTitle>外部 CTA 因子库</SectionTitle>
      <ExternalFactorLibraryPanel />

      <div
        style={{
          marginTop: 26,
          paddingTop: 12,
          borderTop: "1px solid var(--serif-border)",
          fontSize: 12,
          color: "var(--serif-muted-foreground)",
        }}
      >
        因子收益序列缓存于 <Text code>data/factors/</Text>，构建完成后可在「定量因子归因」中直接使用。
      </div>

      {/* Factor detail dialog (公式与代码) ----------------------------------------- */}
      <Modal
        title={
          detail ? (
            <Space size={8}>
              <span>{detail.display_name}</span>
              <Tag>{detail.category}</Tag>
            </Space>
          ) : (
            "因子详情"
          )
        }
        open={detailOpen}
        onCancel={() => setDetailOpen(false)}
        footer={null}
        width={780}
        destroyOnClose
      >
        {detailLoading ? (
          <div style={{ textAlign: "center", padding: "56px 0" }}>
            <Spin />
          </div>
        ) : detailError ? (
          <Alert type="error" showIcon message={detailError} />
        ) : detail ? (
          <div>
            <DetailSection title="交易信号（何时产生信号）">
              <Paragraph style={{ marginBottom: 0, fontSize: 13, lineHeight: 1.9 }}>{detail.signal_rule}</Paragraph>
            </DetailSection>

            <DetailSection title="数学公式">
              <div
                style={{
                  padding: "16px 12px",
                  border: "1px solid var(--serif-border)",
                  borderRadius: 6,
                  background: "var(--serif-card)",
                  overflowX: "auto",
                }}
              >
                <KatexBlock
                  math={detail.formula}
                  renderError={(error) => (
                    <pre style={{ margin: 0, color: "#cf1322", fontSize: 12 }}>{error.message}</pre>
                  )}
                />
              </div>
            </DetailSection>

            <DetailSection title="默认参数">
              <Space size={[6, 6]} wrap>
                {Object.entries(detail.params).map(([key, value]) => (
                  <Tag key={key} style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                    {key} = {String(value)}
                  </Tag>
                ))}
              </Space>
            </DetailSection>

            <DetailSection title="经济逻辑与推导">
              <Paragraph style={{ marginBottom: 0, fontSize: 13, lineHeight: 1.9 }}>{detail.derivation}</Paragraph>
            </DetailSection>

            <DetailSection title="代码实现（compute 方法，运行时自动抽取）">
              <pre
                style={{
                  margin: 0,
                  maxHeight: 340,
                  overflow: "auto",
                  padding: "12px 14px",
                  border: "1px solid var(--serif-border)",
                  borderRadius: 6,
                  background: "var(--serif-muted)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  lineHeight: 1.7,
                  whiteSpace: "pre",
                  color: "var(--serif-foreground)",
                }}
              >
                {detail.code}
              </pre>
            </DetailSection>
          </div>
        ) : null}
      </Modal>
    </Drawer>
  );
}
