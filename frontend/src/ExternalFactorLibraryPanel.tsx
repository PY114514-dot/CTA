import React, { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Dropdown, Empty, Select, Space, Table, Tag, Typography, Upload } from "antd";
import { DownloadOutlined, ReloadOutlined, UploadOutlined } from "@ant-design/icons";
import { externalFactorExportUrl, getGuotaiJunanExternalFactors, importExternalFactorReport, verifyGuotaiJunanExternalFactors, type ExternalFactorObservation } from "./api";

const { Text } = Typography;

export default function ExternalFactorLibraryPanel(): React.JSX.Element {
  const [items, setItems] = useState<ExternalFactorObservation[]>([]);
  const [reportDate, setReportDate] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [message, setMessage] = useState<string>();
  const verifyReport = async (): Promise<void> => {
    if (!reportDate) return;
    setLoading(true); setError(undefined);
    try { const result = await verifyGuotaiJunanExternalFactors(reportDate); setMessage(`已将 ${reportDate} 的 ${result.verified_rows} 条观测标记为已核验。`); await load(); }
    catch (e) { setError(e instanceof Error ? e.message : "核验失败"); }
    finally { setLoading(false); }
  };
  const load = async (): Promise<void> => {
    setLoading(true); setError(undefined);
    try { const response = await getGuotaiJunanExternalFactors(); setItems(response.observations); setReportDate((previous) => previous ?? response.latest_as_of_date ?? undefined); }
    catch (e) { setError(e instanceof Error ? e.message : "因子库加载失败"); }
    finally { setLoading(false); }
  };
  useEffect(() => { void load(); }, []);
  const dates = useMemo(() => [...new Set(items.map((item) => item.as_of_date))].sort((a, b) => b.localeCompare(a)), [items]);
  const rows = useMemo(() => {
    const selected = items.filter((item) => item.as_of_date === reportDate);
    return [...new Map(selected.map((item) => [item.factor_id, item])).values()].map((item) => ({
      key: item.factor_id, factor: item.source_factor_name, verified: item.verified,
      oneWeek: selected.find((x) => x.factor_id === item.factor_id && x.horizon === "1w")?.return_pct,
      oneMonth: selected.find((x) => x.factor_id === item.factor_id && x.horizon === "1m")?.return_pct,
      threeMonth: selected.find((x) => x.factor_id === item.factor_id && x.horizon === "3m")?.return_pct,
      halfYear: selected.find((x) => x.factor_id === item.factor_id && x.horizon === "6m")?.return_pct,
      oneYear: selected.find((x) => x.factor_id === item.factor_id && x.horizon === "1y")?.return_pct,
    }));
  }, [items, reportDate]);
  const importReport = async (file: File): Promise<void> => { setLoading(true); setError(undefined); try { const r = await importExternalFactorReport(file, "国泰君安期货"); setMessage(r.duplicate ? `已存在 ${r.organization_name} ${r.as_of_date} 的相同源文件。` : `已导入 ${r.organization_name} ${r.as_of_date}：新增 ${r.rows_added} 条观测。`); await load(); } catch (e) { setError(e instanceof Error ? e.message : "周报导入失败"); } finally { setLoading(false); } };
  const valueCell = (value: string | undefined): React.JSX.Element => <span style={{ color: Number(value) >= 0 ? "#cf1322" : "#1677ff" }}>{value === undefined ? "—" : `${Number(value).toFixed(2)}%`}</span>;
  return <Card title="外部 CTA 因子库" extra={<Space size={8} wrap><Upload accept="application/pdf,image/png,image/jpeg" showUploadList={false} beforeUpload={(file) => { void importReport(file); return false; }}><Button size="small" icon={<UploadOutlined />}>导入周报</Button></Upload><Button size="small" disabled={!reportDate} loading={loading} onClick={() => void verifyReport()}>核验</Button><Dropdown menu={{ items: [{ key: "csv", label: <a href={externalFactorExportUrl("csv")}>导出 CSV</a> }, { key: "json", label: <a href={externalFactorExportUrl("json")}>导出 JSON</a> }] }}><Button size="small" icon={<DownloadOutlined />}>导出</Button></Dropdown><Button size="small" icon={<ReloadOutlined />} loading={loading} onClick={() => void load()} aria-label="刷新外部因子数据" /></Space>} style={{ marginTop: 24 }}>
    <Alert type="info" showIcon style={{ marginBottom: 12 }} message="外部披露基准，不等同于自建因子；待核验表示尚未人工对照原始周报。" />
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
      <Text type="secondary">来源：国泰君安期货</Text>
      <Select value={reportDate} onChange={setReportDate} style={{ width: 172 }} options={dates.map((value) => ({ value, label: `报告日 ${value}` }))} />
    </div>
    {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 12 }} />}
    {message && <Alert type="success" showIcon message={message} style={{ marginBottom: 12 }} closable onClose={() => setMessage(undefined)} />}
    {rows.length ? <Table size="small" rowKey="key" pagination={false} dataSource={rows} columns={[
      { title: "因子", dataIndex: "factor", width: 170 }, { title: "近一周", dataIndex: "oneWeek", render: valueCell }, { title: "近一月", dataIndex: "oneMonth", render: valueCell }, { title: "近三月", dataIndex: "threeMonth", render: valueCell }, { title: "近半年", dataIndex: "halfYear", render: valueCell }, { title: "近一年", dataIndex: "oneYear", render: valueCell },
      { title: "状态", dataIndex: "verified", render: (v: string) => <Tag color={v === "true" ? "green" : "gold"}>{v === "true" ? "已核验" : "待核验"}</Tag> },
    ]} /> : !loading && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="尚未导入周报因子数据" />}
  </Card>;
}
