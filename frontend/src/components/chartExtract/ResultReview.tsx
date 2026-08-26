import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Descriptions,
  Divider,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import type { ExtractChartResult } from "../../api";
import { FONT_MONO } from "../../theme";
import { buildCsv, downloadTextFile, formatValue, toSeries } from "./utils";
import { TracedChartPreview } from "./TracedChartPreview";

const { Text, Paragraph } = Typography;

// ---------------------------------------------------------------------------
// Result summary & export panel (shared by image + PDF flows)
// ---------------------------------------------------------------------------

interface ResultReviewProps {
  result: ExtractChartResult;
  /** Image src to overlay traced curves on (optional). */
  imageSrc?: string;
  baseName: string;
  onApplyToWorkspace: (navText: string, reviewConfirmed: boolean) => void;
}

const CHART_REVIEW_REASON_LABELS: Record<string, string> = {
  trace_unreliable: "曲线连续性或覆盖率未通过质量门",
  y_axis_unverified: "纵轴刻度尚未完成校准",
  x_axis_unverified: "横轴日期尚未完成校准",
  review_required: "后端未确认自动采用条件",
  manual_digitization_candidate: "图像识别结果仅是候选值",
};

/** Shows traced curves, per-curve stats, a data table and export actions. */
export function ResultReview({ result, imageSrc, baseName, onApplyToWorkspace }: ResultReviewProps): React.JSX.Element {
  const [selectedCurve, setSelectedCurve] = useState<string>("");
  const [adjustMode, setAdjustMode] = useState(false);
  const [overlayAdjustment, setOverlayAdjustment] = useState<[number, number]>([0, 0]);
  const [reviewConfirmed, setReviewConfirmed] = useState(false);
  const requiresReview = result.review_required !== false;
  const reviewReasons = requiresReview
    ? (result.review_reasons?.length ? result.review_reasons : ["review_required"])
    : [];
  useEffect(() => setReviewConfirmed(false), [result]);

  const curveSeries = useMemo(
    () =>
      result.curves.map((curve) => ({
        curve,
        series: toSeries(curve.points),
      })),
    [result],
  );

  const activeName = selectedCurve || curveSeries[0]?.curve.name || "";
  const active = curveSeries.find((c) => c.curve.name === activeName) ?? curveSeries[0];
  const valuePerPixel = useMemo(() => {
    const points = active?.curve.points.filter((p) => p.value !== null) ?? [];
    if (points.length < 2) return 0;
    const meanY = points.reduce((sum, p) => sum + p.y_px, 0) / points.length;
    const meanV = points.reduce((sum, p) => sum + (p.value ?? 0), 0) / points.length;
    const variance = points.reduce((sum, p) => sum + (p.y_px - meanY) ** 2, 0);
    return variance ? points.reduce((sum, p) => sum + (p.y_px - meanY) * ((p.value ?? 0) - meanV), 0) / variance : 0;
  }, [active]);
  const activeAdjustedSeries = useMemo(() => (active?.series ?? []).map((point) => ({
    ...point, value: point.value + valuePerPixel * overlayAdjustment[1],
  })), [active, valuePerPixel, overlayAdjustment]);

  const handleExportCsv = useCallback(
    (all: boolean) => {
      const payload = all
        ? curveSeries.map((c) => ({ name: c.curve.name || "曲线", series: c.series }))
        : active
          ? [{ name: active.curve.name || "曲线", series: activeAdjustedSeries }]
          : [];
      const csv = buildCsv(payload);
      if (!csv) return;
      downloadTextFile(csv, `${baseName}_${all ? "全部曲线" : activeName}.csv`, "text/csv;charset=utf-8");
    },
    [curveSeries, active, activeAdjustedSeries, activeName, baseName],
  );

  const handleExportXlsx = useCallback(async () => {
    const XLSX = await import("xlsx");
    const wb = XLSX.utils.book_new();
    for (const c of curveSeries) {
      const rows = c.series.map((p) => ({ 日期: p.date, 数值: Number(p.value.toFixed(6)) }));
      const ws = XLSX.utils.json_to_sheet(rows);
      ws["!cols"] = [{ wch: 12 }, { wch: 14 }];
      const sheetName = (c.curve.name || "曲线").replace(/[\\/:*?[\]]/g, "_").slice(0, 28);
      XLSX.utils.book_append_sheet(wb, ws, sheetName);
    }
    XLSX.writeFile(wb, `${baseName}_提取结果.xlsx`);
  }, [curveSeries, baseName]);

  const handleCopy = useCallback(async () => {
    if (!active) return;
    const text = activeAdjustedSeries.map((p) => `${p.date},${p.value.toFixed(6)}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard may be unavailable; fall back silently.
    }
  }, [active, activeAdjustedSeries]);

  const handleApply = useCallback(() => {
    if (!active) return;
    const titleYear = result.structure?.chart_title.match(/(20\d{2})/)?.[1];
    const text = activeAdjustedSeries.map((p) => {
      const date = titleYear && /^\d{2}-\d{2}$/.test(p.date) ? `${titleYear}-${p.date}` : p.date;
      return `${date},${p.value.toFixed(6)}`;
    }).join("\n");
    onApplyToWorkspace(text, !requiresReview || reviewConfirmed);
  }, [active, activeAdjustedSeries, onApplyToWorkspace, requiresReview, result.structure?.chart_title, reviewConfirmed]);

  if (result.error && result.curves.length === 0) {
    return <Alert type="warning" showIcon message={result.error} />;
  }

  const tableData = active
    ? activeAdjustedSeries.map((p, index) => ({ key: index, date: p.date, value: p.value }))
    : [];

  return (
    <div>
      {imageSrc && (
        <>
          <TracedChartPreview src={imageSrc} curves={active ? [active.curve] : []} maxHeight={380} curveOffset={result.crop_offset} overlayAdjustment={overlayAdjustment} adjustingOverlay={adjustMode} onOverlayAdjustmentChange={setOverlayAdjustment} overlayColor="#2563eb" overlayDash="10 6" />
          <Space style={{ marginTop: 8 }} wrap>
            <Button size="small" type={adjustMode ? "primary" : "default"} onClick={() => setAdjustMode((value) => !value)} disabled={!active}>调整覆盖曲线</Button>
            <Button size="small" onClick={() => setOverlayAdjustment([0, 0])} disabled={overlayAdjustment[0] === 0 && overlayAdjustment[1] === 0}>还原校准</Button>
            {adjustMode && <Text type="secondary" style={{ fontSize: 12 }}>直接拖动彩色覆盖线使其贴合原图；松开后净值会按纵轴标定实时修正。</Text>}
          </Space>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 6, fontSize: 11, color: "var(--serif-muted-foreground)" }}>
            <span><i style={{ display: "inline-block", width: 22, borderTop: "2px solid #2563eb", marginRight: 5, verticalAlign: "middle" }} />蓝色虚线：识别候选</span>
            <span><i style={{ display: "inline-block", width: 22, borderTop: "2px solid #555", marginRight: 5, verticalAlign: "middle" }} />原图实线：披露曲线</span>
          </div>
        </>
      )}

      {result.curves.length > 0 && (
        <Descriptions
          size="small"
          column={{ xs: 1, sm: 3 }}
          style={{ marginTop: 14 }}
          items={[
            { key: "freq", label: "频率", children: result.frequency },
            { key: "conf", label: "置信度", children: (result.confidence * 100).toFixed(0) + "%" },
            { key: "src", label: "结构来源", children: result.structure?.source === "manual" ? "手动" : "VLM" },
          ]}
        />
      )}

      {requiresReview && (
        <Alert
          type="warning"
          showIcon
          style={{ marginTop: 14 }}
          message="当前结果只能作为候选，不能自动采用"
          description={<>
            <div>{reviewReasons.map((reason) => CHART_REVIEW_REASON_LABELS[reason] ?? reason).join("；")}</div>
            <Checkbox checked={reviewConfirmed} onChange={(event) => setReviewConfirmed(event.target.checked)} style={{ marginTop: 8 }}>
              我已对照原图完成曲线、坐标和日期校准，确认当前曲线可用于研究
            </Checkbox>
          </>}
        />
      )}

      <Divider style={{ margin: "16px 0 12px" }} />

      {curveSeries.map(({ curve, series }) => {
        const values = series.map((p) => p.value);
        const min = values.length ? Math.min(...values) : undefined;
        const max = values.length ? Math.max(...values) : undefined;
        const last = values.length ? values[values.length - 1] : undefined;
        return (
          <div
            key={curve.name}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "6px 10px",
              borderRadius: 6,
              border: activeName === curve.name ? "1px solid var(--serif-accent)" : "1px solid transparent",
              background: activeName === curve.name ? "var(--serif-muted)" : "transparent",
              cursor: "pointer",
              marginBottom: 4,
            }}
            onClick={() => setSelectedCurve(curve.name)}
          >
            <span
              style={{
                width: 14,
                height: 14,
                borderRadius: 3,
                background: curve.color_hex || "#ccc",
                border: "1px solid rgba(0,0,0,0.15)",
                flexShrink: 0,
              }}
            />
            <Text strong style={{ flex: 1 }}>
              {curve.name || "未命名曲线"}
              {curve.is_benchmark && <Tag style={{ marginLeft: 6 }}>基准</Tag>}
            </Text>
            <Text type="secondary" style={{ fontFamily: FONT_MONO, fontSize: 12 }}>
              {series.length} 点
              {min !== undefined && max !== undefined && ` · ${formatValue(min)} ~ ${formatValue(max)}`}
              {last !== undefined && ` · 末值 ${formatValue(last)}`}
            </Text>
          </div>
        );
      })}

      {result.error && (
        <Alert type="warning" showIcon message={result.error} style={{ marginTop: 8 }} closable />
      )}

      <Table
        size="small"
        dataSource={tableData}
        pagination={{ pageSize: 12, showSizeChanger: false, size: "small" }}
        scroll={{ y: 260 }}
        style={{ marginTop: 12 }}
        columns={[
          { title: "日期", dataIndex: "date", width: 130 },
          {
            title: `数值（${activeName || "曲线"}）`,
            dataIndex: "value",
            render: (v: number) => <span style={{ fontFamily: FONT_MONO }}>{v.toFixed(6)}</span>,
          },
        ]}
      />

      <Space wrap style={{ marginTop: 14 }}>
        <Button onClick={() => handleExportCsv(false)} disabled={!active}>导出当前曲线 CSV</Button>
        <Button onClick={() => handleExportCsv(true)}>导出全部曲线 CSV</Button>
        <Button onClick={() => void handleExportXlsx()}>导出 XLSX</Button>
        <Button onClick={() => void handleCopy()} disabled={!active}>复制日期,数值</Button>
        <Tooltip title="将当前曲线写入主工作区净值文本框，核验后再计算">
          <Button type="primary" onClick={handleApply} disabled={!active || (requiresReview && !reviewConfirmed)}>填入主工作区</Button>
        </Tooltip>
      </Space>
      <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
        数值为像素校准值，纵轴口径（净值 / 累计收益率）以原图为准，导出后请人工核验。
      </Paragraph>
    </div>
  );
}
