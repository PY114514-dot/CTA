/**
 * Chart extraction drawer — turn NAV curve images into numeric series.
 *
 * Two intake paths:
 *  1. Single image: user previews the chart, defines curves (name + color via
 *     hex input or click-to-pick on the image), then deterministic CV pixel
 *     tracing extracts the numbers. VLM is optional (auto-detect curves).
 *  2. PDF: full pipeline (render → detect → VLM structure → CV trace) with
 *     job polling and per-region results.
 *
 * VLM only reads chart STRUCTURE; all values come from pixel tracing.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  ColorPicker,
  Descriptions,
  Divider,
  Drawer,
  Empty,
  Input,
  Progress,
  Space,
  Spin,
  Steps,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
} from "antd";
import type { Color } from "antd/es/color-picker";
import {
  extractPdf,
  extractRegionImageUrl,
  extractSingleImage,
  getExtractJobPreview,
  getExtractJobStatus,
  getVlmConfig,
  sampleColorAtPixel,
  type ExtractChartResult,
  type ExtractJobPreview,
  type ExtractJobStatus,
  type ExtractTracedCurve,
  type ExtractTracedPoint,
  type VlmConfig,
} from "./api";
import { FONT_MONO } from "./theme";

const { Text, Paragraph } = Typography;

const IMAGE_ACCEPT = "image/png,image/jpeg,image/webp";
const PDF_ACCEPT = "application/pdf";

// ---------------------------------------------------------------------------
// Local types & helpers
// ---------------------------------------------------------------------------

/** One editable curve row in the configuration step. */
interface CurveRow {
  id: string;
  name: string;
  colorHex: string;
  isBenchmark: boolean;
}

/** Aggregated (date → value) point for tables and export. */
interface SeriesPoint {
  date: string;
  value: number;
}

let curveIdCounter = 0;
function nextCurveId(): string {
  curveIdCounter += 1;
  return `curve-${curveIdCounter}`;
}

/** Collapse per-pixel points into one mean value per date, sorted by date. */
function toSeries(points: ExtractTracedPoint[]): SeriesPoint[] {
  const byDate = new Map<string, number[]>();
  for (const p of points) {
    if (p.value === null || p.value === undefined || !p.date) continue;
    const bucket = byDate.get(p.date);
    if (bucket) bucket.push(p.value);
    else byDate.set(p.date, [p.value]);
  }
  return [...byDate.entries()]
    .map(([date, values]) => ({
      date,
      value: values.reduce((a, b) => a + b, 0) / values.length,
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function formatValue(v: number): string {
  return v.toFixed(4);
}

/** Build CSV text from one or more named series (date column + value columns). */
function buildCsv(seriesByName: Array<{ name: string; series: SeriesPoint[] }>): string {
  if (seriesByName.length === 0) return "";
  const dates = [...new Set(seriesByName.flatMap((s) => s.series.map((p) => p.date)))].sort();
  const header = ["date", ...seriesByName.map((s) => s.name)].join(",");
  const rows = dates.map((date) => {
    const cells = seriesByName.map((s) => {
      const point = s.series.find((p) => p.date === date);
      return point ? point.value.toFixed(6) : "";
    });
    return [date, ...cells].join(",");
  });
  return [header, ...rows].join("\n");
}

function downloadTextFile(content: string, filename: string, mime: string): void {
  const blob = new Blob(["\uFEFF" + content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * Parse the Y-axis tick input. Accepts either an explicit bottom→top list
 * ("0.995, 1.005, 1.015, …") or a compact "min, max, step" triple
 * ("0.995, 1.075, 0.01") which is expanded into the full evenly-spaced list.
 * The triple is only treated as a range when it yields ≥3 ticks, so a genuine
 * 3-tick axis is never misread as a range.
 */
function parseYTicksInput(text: string): string[] {
  const tokens = text.split(/[,，\s]+/).map((s) => s.trim()).filter(Boolean);
  if (tokens.length === 3) {
    const min = Number(tokens[0]);
    const max = Number(tokens[1]);
    const step = Number(tokens[2]);
    if (Number.isFinite(min) && Number.isFinite(max) && Number.isFinite(step) && step > 0 && max > min) {
      const steps = Math.round((max - min) / step);
      if (steps >= 2) {
        const decimals = Math.max(...tokens.map((t) => (t.split(".")[1] || "").length));
        const ticks: string[] = [];
        for (let i = 0; i <= steps; i++) {
          ticks.push((min + i * step).toFixed(decimals));
        }
        return ticks;
      }
    }
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Traced chart preview — image + SVG overlay of traced curves
// ---------------------------------------------------------------------------

interface TracedChartPreviewProps {
  /** Image source (object URL or remote URL). */
  src: string;
  /** Traced curves to overlay (coordinates in natural image pixels). */
  curves: ExtractTracedCurve[];
  /** When set, clicks sample a color for this curve id. */
  picking?: boolean;
  onPick?: (x: number, y: number) => void;
  maxHeight?: number | string;
  /** (x, y) offset of the traced curves' coordinate space within this image
      (non-zero when the backend auto-cropped the chart from a larger page). */
  curveOffset?: [number, number];
  /** Extra offset applied only to the traced overlay during manual calibration. */
  overlayAdjustment?: [number, number];
  adjustingOverlay?: boolean;
  onOverlayAdjustmentChange?: (offset: [number, number]) => void;
  /** High-contrast style keeps the candidate visible over the original line. */
  overlayColor?: string;
  overlayDash?: string;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

/**
 * Renders a chart image with traced curves overlaid. Supports zoom (wheel /
 * double-click / buttons), pan (drag) and click-to-pick. Coordinates are mapped
 * exactly at any zoom level, so picking stays accurate even on thin curves.
 */
function TracedChartPreview({ src, curves, picking, onPick, maxHeight = 420, curveOffset = [0, 0], overlayAdjustment = [0, 0], adjustingOverlay = false, onOverlayAdjustmentChange, overlayColor, overlayDash }: TracedChartPreviewProps): React.JSX.Element {
  const [ox, oy] = curveOffset;
  const [ax, ay] = overlayAdjustment;
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const [containerW, setContainerW] = useState(0);
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 });
  const dragRef = useRef<{ startX: number; startY: number; tx: number; ty: number; moved: boolean; ax: number; ay: number } | null>(null);

  const maxH = typeof maxHeight === "number" ? maxHeight : parseInt(String(maxHeight), 10) || 420;

  // Track container width so the base (fit) size stays correct on resize.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setContainerW(entry.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Reset the zoom/pan whenever a new image is chosen. (dims is refreshed by the
  // hidden measurement image's onLoad below — it is always mounted, so there is
  // no dependency on the display image, which only renders once dims is known.)
  useEffect(() => {
    setView({ scale: 1, tx: 0, ty: 0 });
  }, [src]);

  // Base displayed size: contain-fit the natural image within containerW x maxH.
  const base = useMemo(() => {
    if (!dims || !containerW) return null;
    const s = Math.min(containerW / dims.w, maxH / dims.h);
    return { w: dims.w * s, h: dims.h * s };
  }, [dims, containerW, maxH]);

  // Keep the image centered when it fits, or clamped inside view when zoomed.
  const clampPan = useCallback(
    (tx: number, ty: number, scale: number) => {
      if (!base) return { tx, ty };
      const dispW = base.w * scale;
      const dispH = base.h * scale;
      const ntx = dispW <= containerW ? (containerW - dispW) / 2 : Math.min(0, Math.max(containerW - dispW, tx));
      const nty = dispH <= maxH ? (maxH - dispH) / 2 : Math.min(0, Math.max(maxH - dispH, ty));
      return { tx: ntx, ty: nty };
    },
    [base, containerW, maxH],
  );

  const clampPanRef = useRef(clampPan);
  clampPanRef.current = clampPan;

  // Non-passive wheel listener so we can preventDefault page scrolling.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      setView((prev) => {
        const scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev.scale * (e.deltaY < 0 ? 1.25 : 0.8)));
        const k = scale / prev.scale;
        return { scale, ...clampPanRef.current(cx - (cx - prev.tx) * k, cy - (cy - prev.ty) * k, scale) };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // Convert a container-relative point to natural image pixels.
  const toNatural = useCallback(
    (cx: number, cy: number): { x: number; y: number } | null => {
      if (!dims || !base) return null;
      const bx = (cx - view.tx) / view.scale;
      const by = (cy - view.ty) / view.scale;
      return { x: (bx / base.w) * dims.w, y: (by / base.h) * dims.h };
    },
    [dims, base, view],
  );

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    dragRef.current = { startX: e.clientX, startY: e.clientY, tx: view.tx, ty: view.ty, moved: false, ax, ay };
  }, [view.tx, view.ty, ax, ay]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) drag.moved = true;
    if (drag.moved && adjustingOverlay && onOverlayAdjustmentChange) {
      const naturalDx = (dx / view.scale) * (dims?.w ?? 1) / (base?.w ?? 1);
      const naturalDy = (dy / view.scale) * (dims?.h ?? 1) / (base?.h ?? 1);
      onOverlayAdjustmentChange([drag.ax + naturalDx, drag.ay + naturalDy]);
    } else if (drag.moved) {
      setView((prev) => ({ scale: prev.scale, ...clampPanRef.current(drag.tx + dx, drag.ty + dy, prev.scale) }));
    }
  }, [adjustingOverlay, onOverlayAdjustmentChange, view.scale, dims, base]);

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      const drag = dragRef.current;
      dragRef.current = null;
      // A stationary mouse-up is a click: sample a color when picking.
      if (drag && !drag.moved && picking && onPick) {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        const pt = toNatural(e.clientX - rect.left, e.clientY - rect.top);
        if (pt && pt.x >= 0 && pt.y >= 0 && dims && pt.x <= dims.w && pt.y <= dims.h) {
          onPick(pt.x, pt.y);
        }
      }
    },
    [picking, onPick, toNatural, dims],
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      setView((prev) => (prev.scale > 1.01 ? { scale: 1, ...clampPanRef.current(0, 0, 1) } : { scale: 3, ...clampPanRef.current(cx - (cx - prev.tx) * 3, cy - (cy - prev.ty) * 3, 3) }));
    },
    [],
  );

  const zoomStep = useCallback((factor: number) => {
    setView((prev) => {
      const scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev.scale * factor));
      const k = scale / prev.scale;
      const cx = containerW / 2;
      const cy = maxH / 2;
      return { scale, ...clampPanRef.current(cx - (cx - prev.tx) * k, cy - (cy - prev.ty) * k, scale) };
    });
  }, [containerW, maxH]);

  const zoomed = view.scale > 1.01;
  const cursor = picking ? "crosshair" : adjustingOverlay ? "move" : zoomed ? "grab" : "default";

  return (
    <div>
      <div
        ref={containerRef}
        onDoubleClick={handleDoubleClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { dragRef.current = null; }}
        style={{
          position: "relative",
          width: "100%",
          height: maxH,
          border: "1px solid var(--serif-border)",
          borderRadius: 6,
          background: "var(--serif-background)",
          overflow: "hidden",
          cursor,
          userSelect: "none",
        }}
      >
        {/* Hidden measurement image: always mounted so onLoad fires on every
            src change, even before the display image is rendered. */}
        <img
          src={src}
          alt=""
          aria-hidden
          draggable={false}
          onLoad={(event) => {
            const el = event.currentTarget;
            setDims({ w: el.naturalWidth, h: el.naturalHeight });
          }}
          style={{ position: "absolute", width: 0, height: 0, opacity: 0, pointerEvents: "none" }}
        />

        {base && dims && (
          <div
            style={{
              position: "absolute",
              width: base.w,
              height: base.h,
              transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
              transformOrigin: "0 0",
            }}
          >
            <img
              src={src}
              alt="净值曲线图"
              draggable={false}
              style={{ display: "block", width: "100%", height: "100%", pointerEvents: "none" }}
            />
            {curves.length > 0 && (
              <svg
                viewBox={`0 0 ${dims.w} ${dims.h}`}
                preserveAspectRatio="none"
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
              >
                {curves.map((curve, index) => {
                  const pts = curve.points.filter((p) => p.value !== null);
                  if (pts.length === 0) return null;
                  const pathData = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x_px + ox + ax},${p.y_px + oy + ay}`).join(" ");
                  return (
                    <path
                      key={`${curve.name}-${index}`}
                      d={pathData}
                      fill="none"
                      stroke={overlayColor || curve.color_hex || "#FF00FF"}
                      strokeDasharray={overlayDash}
                      strokeWidth={Math.max(dims.w, dims.h) * 0.0022}
                      strokeOpacity={0.85}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  );
                })}
              </svg>
            )}
          </div>
        )}

        {/* Floating zoom controls */}
        <div
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            display: "flex",
            alignItems: "center",
            gap: 4,
            background: "rgba(255,255,255,0.9)",
            border: "1px solid var(--serif-border)",
            borderRadius: 6,
            padding: "2px 6px",
            fontSize: 12,
          }}
          onMouseDown={(e) => e.stopPropagation()}
          onDoubleClick={(e) => e.stopPropagation()}
        >
          <span role="button" tabIndex={0} onClick={() => zoomStep(0.8)} style={{ cursor: "pointer", padding: "0 4px" }}>−</span>
          <span style={{ fontFamily: FONT_MONO, minWidth: 40, textAlign: "center" }}>{Math.round(view.scale * 100)}%</span>
          <span role="button" tabIndex={0} onClick={() => zoomStep(1.25)} style={{ cursor: "pointer", padding: "0 4px" }}>＋</span>
          {zoomed && (
            <span
              role="button"
              tabIndex={0}
              onClick={() => setView({ scale: 1, ...clampPan(0, 0, 1) })}
              style={{ cursor: "pointer", padding: "0 4px", color: "var(--serif-accent)" }}
            >
              复位
            </span>
          )}
        </div>
      </div>
      <div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginTop: 4 }}>
        滚轮缩放 · 拖动平移 · 双击放大/还原{picking ? " · 单击曲线取色" : ""}
      </div>
    </div>
  );
}

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
function ResultReview({ result, imageSrc, baseName, onApplyToWorkspace }: ResultReviewProps): React.JSX.Element {
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

// ---------------------------------------------------------------------------
// Main drawer
// ---------------------------------------------------------------------------

interface ChartExtractDrawerProps {
  open: boolean;
  onClose: () => void;
  /** Write an extracted "date,value" series into the main workspace textarea. */
  onApplyToWorkspace: (navText: string, reviewConfirmed: boolean) => void;
}

type StepKey = "upload" | "configure" | "results";

export default function ChartExtractDrawer({
  open,
  onClose,
  onApplyToWorkspace,
}: ChartExtractDrawerProps): React.JSX.Element {
  const [step, setStep] = useState<StepKey>("upload");

  // --- image intake ---
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");

  // --- curve configuration ---
  const [curves, setCurves] = useState<CurveRow[]>([]);
  const [pickingId, setPickingId] = useState<string | null>(null);
  const [isSampling, setIsSampling] = useState(false);
  const [yTicksText, setYTicksText] = useState<string>("");

  // --- extraction result (single image) ---
  const [result, setResult] = useState<ExtractChartResult | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState<string>("");

  // --- pdf flow ---
  const [pdfName, setPdfName] = useState<string>("");
  const [pdfJob, setPdfJob] = useState<ExtractJobStatus | null>(null);
  const [pdfPreview, setPdfPreview] = useState<ExtractJobPreview | null>(null);
  const [expandedPdfResult, setExpandedPdfResult] = useState<number | null>(null);

  const [vlmConfig, setVlmConfig] = useState<VlmConfig | null>(null);

  const pollTimerRef = useRef<number | null>(null);

  // Load VLM config whenever the drawer opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void getVlmConfig()
      .then((config) => {
        if (!cancelled) setVlmConfig(config);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [open]);

  // Clean up the object URL and poll timer on close / unmount.
  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current);
    };
  }, [imageUrl]);

  // ESC cancels color-picking mode (matches the hint shown while picking).
  useEffect(() => {
    if (!pickingId) return;
    const onKeyDown = (e: KeyboardEvent): void => {
      if (e.key === "Escape") setPickingId(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [pickingId]);

  const resetAll = useCallback(() => {
    setStep("upload");
    setImageFile(null);
    setImageUrl("");
    setCurves([]);
    setPickingId(null);
    setYTicksText("");
    setResult(null);
    setIsExtracting(false);
    setError("");
    setPdfName("");
    setPdfJob(null);
    setPdfPreview(null);
    setExpandedPdfResult(null);
  }, []);

  // --- image selection ---
  const handleSelectImage = useCallback((file: File) => {
    setImageFile(file);
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    setResult(null);
    setError("");
    setCurves([
      { id: nextCurveId(), name: "产品净值", colorHex: "#E03030", isBenchmark: false },
    ]);
    setStep("configure");
  }, []);

  // --- curve row editing ---
  const updateCurve = useCallback((id: string, patch: Partial<CurveRow>) => {
    setCurves((rows) => rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }, []);

  const addCurve = useCallback(() => {
    setCurves((rows) => [
      ...rows,
      { id: nextCurveId(), name: `曲线${rows.length + 1}`, colorHex: "#1E5799", isBenchmark: false },
    ]);
  }, []);

  const removeCurve = useCallback((id: string) => {
    setCurves((rows) => rows.filter((row) => row.id !== id));
    setPickingId((current) => (current === id ? null : current));
  }, []);

  // --- click-to-pick color ---
  const handlePickColor = useCallback(
    async (x: number, y: number) => {
      if (!pickingId || !imageFile) return;
      setIsSampling(true);
      try {
        const sampled = await sampleColorAtPixel(imageFile, x, y);
        updateCurve(pickingId, { colorHex: sampled.color_hex });
        setPickingId(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "取色失败");
      } finally {
        setIsSampling(false);
      }
    },
    [pickingId, imageFile, updateCurve],
  );

  // --- manual extraction (deterministic CV, no VLM) ---
  const handleManualExtract = useCallback(async () => {
    if (!imageFile) return;
    const valid = curves.filter((c) => c.colorHex.trim());
    if (valid.length === 0) {
      setError("请至少配置一条曲线颜色（可输入色值或在图上点击取色）。");
      return;
    }
    setIsExtracting(true);
    setError("");
    try {
      const yTicks = parseYTicksInput(yTicksText);
      const response = await extractSingleImage(
        imageFile,
        valid.map((c) => ({
          name: c.name,
          color_hex: c.colorHex,
          is_benchmark: c.isBenchmark,
        })),
        { useVlm: false, yTicks: yTicks.length > 0 ? yTicks : undefined },
      );
      setResult(response.result);
      setStep("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "提取失败");
    } finally {
      setIsExtracting(false);
    }
  }, [imageFile, curves, yTicksText]);

  // --- VLM auto detection (structure + trace in one call) ---
  const handleAutoExtract = useCallback(async () => {
    if (!imageFile) return;
    setIsExtracting(true);
    setError("");
    try {
      const response = await extractSingleImage(imageFile, undefined, { useVlm: true });
      setResult(response.result);
      // Pre-fill the curve editor from the VLM structure for later tweaking.
      if (response.result.structure) {
        const detected = response.result.structure.curves
          .map((c) => ({
            id: nextCurveId(),
            name: String(c.name ?? "曲线"),
            colorHex: String(c.color_hex ?? "#888888"),
            isBenchmark: Boolean(c.is_benchmark),
          }));
        if (detected.length > 0) setCurves(detected);
      }
      setStep("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "VLM 识别失败");
    } finally {
      setIsExtracting(false);
    }
  }, [imageFile]);

  // --- pdf flow ---
  const pollPdfJob = useCallback((jobId: string) => {
    const tick = async (): Promise<void> => {
      try {
        const status = await getExtractJobStatus(jobId);
        setPdfJob(status);
        if (status.status === "completed" || status.status === "failed") {
          const preview = await getExtractJobPreview(jobId);
          setPdfPreview(preview);
          return;
        }
        pollTimerRef.current = window.setTimeout(() => void tick(), 1500);
      } catch (err) {
        setError(err instanceof Error ? err.message : "查询任务失败");
      }
    };
    void tick();
  }, []);

  const handleSelectPdf = useCallback(
    async (file: File) => {
      setPdfName(file.name);
      setPdfPreview(null);
      setPdfJob(null);
      setError("");
      setStep("results");
      try {
        const response = await extractPdf(file, { useVlm: true });
        pollPdfJob(response.job_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "PDF 提取失败");
      }
    },
    [pollPdfJob],
  );

  const stepIndex = step === "upload" ? 0 : step === "configure" ? 1 : 2;
  const vlmReady = Boolean(vlmConfig?.api_key_set);

  return (
    <Drawer
      title="图表净值提取（CV 像素追踪）"
      open={open}
      onClose={onClose}
      width={760}
      destroyOnHidden
      extra={
        <Button size="small" onClick={resetAll}>
          重新开始
        </Button>
      }
    >
      <Steps
        size="small"
        current={stepIndex}
        style={{ marginBottom: 20 }}
        items={[
          { title: "上传", description: "图片或 PDF" },
          { title: "配置曲线", description: "颜色 / 取色" },
          { title: "结果导出", description: "核验 / 下载" },
        ]}
      />

      {error && (
        <Alert
          type="error"
          showIcon
          closable
          message={error}
          onClose={() => setError("")}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* ---------------- Step 1: upload ---------------- */}
      {step === "upload" && (
        <div>
          <Paragraph type="secondary">
            私募产品净值大多只以路演 PDF 中的曲线图形式存在。这里从图片像素中还原数值序列：
            视觉模型（可选）只负责读取图表结构（曲线颜色、坐标刻度），真正的数值由确定性的像素追踪给出。
          </Paragraph>

          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            <div
              style={{
                border: "1px dashed var(--serif-border)",
                borderRadius: 8,
                padding: 20,
                background: "var(--serif-card)",
              }}
            >
              <Text strong>方式一：单张曲线图片（推荐，可手动取色）</Text>
              <Paragraph type="secondary" style={{ fontSize: 13, margin: "6px 0 12px" }}>
                上传净值曲线截图，随后在图上点击即可为每条曲线取色，零 API 成本。
              </Paragraph>
              <Upload
                accept={IMAGE_ACCEPT}
                maxCount={1}
                showUploadList={false}
                beforeUpload={(file) => {
                  handleSelectImage(file);
                  return false;
                }}
              >
                <Button type="primary">选择曲线图片</Button>
              </Upload>
            </div>

            <div
              style={{
                border: "1px dashed var(--serif-border)",
                borderRadius: 8,
                padding: 20,
                background: "var(--serif-card)",
              }}
            >
              <Text strong>方式二：整份 PDF（自动检测图表）</Text>
              <Paragraph type="secondary" style={{ fontSize: 13, margin: "6px 0 12px" }}>
                自动渲染、定位图表区域并用 VLM 读取结构后逐图追踪。
                {!vlmReady && (
                  <Tag color="warning" style={{ marginLeft: 8 }}>未配置 VLM API Key</Tag>
                )}
              </Paragraph>
              <Upload
                accept={PDF_ACCEPT}
                maxCount={1}
                showUploadList={false}
                beforeUpload={(file) => {
                  void handleSelectPdf(file);
                  return false;
                }}
              >
                <Button>选择 PDF 文件</Button>
              </Upload>
              {vlmConfig && (
                <div style={{ marginTop: 10, fontSize: 12, color: "var(--serif-muted-foreground)" }}>
                  VLM：{vlmConfig.provider} / {vlmConfig.model}
                  {vlmReady ? "（已配置 Key）" : "（未配置 Key，请在后端设置 DASHSCOPE_API_KEY）"}
                </div>
              )}
            </div>
          </Space>
        </div>
      )}

      {/* ---------------- Step 2: configure curves (image) ---------------- */}
      {step === "configure" && imageFile && (
        <div>
          <Space style={{ marginBottom: 10 }} wrap>
            <Button size="small" onClick={() => setStep("upload")}>← 重新选择</Button>
            <Text type="secondary" style={{ fontSize: 13 }}>{imageFile.name}</Text>
          </Space>

          <TracedChartPreview
            src={imageUrl}
            curves={[]}
            picking={pickingId !== null}
            onPick={(x, y) => void handlePickColor(x, y)}
            maxHeight={360}
          />

          {pickingId && (
            <Alert
              type="info"
              showIcon
              style={{ marginTop: 10 }}
              message={
                isSampling
                  ? "正在取色…"
                  : "取色模式：在上方图片中点击目标曲线像素即可自动填入颜色，ESC 可取消。"
              }
            />
          )}

          <Divider titlePlacement="left" style={{ margin: "18px 0 12px" }}>曲线配置</Divider>

          {curves.map((row) => (
            <div
              key={row.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "8px 10px",
                border: "1px solid var(--serif-border)",
                borderRadius: 6,
                marginBottom: 8,
                background: "var(--serif-card)",
                flexWrap: "wrap",
              }}
            >
              <Input
                value={row.name}
                onChange={(e) => updateCurve(row.id, { name: e.target.value })}
                placeholder="曲线名称"
                style={{ width: 150 }}
                size="middle"
              />
              <ColorPicker
                value={row.colorHex}
                onChange={(color: Color) => updateCurve(row.id, { colorHex: color.toHexString() })}
                showText
                size="middle"
              />
              <Input
                value={row.colorHex}
                onChange={(e) => updateCurve(row.id, { colorHex: e.target.value })}
                placeholder="#RRGGBB"
                style={{ width: 110, fontFamily: FONT_MONO }}
              />
              <Tooltip title="点击后在图上选取该曲线颜色">
                <Button
                  type={pickingId === row.id ? "primary" : "default"}
                  onClick={() => setPickingId(pickingId === row.id ? null : row.id)}
                >
                  {pickingId === row.id ? "取消取色" : "图上取色"}
                </Button>
              </Tooltip>
              <Checkbox
                checked={row.isBenchmark}
                onChange={(e) => updateCurve(row.id, { isBenchmark: e.target.checked })}
              >
                基准
              </Checkbox>
              <Button danger size="small" onClick={() => removeCurve(row.id)} style={{ marginLeft: "auto" }}>
                删除
              </Button>
            </div>
          ))}

          <Space wrap style={{ marginTop: 4 }}>
            <Button onClick={addCurve}>＋ 添加曲线</Button>
          </Space>

          <div style={{ marginTop: 14 }}>
            <Text strong style={{ fontSize: 13 }}>Y 轴刻度（自下而上）</Text>
            <Input
              value={yTicksText}
              onChange={(e) => setYTicksText(e.target.value)}
              placeholder="逐个填 0.95, 1.00, 1.05…  或简写 最小, 最大, 间距（如 0.995, 1.075, 0.01）"
              style={{ marginTop: 6, fontFamily: FONT_MONO }}
              allowClear
            />
            <Paragraph type="secondary" style={{ fontSize: 12, margin: "4px 0 0" }}>
              按图上 Y 轴刻度自下而上填写（逗号分隔）；也可只填「最小值, 最大值, 间距」三个数自动展开。
              像素位置由网格线 / 等距自动定位并校准为真实净值；留空则只返回像素位置。
              {yTicksText.trim() !== "" && (
                <span style={{ color: "var(--serif-accent)" }}>
                  {" "}（当前解析为 {parseYTicksInput(yTicksText).length} 个刻度）
                </span>
              )}
            </Paragraph>
          </div>

          <Divider style={{ margin: "18px 0 12px" }} />

          <Space wrap>
            <Button
              type="primary"
              loading={isExtracting}
              onClick={() => void handleManualExtract()}
              disabled={curves.filter((c) => c.colorHex.trim()).length === 0}
            >
              按指定颜色提取（CV，零成本）
            </Button>
            <Tooltip title={vlmReady ? "用视觉模型自动识别曲线结构后追踪" : "未配置 VLM API Key"}>
              <Button
                loading={isExtracting}
                onClick={() => void handleAutoExtract()}
                disabled={!vlmReady}
              >
                VLM 自动识别并提取
              </Button>
            </Tooltip>
          </Space>
          <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 10, marginBottom: 0 }}>
            提取后可在结果页核对数值范围；如需更高精度，可填写更密的 Y 轴刻度。
          </Paragraph>
        </div>
      )}

      {/* ---------------- Step 3: results ---------------- */}
      {step === "results" && (
        <div>
          {imageFile && result && (
            <>
              <Space style={{ marginBottom: 12 }}>
                <Button size="small" onClick={() => setStep("configure")}>← 返回调整曲线</Button>
                <Text type="secondary" style={{ fontSize: 13 }}>{imageFile.name}</Text>
              </Space>
              <ResultReview
                result={result}
                imageSrc={imageUrl}
                baseName={imageFile.name.replace(/\.[^.]+$/, "")}
                onApplyToWorkspace={(text, confirmed) => {
                  onApplyToWorkspace(text, confirmed);
                  onClose();
                }}
              />
            </>
          )}

          {pdfName && (
            <div>
              <Space style={{ marginBottom: 12 }} wrap>
                <Button size="small" onClick={() => setStep("upload")}>← 重新选择</Button>
                <Text type="secondary" style={{ fontSize: 13 }}>{pdfName}</Text>
              </Space>

              {pdfJob && pdfJob.status !== "completed" && pdfJob.status !== "failed" && (
                <div style={{ padding: "24px 0", textAlign: "center" }}>
                  <Spin />
                  <Progress
                    percent={Math.round((pdfJob.progress ?? 0) * 100)}
                    size="small"
                    style={{ maxWidth: 320, margin: "12px auto 0" }}
                  />
                  <div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginTop: 6 }}>
                    正在处理（{pdfJob.status}）…
                  </div>
                </div>
              )}

              {pdfJob?.status === "failed" && (
                <Alert type="error" showIcon message={pdfJob.error ?? "提取失败"} />
              )}

              {pdfPreview && pdfPreview.results.length > 0 && (
                <div>
                  <Paragraph type="secondary" style={{ fontSize: 13 }}>
                    共检测到 {pdfPreview.regions.length} 个图表区域，成功追踪如下。点击展开查看与导出。
                  </Paragraph>
                  {pdfPreview.results.map((res, index) => {
                    const region = pdfPreview.regions[index];
                    const nCurves = res.curves.length;
                    const isOpen = expandedPdfResult === index;
                    return (
                      <div
                        key={index}
                        style={{
                          border: "1px solid var(--serif-border)",
                          borderRadius: 6,
                          marginBottom: 10,
                          background: "var(--serif-card)",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 10,
                            padding: "10px 14px",
                            cursor: "pointer",
                          }}
                          onClick={() => setExpandedPdfResult(isOpen ? null : index)}
                        >
                          <Text strong>
                            第 {(region?.page_index ?? 0) + 1} 页 · 区域 {(region?.region_index ?? 0) + 1}
                          </Text>
                          <Tag>{nCurves} 条曲线</Tag>
                          <Tag color={res.error ? "warning" : "success"}>
                            {res.error ? "部分失败" : `置信 ${(res.confidence * 100).toFixed(0)}%`}
                          </Tag>
                          <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--serif-muted-foreground)" }}>
                            {isOpen ? "收起 ▴" : "展开 ▾"}
                          </span>
                        </div>
                        {isOpen && region && (
                          <div style={{ padding: "0 14px 14px" }}>
                            <ResultReview
                              result={res}
                              imageSrc={extractRegionImageUrl(
                                pdfPreview.job_id,
                                region.page_index,
                                region.region_index,
                              )}
                              baseName={`${pdfName.replace(/\.[^.]+$/, "")}_p${region.page_index + 1}r${region.region_index + 1}`}
                              onApplyToWorkspace={(text, confirmed) => {
                                onApplyToWorkspace(text, confirmed);
                                onClose();
                              }}
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {pdfPreview && pdfPreview.results.length === 0 && pdfJob?.status === "completed" && (
                <Empty description="未检测到可追踪的图表曲线" />
              )}
            </div>
          )}

          {!imageFile && !pdfName && (
            <Empty description="暂无提取结果" />
          )}
        </div>
      )}
    </Drawer>
  );
}
