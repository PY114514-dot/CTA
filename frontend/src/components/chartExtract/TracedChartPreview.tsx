import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ExtractTracedCurve } from "../../api";
import { FONT_MONO } from "../../theme";

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
export function TracedChartPreview({ src, curves, picking, onPick, maxHeight = 420, curveOffset = [0, 0], overlayAdjustment = [0, 0], adjustingOverlay = false, onOverlayAdjustmentChange, overlayColor, overlayDash }: TracedChartPreviewProps): React.JSX.Element {
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
