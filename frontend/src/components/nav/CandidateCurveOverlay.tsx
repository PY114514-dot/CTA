import { useState } from "react";

export interface CandidateCurvePoint {
  x: number;
  y: number;
}

/** Displays the CV candidate over the source chart and optionally exposes three edit anchors. */
export default function CandidateCurveOverlay({
  points,
  editable = false,
  anchorOnly = false,
  onPointMove,
}: {
  points: CandidateCurvePoint[];
  editable?: boolean;
  anchorOnly?: boolean;
  onPointMove?: (index: number, yRatio: number) => void;
}): React.JSX.Element | null {
  const [draggingIndex, setDraggingIndex] = useState<number>();
  if (points.length < 2) return null;
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${(point.x * 1000).toFixed(1)},${(point.y * 1000).toFixed(1)}`).join(" ");

  function pointY(event: React.PointerEvent<SVGSVGElement>): number {
    const bounds = event.currentTarget.getBoundingClientRect();
    return Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
  }

  return (
    <svg
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
      aria-label="当前候选净值曲线叠加层"
      onPointerMove={(event) => {
        if (draggingIndex !== undefined) onPointMove?.(draggingIndex, pointY(event));
      }}
      onPointerUp={(event) => {
        if (draggingIndex !== undefined) onPointMove?.(draggingIndex, pointY(event));
        setDraggingIndex(undefined);
      }}
      onPointerCancel={() => setDraggingIndex(undefined)}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: editable ? "auto" : "none", overflow: "visible", touchAction: "none" }}
    >
      <path d={path} fill="none" stroke="#D97706" strokeWidth="5" strokeDasharray="12 8" vectorEffect="non-scaling-stroke" opacity="0.9" />
      {editable && points.map((point, index) => {
        const mid = Math.floor((points.length - 1) / 2);
        if (anchorOnly && index !== 0 && index !== mid && index !== points.length - 1) return null;
        return (
          <circle
            key={`${index}-${point.x.toFixed(4)}`}
            cx={point.x * 1000}
            cy={point.y * 1000}
            r="9"
            fill="#FFF8E1"
            stroke="#D97706"
            strokeWidth="4"
            vectorEffect="non-scaling-stroke"
            aria-label={anchorOnly ? `调整${index === 0 ? "起点" : index === mid ? "中段" : "终点"}锚点` : `调整第 ${index + 1} 个候选净值点`}
            onPointerDown={(event) => {
              event.preventDefault();
              event.stopPropagation();
              event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
              setDraggingIndex(index);
            }}
          />
        );
      })}
    </svg>
  );
}
