import type { MouseEventHandler } from "react";
import { ANCHOR_LINE_COLORS } from "../StepSection";
import CandidateCurveOverlay, { type CandidateCurvePoint } from "./CandidateCurveOverlay";

export interface ChartAnchors {
  startXRatio?: number;
  endXRatio?: number;
  topYRatio?: number;
  bottomYRatio?: number;
}

/** One source-of-truth canvas for the original chart, anchors and CV candidate. */
export default function ChartImageCanvas({
  src,
  alt,
  anchors,
  candidateCurve,
  editable,
  isPicking,
  onImageClick,
  onImageDoubleClick,
  onPointMove,
  fillWidth = false,
}: {
  src: string;
  alt: string;
  anchors: ChartAnchors;
  candidateCurve: CandidateCurvePoint[];
  editable: boolean;
  isPicking: boolean;
  onImageClick: MouseEventHandler<HTMLImageElement>;
  onImageDoubleClick?: MouseEventHandler<HTMLImageElement>;
  onPointMove: (index: number, yRatio: number) => void;
  fillWidth?: boolean;
}): React.JSX.Element {
  const crosshair = editable || isPicking;
  return (
    <div style={{ position: "relative", width: fillWidth ? "100%" : "fit-content", maxWidth: "100%" }}>
      <img
        src={src}
        alt={alt}
        draggable={false}
        onClick={onImageClick}
        onDoubleClick={onImageDoubleClick}
        style={{
          display: "block",
          width: fillWidth ? "100%" : "auto",
          height: "auto",
          maxWidth: "100%",
          maxHeight: fillWidth ? undefined : "70vh",
          cursor: crosshair ? "crosshair" : onImageDoubleClick ? "zoom-in" : "default",
        }}
      />
      {anchors.startXRatio !== undefined && <span style={{ position: "absolute", top: 0, bottom: 0, left: `${anchors.startXRatio * 100}%`, borderLeft: `2px solid ${ANCHOR_LINE_COLORS.start}`, pointerEvents: "none" }} />}
      {anchors.endXRatio !== undefined && <span style={{ position: "absolute", top: 0, bottom: 0, left: `${anchors.endXRatio * 100}%`, borderLeft: `2px solid ${ANCHOR_LINE_COLORS.end}`, pointerEvents: "none" }} />}
      {anchors.topYRatio !== undefined && <span style={{ position: "absolute", left: 0, right: 0, top: `${anchors.topYRatio * 100}%`, borderTop: `2px solid ${ANCHOR_LINE_COLORS.top}`, pointerEvents: "none" }} />}
      {anchors.bottomYRatio !== undefined && <span style={{ position: "absolute", left: 0, right: 0, top: `${anchors.bottomYRatio * 100}%`, borderTop: `2px solid ${ANCHOR_LINE_COLORS.bottom}`, pointerEvents: "none" }} />}
      <CandidateCurveOverlay points={candidateCurve} editable={editable} anchorOnly onPointMove={onPointMove} />
    </div>
  );
}
