/**
 * Step progression and anchor marking UI components for image digitization.
 *
 * Extracted from main.tsx for maintainability.
 */
import React from "react";
import { FONT_DISPLAY, FONT_MONO } from "../theme";

/** Distinct calibration line colors, one per anchor. */
export const ANCHOR_LINE_COLORS: Record<"start" | "end" | "top" | "bottom", string> = {
  start: "#2563EB",
  end: "#A8503F",
  top: "#4C7A5C",
  bottom: "#4A6B8A",
};

export const ANCHOR_LABELS: Record<"start" | "end" | "top" | "bottom", string> = {
  start: "首日期",
  end: "末日期",
  top: "最大值",
  bottom: "最小值",
};

/** Numbered step header with serif title — the editorial progression pattern. */
export function StepSection({
  step,
  title,
  children,
  visible = true,
}: {
  step: number;
  title: string;
  children: React.ReactNode;
  visible?: boolean;
}): React.JSX.Element {
  if (!visible) return <></>;
  return (
    <div style={{ marginBottom: 26 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span
          style={{
            minWidth: 24,
            height: 24,
            borderRadius: "50%",
            background: "var(--serif-accent-secondary)",
            color: "var(--serif-on-accent)",
            fontSize: 12,
            fontWeight: 600,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: FONT_MONO,
          }}
        >
          {step}
        </span>
        <span style={{ fontFamily: FONT_DISPLAY, fontSize: 16, fontWeight: 600, color: "var(--serif-foreground)" }}>
          {title}
        </span>
      </div>
      <div style={{ paddingLeft: 34 }}>{children}</div>
    </div>
  );
}

/**
 * Square marking button: muted by default, blue when armed, tinted border
 * with a check once the anchor is placed.
 */
export function AnchorMarkButton({
  anchor,
  active,
  completed,
  disabled,
  onClick,
}: {
  anchor: "start" | "end" | "top" | "bottom";
  active: boolean;
  completed: boolean;
  disabled: boolean;
  onClick: () => void;
}): React.JSX.Element {
  const lineColor = ANCHOR_LINE_COLORS[anchor];
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      style={{
        flex: 1,
        padding: "10px 4px 8px",
        border: active
          ? "1.5px solid var(--serif-accent-secondary)"
          : completed
            ? `1.5px solid ${lineColor}`
            : "1px solid var(--serif-border)",
        borderRadius: 6,
        background: active ? "var(--serif-accent-secondary)" : completed ? "var(--serif-card)" : "var(--serif-muted)",
        color: active ? "var(--serif-on-accent)" : "var(--serif-foreground)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.45 : 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 5,
        transition: "background 0.2s, border-color 0.2s",
      }}
    >
      <span style={{ fontSize: 13, fontWeight: 500, whiteSpace: "nowrap" }}>
        {completed && !active ? "✓ " : ""}
        {ANCHOR_LABELS[anchor]}
      </span>
      <span style={{ width: 18, height: 3, borderRadius: 2, background: active ? "var(--serif-on-accent)" : lineColor }} />
    </button>
  );
}
