/**
 * Shared numeric display helpers for tables and metric cells.
 */
import React from "react";

/** Format a ratio as a one-decimal percentage string. */
export const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;

/** Render a formatted number in tabular figures, tinted red when negative. */
export function SignedValue({ value, format }: { value: number; format: (v: number) => string }): React.JSX.Element {
  return <span style={{ color: value < 0 ? "#cf1322" : undefined, fontVariantNumeric: "tabular-nums" }}>{format(value)}</span>;
}
