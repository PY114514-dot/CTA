/**
 * Shared editorial section primitives used across drawers and panels.
 */
import React from "react";
import { FONT_DISPLAY } from "../theme";

/** Serif section heading with editorial spacing. */
export function SectionTitle({ children, first }: { children: React.ReactNode; first?: boolean }): React.JSX.Element {
  return (
    <div
      style={{
        fontFamily: FONT_DISPLAY,
        fontSize: 15,
        fontWeight: 600,
        color: "var(--serif-foreground)",
        margin: first ? "0 0 12px" : "26px 0 12px",
      }}
    >
      {children}
    </div>
  );
}

/** Small muted label above a form field. */
export function FieldLabel({ children }: { children: React.ReactNode }): React.JSX.Element {
  return <div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginBottom: 4 }}>{children}</div>;
}
