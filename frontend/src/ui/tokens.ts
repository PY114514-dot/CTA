/**
 * Shared UI theming tokens and helpers.
 *
 * ECharts renders to a <canvas>, which cannot resolve CSS custom properties,
 * so theme colors must be read from computed style at render time instead.
 */

/** Fallback for the --serif-accent variable (matches the default gold preset). */
export const ACCENT_FALLBACK = "#B8860B";

/** Read a --serif-* CSS variable at render time (ECharts canvas cannot resolve CSS vars). */
export function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}
