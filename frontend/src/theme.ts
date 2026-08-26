import type { ThemeConfig } from "antd";

/**
 * Research dashboard design tokens (default professional-blue palette).
 *
 * Kept as a static reference of the default palette; runtime theming is
 * driven by AccentPreset values and the --serif-* CSS variables.
 */
export const SERIF_TOKENS = {
  background: "#F8FAFC",
  foreground: "#1E293B",
  muted: "#EFF4FA",
  mutedForeground: "#64748B",
  accent: "#2563EB",
  accentSecondary: "#3B82F6",
  border: "#E2E8F0",
  borderHover: "#CBD5E1",
  card: "#FFFFFF",
} as const;

// Product UI uses one sans-serif family throughout.  The old combination of
// Playfair + Songti for headings and YaHei for controls made Chinese glyphs
// visibly darker and shifted their baseline between cards, drawers and forms.
export const FONT_DISPLAY = `"Source Sans 3", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", system-ui, sans-serif`;
export const FONT_BODY = `"Source Sans 3", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", system-ui, sans-serif`;
export const FONT_MONO = `"IBM Plex Mono", "PingFang SC", "Microsoft YaHei", "SFMono-Regular", Consolas, monospace`;

// ---------------------------------------------------------------------------
// Accent presets — complete palettes exposed in the settings drawer.
// Switching a preset retints the whole surface system (canvas, cards, rules,
// text, masthead), not just the primary button.
// ---------------------------------------------------------------------------

export interface AccentPreset {
  key: string;
  label: string;
  /** Bright surface color for solid buttons and decoration. */
  accent: string;
  /** Deeper companion readable as text on white surfaces. */
  deep: string;
  /** Page canvas background. */
  background: string;
  /** Card / panel surface. */
  card: string;
  /** Hairline borders. */
  border: string;
  /** Muted fills (table headers, idle buttons, tags). */
  muted: string;
  /** Secondary text. */
  mutedForeground: string;
  /** Primary text and masthead surface. */
  foreground: string;
}

export const ACCENT_PRESETS: AccentPreset[] = [
  {
    key: "blue",
    label: "专业蓝",
    accent: "#2563EB",
    deep: "#1D4ED8",
    background: "#F8FAFC",
    card: "#FFFFFF",
    border: "#E2E8F0",
    muted: "#EFF4FA",
    mutedForeground: "#64748B",
    foreground: "#1E293B",
  },
  {
    key: "pine",
    label: "松绿",
    accent: "#4C7A5C",
    deep: "#3B6149",
    background: "#F5F8F5",
    card: "#FFFFFF",
    border: "#DCE5DC",
    muted: "#EAF1EA",
    mutedForeground: "#5D6B5D",
    foreground: "#1C261C",
  },
  {
    key: "claret",
    label: "绛红",
    accent: "#A8503F",
    deep: "#8A3E30",
    background: "#FAF6F5",
    card: "#FFFFFF",
    border: "#EADFD9",
    muted: "#F3EAE6",
    mutedForeground: "#75635D",
    foreground: "#2A1D19",
  },
];

export const ACCENT_STORAGE_KEY = "cta_research_accent";

export function getPreset(key: string): AccentPreset {
  return ACCENT_PRESETS.find((preset) => preset.key === key) ?? ACCENT_PRESETS[0]!;
}

export function loadAccentKey(): string {
  try {
    return localStorage.getItem(ACCENT_STORAGE_KEY) ?? ACCENT_PRESETS[0]!.key;
  } catch {
    return ACCENT_PRESETS[0]!.key;
  }
}

/** Approximate relative luminance; used to pick legible text on solid accents. */
function isLightColor(hex: string): boolean {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55;
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Sync every --serif-* CSS variable consumed by serif.css and inline styles
 * with the active preset, so the whole surface system retints at once.
 */
export function applyAccentCssVariables(preset: AccentPreset): void {
  const variables: Record<string, string> = {
    "--serif-background": preset.background,
    "--serif-foreground": preset.foreground,
    "--serif-muted": preset.muted,
    "--serif-muted-foreground": preset.mutedForeground,
    "--serif-accent": preset.deep,
    "--serif-accent-secondary": preset.accent,
    "--serif-border": preset.border,
    "--serif-card": preset.card,
    "--serif-on-accent": isLightColor(preset.accent) ? "#1A1A1A" : "#FFFFFF",
  };
  const root = document.documentElement;
  for (const [name, value] of Object.entries(variables)) {
    root.style.setProperty(name, value);
  }
}

/**
 * Ant Design v5 theme built from one complete accent preset. Bright accents
 * Bright accents get near-black button text; dark accents
 * get white text.
 */
export function buildSerifTheme(preset: AccentPreset): ThemeConfig {
  const onAccent = isLightColor(preset.accent) ? "#1A1A1A" : "#FFFFFF";
  return {
    token: {
      // Brand & accent
      colorPrimary: preset.accent,
      colorInfo: preset.accent,
      colorLink: preset.deep,
      colorLinkHover: preset.accent,
      colorTextLightSolid: onAccent,

      // Keep semantic colors but soften them toward the warm palette
      colorSuccess: "#4C7A5C",
      colorWarning: preset.deep,
      colorError: "#A8503F",

      // Surfaces
      colorBgLayout: preset.background,
      colorBgContainer: preset.card,
      colorBgElevated: preset.card,
      colorBgSpotlight: preset.card,

      // Text
      colorText: preset.foreground,
      colorTextSecondary: preset.mutedForeground,
      colorTextTertiary: preset.mutedForeground,
      colorTextQuaternary: preset.mutedForeground,

      // Borders (the rule-line system)
      colorBorder: preset.border,
      colorBorderSecondary: preset.background,

      // Fills
      colorFill: preset.muted,
      colorFillSecondary: preset.muted,
      colorFillTertiary: preset.background,
      colorFillQuaternary: preset.card,

      // Radii — refined, not too round
      borderRadius: 6,
      borderRadiusLG: 8,
      borderRadiusSM: 4,
      borderRadiusXS: 2,

      // Typography
      fontFamily: FONT_BODY,
      fontSize: 15,
      lineHeight: 1.7,

      // Controls — comfortable, accessible height
      controlHeight: 40,
      controlHeightLG: 48,
      controlHeightSM: 32,

      // Shadows — very subtle, warm-tinted
      boxShadow: "0 1px 2px rgba(26, 26, 26, 0.04)",
      boxShadowSecondary: "0 4px 12px rgba(26, 26, 26, 0.06)",
      boxShadowTertiary: "0 8px 24px rgba(26, 26, 26, 0.08)",

      // Motion — restrained
      motionDurationMid: "0.2s",
      motionEaseInOut: "cubic-bezier(0.25, 0.1, 0.25, 1)",
    },
    components: {
      Layout: {
        headerBg: preset.foreground,
        bodyBg: preset.background,
        headerHeight: 72,
        headerPadding: "0 48px",
      },
      Card: {
        colorBgContainer: preset.card,
        borderRadiusLG: 8,
        paddingLG: 28,
        headerBg: "transparent",
        boxShadowTertiary: "0 1px 2px rgba(26, 26, 26, 0.04)",
      },
      Button: {
        borderRadius: 6,
        controlHeight: 40,
        fontWeight: 400,
        primaryShadow: `0 1px 2px ${hexToRgba(preset.accent, 0.28)}`,
        defaultBorderColor: preset.foreground,
        defaultColor: preset.foreground,
      },
      Input: {
        borderRadius: 6,
        controlHeight: 40,
        activeBorderColor: preset.accent,
        activeShadow: `0 0 0 2px ${hexToRgba(preset.accent, 0.14)}`,
      },
      InputNumber: {
        borderRadius: 6,
        controlHeight: 40,
        handleBorderColor: preset.border,
      },
      Select: {
        borderRadius: 6,
        controlHeight: 40,
      },
      Table: {
        headerBg: preset.muted,
        headerColor: preset.mutedForeground,
        borderColor: preset.border,
        rowHoverBg: preset.background,
      },
      Statistic: {
        contentFontSize: 26,
      },
      Tag: {
        borderRadiusSM: 4,
      },
      Divider: {
        colorSplit: preset.border,
      },
      Progress: {
        defaultColor: preset.accent,
      },
      Steps: {
        colorPrimary: preset.accent,
      },
      Alert: {
        borderRadiusLG: 8,
      },
      Modal: {
        borderRadiusLG: 8,
      },
      Drawer: {
        colorBgElevated: preset.card,
      },
    },
  };
}

/** Default theme instance (professional blue). */
export const serifTheme: ThemeConfig = buildSerifTheme(ACCENT_PRESETS[0]!);
