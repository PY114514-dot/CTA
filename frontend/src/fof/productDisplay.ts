import type { KbProduct } from "../api";

const DEMO_PRODUCT_NAMES = new Set([
  "cta alpha",
  "cta beta",
  "cta gamma",
  "cta steady no.1",
  "cta steady no 1",
  "cta growth",
]);

const STRATEGY_LABELS: Record<string, string> = {
  commodity_cta: "商品 CTA",
  trend_cta: "趋势 CTA",
  arbitrage_cta: "套利 CTA",
  mixed_cta: "复合 CTA",
  equity_quant: "股票/股指方向（待确认）",
  market_neutral: "市场中性",
};

/** Historical examples are intentionally hidden from the working product pool. */
export function isDemoProduct(product: KbProduct): boolean {
  const name = product.standard_name.trim().toLowerCase();
  const manager = product.manager_name?.trim().toLowerCase();
  return DEMO_PRODUCT_NAMES.has(name) || manager === "alphacap" || manager === "testmanager";
}

/** A product without a retained material cannot be inspected or audited. */
export function hasTraceableSource(product: KbProduct): boolean {
  return Boolean(product.source_file_ids?.length || product.source_files.length);
}

/** Keep historical fixture files out of the formal workbench as well. */
export function isDemoFileName(filename: string): boolean {
  const value = filename.trim().toLowerCase();
  return /(?:cta[ _-]?(?:alpha|beta|gamma|growth)|steady[ _-]?no|testmanager|fixture|sample)/.test(value);
}

/** A code plus a pasted OCR row is not a safe product identity. */
export function needsManualNaming(product: KbProduct): boolean {
  const nameIsCode = /^[A-Z]{2,8}\d{3,}$/.test(product.standard_name.trim());
  const strategy = product.strategy?.trim() ?? "";
  return nameIsCode && (strategy.length > 28 || /["“”]|20\d{2}[/-]\d{1,2}[/-]\d{1,2}/.test(strategy));
}

/** Internal identifiers and untrusted OCR prose never leak into the compact UI. */
export function strategyLabel(strategy: string | null): string | null {
  const value = strategy?.trim();
  if (!value) return null;
  const normalized = value.toLowerCase().replace(/[\s-]+/g, "_");
  if (STRATEGY_LABELS[normalized]) return STRATEGY_LABELS[normalized];
  return value.length > 24 ? "待确认" : value;
}
