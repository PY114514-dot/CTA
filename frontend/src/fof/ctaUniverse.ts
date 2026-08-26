/** Shared CTA universe builder for ranking and scoring panels.
 *
 * Both the CODEX ranking panel and the product score panel need the same
 * input universe: research-ready products with reviewed NAV points, reduced
 * to one common frequency.  This module keeps that definition in one place
 * so the two read-only panels cannot drift apart.
 */

import { kbGetNavSeriesBulk, kbListProducts, type DataFrequency, type NavPoint } from "../api";
import { isResearchReady } from "./productWorkflow";

export interface CtaUniverseProductInput {
  product_id: string;
  product_name: string;
  nav_points: NavPoint[];
  frequency: DataFrequency;
  strategy?: string | null;
}

export interface CtaUniverse {
  products: CtaUniverseProductInput[];
  frequency: DataFrequency;
  asOfDate?: string;
  excludedCount: number;
  totalCount: number;
}

function normalizeFrequency(value: string | null): DataFrequency {
  return value === "daily" || value === "monthly" ? value : "weekly";
}

function isConfirmedCta(strategy: string | null): boolean {
  const normalized = (strategy ?? "").trim().toLowerCase().replace(/[- ]/g, "_");
  const englishCodes = [
    "commodity_cta", "trend_cta", "arbitrage_cta", "mixed_cta",
    "commodity_arbitrage", "commodity_spread_arbitrage",
  ];
  // 产品库当前实际使用的中文策略标签，与英文代码一一对应。
  const chineseLabels = ["量化期货", "主观期货", "商品套利"];
  return englishCodes.includes(normalized) || chineseLabels.includes((strategy ?? "").trim());
}

function commonAsOfDate(products: Array<{ nav_end: string | null }>): string | undefined {
  const dates = products.map((product) => product.nav_end).filter((value): value is string => Boolean(value)).sort();
  return dates[0];
}

export async function buildCtaUniverse(selectedProductIds: string[], purpose: string): Promise<CtaUniverse> {
  const products = await kbListProducts({ limit: 1000 });
  const selected = selectedProductIds.length
    ? products.filter((product) => selectedProductIds.includes(product.id))
    : products;
  const candidates = selected.filter((product) => (
    isResearchReady(product)
    && isConfirmedCta(product.strategy)
    && (product.reviewed_nav_count ?? 0) >= 2
  ));
  if (candidates.length === 0) {
    throw new Error(`没有策略已确认为 CTA、可研究且已复核净值的产品可${purpose}`);
  }

  const enriched = await (async () => {
    // 一次批量请求替代逐只请求：749 只产品逐个发请求会被浏览器连接池
    //（每域名约 6 并发）串行化，批量接口单次查询即可取回全部净值。
    const navByProduct = await kbGetNavSeriesBulk(candidates.map((product) => product.id), true);
    return candidates.map((product) => {
      const nav = navByProduct[product.id] ?? [];
      return {
        product,
        frequency: normalizeFrequency(product.nav_frequency),
        navPoints: nav
          .filter((point) => point.review_status === "reviewed" && Number.isFinite(point.nav) && point.nav > 0)
          .map((point) => ({
            observation_date: point.observation_date,
            net_asset_value: point.nav,
          })),
      };
    });
  })();
  const frequencyCounts = new Map<DataFrequency, number>();
  for (const item of enriched) {
    frequencyCounts.set(item.frequency, (frequencyCounts.get(item.frequency) ?? 0) + 1);
  }
  const frequency = [...frequencyCounts.entries()]
    .sort((left, right) => right[1] - left[1])[0]?.[0] ?? "weekly";
  const compatible = enriched.filter((item) => item.frequency === frequency && item.navPoints.length >= 2);
  if (compatible.length === 0) {
    throw new Error(`没有同频率的已复核净值序列可${purpose}`);
  }
  return {
    products: compatible.map((item) => ({
      product_id: item.product.id,
      product_name: item.product.standard_name,
      nav_points: item.navPoints,
      frequency: item.frequency,
      strategy: item.product.strategy,
    })),
    frequency,
    asOfDate: commonAsOfDate(compatible.map((item) => item.product)),
    excludedCount: enriched.length - compatible.length,
    totalCount: compatible.length,
  };
}
