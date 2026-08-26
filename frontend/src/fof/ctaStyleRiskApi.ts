/** CTA style-risk contract, isolated from the general-purpose HTTP client. */

import { API_BASE_URL, errorMessage, type ConfirmationStatus, type DataFrequency } from "../api";

export interface CtaStyleRiskResponse {
  product_id: string;
  product_name: string;
  frequency: DataFrequency;
  data_contract: { confirmation_status: ConfirmationStatus; reviewed_nav_count: number; source: string; parsing_triggered: false };
  alignment: { factor_names: string[]; aligned_observation_count: number; start_date: string; end_date: string };
  method: "return_based_style_risk_v0";
  factor_exposures: Array<{ factor_name: string; beta: number; risk_contribution: number | null }>;
  risk: { total_variance: number; systematic_variance: number; idiosyncratic_variance: number; idiosyncratic_risk_share: number | null; factor_covariance: Record<string, Record<string, number>> };
  diagnostics: { condition_number: number };
  style_drift: { window: number; causal: true; beta_range: Record<string, number | null>; paths: Array<{ date: string; betas: Record<string, number> }> };
  warnings: string[];
}

/** Return-based CTA style risk; never represents holdings or positions. */
export async function getCtaStyleRisk(productId: string): Promise<CtaStyleRiskResponse> {
  const response = await fetch(`${API_BASE_URL}/cta-style-risk/products/${productId}`);
  if (!response.ok) throw new Error(await errorMessage(response, "CTA 风格风险评价失败"));
  return response.json() as Promise<CtaStyleRiskResponse>;
}
