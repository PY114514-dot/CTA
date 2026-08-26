/** One UI mapping for the backend-owned research workflow. */

import type { KbFile, KbProduct } from "../api";

export type ProductWorkflowAction = { label: string; run: () => void } | null;

/** SQLite files are batch data sources; their rows are linked to many products. */
export function isBulkNavDataset(file: KbFile): boolean {
  return file.source === "futures_weekly_sqlite";
}

/** A parsed source already has one product target and can enter NAV review. */
export function isProductReviewFile(file: KbFile): boolean {
  if (!file.linked_product_ids?.length) return false;
  const hasUnreviewedNav = (file.nav_count ?? 0) > 0
    && (file.reviewed_nav_count ?? 0) < (file.nav_count ?? 0);
  return hasUnreviewedNav
    || file.workflow?.next_action === "review_candidate"
    || file.workflow?.next_action === "recover_chart_calibration"
    || file.parsing_status === "completed_no_nav";
}

/** A source still needs a human product-identity decision before review. */
export function needsProductBinding(file: KbFile): boolean {
  return !isBulkNavDataset(file) && !file.linked_product_ids?.length;
}

/** Prefer the current upload batch while preserving the server's queue order. */
export function findNextProductReviewFile(
  files: KbFile[],
  excludedIds: ReadonlySet<string> = new Set<string>(),
  preferredIds: readonly string[] = [],
): KbFile | undefined {
  for (const fileId of preferredIds) {
    const file = files.find((candidate) => candidate.id === fileId);
    if (file && !excludedIds.has(file.id) && isProductReviewFile(file)) return file;
  }
  return files.find((file) => !excludedIds.has(file.id) && isProductReviewFile(file));
}

function workflow(product: KbProduct): NonNullable<KbProduct["research_workflow"]> {
  if (product.research_workflow) return product.research_workflow;
  if (product.confirmation_status === "rejected") {
    return { stage: "rejected", label: "已标记为非产品", next_action: null, blocking_reasons: ["该记录不会进入研究上下文"], completed_steps: {} };
  }
  if (product.nav_count < 2) {
    return { stage: "needs_nav", label: "待补净值", next_action: "calibrate_nav", blocking_reasons: ["净值点不足 2 条"], completed_steps: {} };
  }
  return { stage: "needs_review", label: "待复核净值", next_action: "calibrate_nav", blocking_reasons: ["请先完成产品身份与净值复核"], completed_steps: {} };
}

export function isResearchReady(product: KbProduct): boolean {
  return workflow(product).stage === "research_ready";
}

export function productWorkflowStatus(product: KbProduct): { color: string; label: string } {
  const current = workflow(product);
  if (current.stage === "research_ready") return { color: "success", label: current.label };
  if (current.stage === "needs_quality_review" || current.stage === "rejected") {
    return { color: "error", label: current.label };
  }
  return { color: "processing", label: current.label };
}

export function productWorkflowReason(product: KbProduct): string {
  return workflow(product).blocking_reasons[0] ?? "请先完成产品身份与净值复核";
}

export function productWorkflowAction(
  product: KbProduct,
  actions: { editIdentity: () => void; calibrateNav: () => void; confirmIdentity: () => void },
): ProductWorkflowAction {
  switch (workflow(product).next_action) {
    case "edit_identity": return { label: "完善产品信息", run: actions.editIdentity };
    case "calibrate_nav": return { label: product.nav_count < 2 ? "提取 / 补录净值" : "复核净值", run: actions.calibrateNav };
    case "confirm_identity": return { label: "确认产品", run: actions.confirmIdentity };
    default: return null;
  }
}
