/**
 * Markdown report generation from history records.
 *
 * Extracted from main.tsx for testability and reuse.
 */
import type { CtaAttributionSnapshotResponse, ProductStrategyProfileResponse, ReportGenerateResponse } from "../api";
import type { HistoryRecord } from "../storage/historyStorage";

/** Human labels for AI-classified strategy types (mirrors AnalysisPanel). */
export const STRATEGY_TYPE_LABELS: Record<string, string> = {
  commodity_cta: "商品 CTA",
  equity_quant: "股票/股指方向（CTA或量化，待确认）",
  mixed: "混合型",
  insufficient_data: "数据不足",
};

/** Append the keyword-evidence strategy profile section (from OCR text). */
export function appendStrategyProfileSection(
  lines: string[],
  profile: ProductStrategyProfileResponse | undefined,
): void {
  if (!profile) return;
  lines.push("### 产品画像（OCR 文本推断）", "");
  lines.push(`**策略假设：** ${profile.strategy_hypothesis.label}`, "");
  if (profile.strategy_hypothesis.evidence.length) {
    lines.push(`**判断依据：** ${profile.strategy_hypothesis.evidence.join("；")}`, "");
  }
  if (profile.factor_hypotheses.length) {
    lines.push("**因子线索：**", "", "| 因子 | 依据 |", "| --- | --- |");
    for (const item of profile.factor_hypotheses) {
      lines.push(`| ${item.label} | ${item.evidence.join("；") || "—"} |`);
    }
    lines.push("");
  }
  if (profile.futures_categories.length) {
    lines.push("**期货品种线索：**", "", "| 品种 | 依据 |", "| --- | --- |");
    for (const item of profile.futures_categories) {
      lines.push(`| ${item.label} | ${item.evidence.join("；") || "—"} |`);
    }
    lines.push("");
  }
  if (profile.disclaimer) lines.push(profile.disclaimer, "");
}

/** Append the AI attribution analysis section. */
export function appendAiAnalysisSection(lines: string[], ai: ReportGenerateResponse | undefined): void {
  if (!ai) return;
  lines.push("### AI 策略归因分析", "");
  lines.push(`**分析引擎：** ${ai.engine}`, `**生成时间：** ${ai.generated_at}`, "");
  if (ai.strategy_summary) lines.push("**分析摘要：**", "", ai.strategy_summary, "");
  const classification = ai.structured?.classification;
  if (classification) {
    const typeLabel = STRATEGY_TYPE_LABELS[classification.strategy_type] ?? classification.strategy_type;
    lines.push(
      `**策略分类：** ${typeLabel}（置信度 ${classification.confidence_pct.toFixed(0)}% · ${classification.confidence_label}）`,
      "",
    );
    if (classification.evidence.length) {
      lines.push(`**分类依据：** ${classification.evidence.join("；")}`, "");
    }
  }
  const disclosure = classification?.details?.disclosure_hint as {
    label?: string;
    status?: string;
    conflict?: boolean;
  } | undefined;
  if (disclosure?.label) {
    const status = disclosure.conflict ? "与净值统计冲突" : disclosure.status === "aligned" ? "与统计方向一致" : "披露候选";
    lines.push(
      `**管理人披露：** ${disclosure.label}（${status}）`,
      "披露文本仅作为管理人声明候选，不等同于实际持仓或统计确认。",
      "",
    );
  }
  const confirmation = classification?.details?.user_confirmation as {
    confirmed?: boolean;
    requested_type?: string;
    statistical_conflict?: boolean;
    disclosure_conflict?: boolean;
  } | undefined;
  if (confirmation?.confirmed) {
    const labels: Record<string, string> = {
      commodity_cta: "商品 CTA",
      equity_cta: "股指 CTA",
      mixed: "混合 / 多资产",
    };
    lines.push(
      `**用户确认研究范围：** ${labels[confirmation.requested_type ?? ""] ?? confirmation.requested_type ?? "未指定"}`,
      confirmation.statistical_conflict || confirmation.disclosure_conflict ? "用户确认范围与统计分类或披露候选存在差异，因子结论按确认范围作条件性分析。" : "因子结论按用户确认范围作条件性分析。",
      "",
    );
  }
  const externalFactors = ai.structured?.external_factors;
  if (externalFactors?.available) {
    lines.push(
      `**外部因子参考：** 截至 ${externalFactors.latest_as_of_date ?? "未知"}（${externalFactors.verified ? "已核验" : "待核验"}）`,
      `覆盖 ${externalFactors.factor_count ?? 0} 个因子、${externalFactors.observation_count ?? 0} 条观测；${externalFactors.usable_for_regression ? "可用于回归" : "仅作披露快照参考，未直接进入回归"}。`,
      "",
    );
  } else if (externalFactors?.date_relation === "no_snapshot_on_or_before_product_end") {
    lines.push(
      `**外部因子参考：** 未使用。产品截止日为 ${externalFactors.product_end ?? "未知"}，现有外部周报快照截至 ${externalFactors.latest_available_as_of_date ?? "未知"}，晚于产品截止日。`,
      "",
    );
  }

  const factors = ai.structured?.factors;
  if (factors && factors.factors.length) {
    lines.push(
      `**风格与归因参考（历史共同解释度 R² = ${factors.r_squared.toFixed(3)}，Adj R² = ${factors.adj_r_squared.toFixed(3)}；不构成未来净值预测）：**`,
      "",
      "| 因子 | Beta | t 值 | 置信 |",
      "| --- | --- | --- | --- |",
    );
    for (const factor of factors.factors) {
      lines.push(
        `| ${factor.factor_label} | ${factor.exposure_beta.toFixed(3)} | ${factor.t_statistic.toFixed(2)} | ${factor.confidence_label} |`,
      );
    }
    lines.push("");
  }
  const varieties = ai.structured?.varieties;
  if (varieties && varieties.top_varieties.length) {
    const label = classification?.strategy_type === "equity_quant" ? "股指 CTA 市场线索" : "品种推断";
    lines.push(
      `**${label} Top ${varieties.top_varieties.length}（方法稳定性 ${(varieties.method_stability * 100).toFixed(0)}%）：**`,
      "",
      `| ${classification?.strategy_type === "equity_quant" ? "市场参考" : "品种"} | 分组 | 概率 |`,
      "| --- | --- | --- |",
    );
    for (const variety of varieties.top_varieties) {
      lines.push(`| ${variety.name} | ${variety.sector} | ${variety.probability_pct.toFixed(0)}% |`);
    }
    lines.push("");
  }
  const deep = ai.structured?.deep_attribution;
  if (deep) {
    const diagnostics = deep.diagnostics ?? {};
    const asset = deep.asset_class ?? {};
    lines.push(
      "**深度 CTA 归因（非线性/状态模型）：**",
      `资产大类候选：${asset.most_likely_label ?? "不可识别"}（候选概率 ${typeof asset.confidence_pct === "number" ? asset.confidence_pct.toFixed(1) : "—"}%）；可靠性：${diagnostics.reliability_label ?? "—"}。`,
      `有效共同期数：${diagnostics.observation_count ?? "—"}；线性样本外 R²：${typeof diagnostics.linear_oos_r2 === "number" ? diagnostics.linear_oos_r2.toFixed(3) : "—"}；非线性相对提升：${typeof diagnostics.nonlinear_uplift === "number" ? diagnostics.nonlinear_uplift.toFixed(3) : "—"}。`,
      "",
    );
    if (deep.sector_exposures?.length) {
      lines.push("**板块候选（不是实际仓位）：**", "", "| 板块 | 候选概率 | 方向 | 稳定性 |", "| --- | ---: | --- | ---: |");
      for (const sector of deep.sector_exposures) {
        lines.push(`| ${sector.label} | ${sector.candidate_probability_pct.toFixed(1)}% | ${sector.direction} | ${sector.stability_pct.toFixed(0)}% |`);
      }
      lines.push("");
    }
    if (deep.strategy_fingerprints?.length) {
      lines.push("**策略行为指纹（不是管理人规则确认）：**", "", "| 行为 | 证据分数 | 方向 | 稳定性 | 状态 |", "| --- | ---: | --- | ---: | --- |");
      for (const strategy of deep.strategy_fingerprints.slice(0, 8)) {
        const status = strategy.status === "supported" ? "支持" : strategy.status === "not_available" ? "不可识别" : "弱/不稳定";
        lines.push(`| ${strategy.label} | ${strategy.evidence_score_pct.toFixed(1)}% | ${strategy.direction} | ${strategy.stability_pct.toFixed(0)}% | ${status} |`);
      }
      lines.push("");
    }
    if (deep.state_analysis?.length) {
      lines.push("**状态条件表现：**", "", "| 状态 | 期数 | 产品平均收益 | 正收益率 |", "| --- | ---: | ---: | ---: |");
      for (const state of deep.state_analysis) {
        lines.push(`| ${state.state} | ${state.periods} | ${(state.product_mean_return * 100).toFixed(2)}% | ${(state.product_positive_rate * 100).toFixed(1)}% |`);
      }
      lines.push("");
    }
    if (deep.warnings?.length) lines.push(`**深度归因限制：** ${deep.warnings.join("；")}`, "");
  }
  if (ai.disclaimer) lines.push(ai.disclaimer, "");
}

/** Append a frozen, computed CTA evidence package without invoking an LLM. */
export function appendCtaAttributionEvidenceSection(
  lines: string[],
  snapshot: CtaAttributionSnapshotResponse | undefined,
): void {
  if (!snapshot?.evidence_package?.data_lineage) return;
  const evidence = snapshot.evidence_package;
  const lineage = evidence.data_lineage;
  lines.push("### CTA 归因证据包（冻结快照）", "");
  lines.push(
    `**快照：** ${snapshot.snapshot_id} · ${snapshot.model_version}`,
    `**净值版本：** ${snapshot.nav_fingerprint} · 因子版本：${snapshot.factor_data_version}`,
    `**样本：** ${lineage.date_range.start} 至 ${lineage.date_range.end}，已审核净值 ${lineage.reviewed_observation_count} 条；其中 ${lineage.source_linked_observation_count} 条有关联来源，${lineage.source_unlinked_observation_count} 条暂缺来源关联。`,
    "",
  );
  for (const item of evidence.claims) {
    lines.push(`**结论（可信度：${item.confidence === "medium" ? "中" : "低"}）：** ${item.claim}`, "");
    lines.push(`- 支持证据：${item.supporting_evidence.join("；")}`);
    lines.push(`- 局限/反证：${item.counter_evidence.join("；")}`, "");
  }
  if (lineage.source_files.length) {
    lines.push("**来源文件：**", "");
    for (const file of lineage.source_files) lines.push(`- ${file.filename}（版本 ${file.version}，哈希 ${file.file_hash}）`);
    lines.push("");
  }
  lines.push("**模型边界：** 本节只记录确定性计算及其来源。LLM 如参与，只能解释此证据包，不能改写净值、将统计暴露表述为真实持仓，或据此生成投资建议。", "");
}

/** Exportable standalone report for a frozen CTA attribution snapshot. */
export function generateCtaAttributionEvidenceMarkdown(snapshot: CtaAttributionSnapshotResponse): string {
  const lines = [
    `## ${snapshot.product_name} — CTA 归因证据报告`,
    "",
    `**冻结时间：** ${snapshot.created_at ?? "—"}`,
  ];
  appendCtaAttributionEvidenceSection(lines, snapshot);
  lines.push("---", "*本报告仅供内部研究复核，不构成投资建议。*", "");
  return lines.join("\n");
}

/** Generate a Markdown report from a history record. */
export function generateMarkdownReport(record: HistoryRecord, attributionSnapshot?: CtaAttributionSnapshotResponse): string {
  const lines: string[] = [
    `## ${record.product_name || "未命名产品"} — 产品分析报告`,
    "",
    `**生成时间：** ${record.saved_at}`,
    `**数据频率：** ${record.frequency === "daily" ? "日频" : record.frequency === "weekly" ? "周频" : "月频"}`,
    "",
    "### 业绩指标",
    "",
    "| 指标 | 数值 |",
    "| --- | --- |",
    `| 累计收益 | ${(record.metrics.cumulative_return * 100).toFixed(2)}% |`,
    `| 年化收益 | ${(record.metrics.annualized_return * 100).toFixed(2)}% |`,
    `| 年化波动 | ${(record.metrics.annualized_volatility * 100).toFixed(2)}% |`,
    `| 夏普比率 | ${record.metrics.sharpe_ratio !== null ? record.metrics.sharpe_ratio.toFixed(4) : "—"} |`,
    `| 最大回撤 | ${(record.metrics.maximum_drawdown * 100).toFixed(2)}% |`,
    `| 卡玛比率 | ${record.metrics.calmar_ratio !== null ? record.metrics.calmar_ratio.toFixed(4) : "—"} |`,
    "",
  ];

  appendStrategyProfileSection(lines, record.strategy_profile);
  appendAiAnalysisSection(lines, record.ai_report);
  appendCtaAttributionEvidenceSection(lines, attributionSnapshot);

  lines.push("", "---", "*本报告由私募 CTA 研究平台自动生成，仅供内部研究参考。*", "");
  return lines.join("\n");
}
