/** 产品两两对比的「相同点 / 不同点」摘要规则。
 *
 * 阈值集中定义在本文件顶部，便于统一调整；规则只做可解释、可复现的
 * 简单判定，任何数据缺失都跳过对应规则，绝不臆造结论。全部摘要文案
 * 在此生成，组件层只负责渲染，保证同一组数据在任何入口得到同一段话。
 */

import { strategyLabel } from "./productDisplay";

export const COMPARE_THRESHOLDS = {
  /** 共同收益期数下限：低于此值相关性不计算或仅作参考。 */
  minOverlap: 12,
  /** 相关性达到该值视为高度相关（相同点）。 */
  highCorrelation: 0.8,
  /** 相关性不高于该值视为低相关、具备分散价值（不同点）。 */
  lowCorrelation: 0.2,
  /** 年化收益差达到 2 个百分点记为不同点。 */
  returnGap: 0.02,
  /** 最大回撤差达到 5 个百分点记为不同点。 */
  drawdownGap: 0.05,
  /** 年化波动差达到 5 个百分点记为不同点。 */
  volatilityGap: 0.05,
} as const;

export interface CompareProductProfile {
  id: string;
  name: string;
  manager: string | null;
  strategy: string | null;
  frequency: string | null;
  annualizedReturn: number | null;
  maxDrawdown: number | null;
  annualizedVolatility: number | null;
  alphaTStat: number | null;
  qualityScore: number | null;
  confidenceScore: number | null;
  topFactorName: string | null;
  topFactorBeta: number | null;
}

export interface CompareSummary {
  similarities: string[];
  differences: string[];
  correlation: number | null;
  overlap: number;
  correlationNote: string | null;
}

const FREQUENCY_LABELS: Record<string, string> = { daily: "日频", weekly: "周频", monthly: "月频" };

export function frequencyLabel(value: string | null | undefined): string | null {
  if (!value) return null;
  return FREQUENCY_LABELS[value] ?? value;
}

function points(value: number): string {
  return `${(value * 100).toFixed(1)} 个百分点`;
}

function strategyText(profile: CompareProductProfile): string {
  return strategyLabel(profile.strategy) ?? profile.strategy ?? "未披露";
}

export function buildCompareSummary(
  left: CompareProductProfile,
  right: CompareProductProfile,
  correlation: number | null,
  overlap: number,
): CompareSummary {
  const similarities: string[] = [];
  const differences: string[] = [];

  if (left.frequency && right.frequency && left.frequency === right.frequency) {
    similarities.push(`同为${frequencyLabel(left.frequency)}净值产品`);
  }
  if (left.frequency && right.frequency && left.frequency !== right.frequency) {
    differences.push(`净值频率不同：${left.name}为${frequencyLabel(left.frequency)}，${right.name}为${frequencyLabel(right.frequency)}`);
  }
  if (left.strategy && right.strategy && left.strategy === right.strategy) {
    similarities.push(`同属${strategyText(left)}策略`);
  }
  if (left.strategy && right.strategy && left.strategy !== right.strategy) {
    differences.push(`策略不同：${left.name}为${strategyText(left)}，${right.name}为${strategyText(right)}`);
  }
  if (left.manager && right.manager && left.manager === right.manager) {
    similarities.push(`同一管理人：${left.manager}`);
  }

  if (left.annualizedReturn != null && right.annualizedReturn != null) {
    const gap = left.annualizedReturn - right.annualizedReturn;
    if (Math.abs(gap) >= COMPARE_THRESHOLDS.returnGap) {
      const higher = gap > 0 ? left : right;
      differences.push(`${higher.name}年化收益高 ${points(Math.abs(gap))}`);
    }
  }
  if (left.maxDrawdown != null && right.maxDrawdown != null) {
    const gap = left.maxDrawdown - right.maxDrawdown;
    if (Math.abs(gap) >= COMPARE_THRESHOLDS.drawdownGap) {
      const deeper = gap < 0 ? left : right;
      differences.push(`${deeper.name}最大回撤更深 ${points(Math.abs(gap))}`);
    }
  }
  if (left.annualizedVolatility != null && right.annualizedVolatility != null) {
    const gap = left.annualizedVolatility - right.annualizedVolatility;
    if (Math.abs(gap) >= COMPARE_THRESHOLDS.volatilityGap) {
      const higher = gap > 0 ? left : right;
      differences.push(`${higher.name}年化波动高 ${points(Math.abs(gap))}`);
    }
  }
  if (left.alphaTStat != null && right.alphaTStat != null && left.alphaTStat * right.alphaTStat < 0) {
    differences.push("未解释收益（α）方向相反");
  }

  if (left.topFactorName && right.topFactorName && left.topFactorBeta != null && right.topFactorBeta != null) {
    if (left.topFactorName === right.topFactorName) {
      if (left.topFactorBeta * right.topFactorBeta > 0) {
        similarities.push(`主因子暴露同向：${left.topFactorName}`);
      } else {
        differences.push(`主因子暴露方向相反：${left.topFactorName}`);
      }
    } else {
      differences.push(`主因子不同：${left.name}为${left.topFactorName}，${right.name}为${right.topFactorName}`);
    }
  }

  let correlationNote: string | null = null;
  if (correlation != null) {
    if (correlation >= COMPARE_THRESHOLDS.highCorrelation) {
      similarities.push(`共同区间收益高度相关（${correlation.toFixed(2)}）`);
    } else if (correlation <= COMPARE_THRESHOLDS.lowCorrelation) {
      differences.push(`共同区间收益低相关（${correlation.toFixed(2)}），具备分散价值`);
    }
    if (overlap < COMPARE_THRESHOLDS.minOverlap) {
      correlationNote = `共同收益期数仅 ${overlap} 期（少于 ${COMPARE_THRESHOLDS.minOverlap} 期），相关性仅供参考。`;
    }
  } else if (overlap > 0) {
    correlationNote = `共同收益期数不足（${overlap} 期），无法计算可靠的相关性。`;
  }

  return { similarities, differences, correlation, overlap, correlationNote };
}
