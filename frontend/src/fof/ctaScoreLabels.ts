export const DIMENSION_LABELS: Record<string, string> = {
  absolute: "绝对收益",
  risk_adjusted: "风险调整收益",
  trend_regime: "趋势与市场环境",
  tail: "尾部风险",
  robustness: "稳健性",
};

export const METRIC_LABELS: Record<string, string> = {
  annualized_return: "年化收益",
  rolling_return_median: "滚动收益中位数",
  positive_period_ratio: "正收益期占比",
  gain_loss_asymmetry: "盈亏不对称性",
  sharpe_ratio: "夏普比率",
  sortino_ratio: "索提诺比率",
  calmar_ratio: "卡玛比率",
  tail_loss_protection: "尾部损失保护",
  trend_beta: "趋势关联系数",
  trend_alignment: "趋势一致性",
  reversal_resilience: "反转韧性",
  exposure_stability: "暴露稳定性",
  stress_hit_rate: "压力期正收益占比",
  left_tail_capture: "左尾捕获能力",
  recovery_speed: "回撤修复速度",
  tail_dependence_protection: "尾部相关性保护",
  rolling_rank_stability: "滚动排名稳定性",
  coefficient_stability: "系数稳定性",
  bootstrap_persistence: "自助法持续性",
  model_sensitivity: "模型敏感度",
  oos_validation: "样本外验证",
};

export function dimensionLabel(key: string, fallback?: string): string {
  return DIMENSION_LABELS[key] ?? fallback ?? key;
}

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key;
}
