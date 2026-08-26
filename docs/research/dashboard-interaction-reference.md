# 投研 Dashboard 交互参考

调研日期：2026-08-25。目标不是复制视觉风格，而是借鉴成熟数据产品的导航、筛选和钻取规则。

## 已采用的收敛

- 产品库只承担产品管理、筛选和进入研究的职责。
- 工作台的「产品评分与画像」集中展示全量评分，并从列表展开单产品的收益、风险与归因明细。
- 因此删除产品库中的「评分总览」入口，避免同一套评分表出现两次。

## GitHub 参考与可迁移原则

### Metabase：点击必须有明确的分析动作

Metabase 把表格和图表点击定义为有限的动作，例如按该值筛选、查看底层记录、拆分维度或缩放；每次钻取会生成新查询，不改写原问题。[官方仓库](https://github.com/metabase/metabase) 的 [drill-through 文档](https://github.com/metabase/metabase/blob/master/docs/questions/visualizations/drill-through.md) 和 [dashboard interactivity 文档](https://github.com/metabase/metabase/blob/master/docs/dashboards/interactive.md) 均明确了这一点。

适用于本项目：

- 产品名称：进入该产品的研究工作台；
- 勾选框：只用于两产品对比；
- 「加入对话」：只用于 Agent 研究上下文；
- 不让同一个点击既选择、又跳页、又开始计算。

### Apache Superset：全局筛选要有范围和可见状态

Superset 在 [native filter](https://github.com/apache/superset/tree/master/superset-frontend/src/dashboard/components/nativeFilters) 与 [cross filter](https://github.com/apache/superset/blob/master/superset-frontend/src/dashboard/components/nativeFilters/FilterBar/CrossFilters/CrossFilter.tsx) 中将筛选条件、作用范围和跨图筛选作为明确概念处理。

适用于本项目：

- 以后增加产品池、策略、频率或时间范围筛选时，应在工作台顶部持续显示当前条件；
- 只影响当前面板的筛选，不能悄悄改变别的面板；
- 产品库的管理筛选与工作台中的研究对象是两种上下文，需分别命名、分别展示。

### Perspective：高密度数据界面先总览、后钻取

[Perspective](https://github.com/perspective-dev/perspective) 是成熟的可组合数据工作区项目，适合作为高密度表格、图表并列和按需加载的实现参考。

适用于本项目：

- 保持「全量评分表 → 展开单产品画像」的逐层展示；
- 把低频使用的归因、组合风险等面板延迟加载；
- 不在首屏同时展示所有指标和研究明细。

## 下一步优先级

1. 产品库、工作台、对比栏都显示各自的已选数量与清空动作，避免用户不清楚当前上下文。
2. 产品评分表的产品名称增加明确的「查看画像」动作；勾选框只保留对比含义。
3. 将工作台的产品、面板和关键筛选写入 URL 参数，支持刷新后恢复与分享链接。

上述建议均以当前产品工作流为前提；不会把通用 BI 的复杂查询编辑器引入本项目。
