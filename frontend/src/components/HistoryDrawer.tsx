import React from "react";
import { Button, Drawer, Empty, List, Popconfirm, Space, Tag, Typography } from "antd";
import { decimal, percentage, downloadFile } from "../utils/format";
import { generateMarkdownReport } from "../report/markdownReport";
import { exportPdfReport } from "../report/exportPdf";
import type { HistoryRecord } from "../storage/historyStorage";

const { Text } = Typography;

export interface HistoryDrawerProps {
  open: boolean;
  onClose: () => void;
  records: HistoryRecord[];
  onDelete: (id: string) => void;
  onView: (record: HistoryRecord) => void;
}

const FREQUENCY_LABEL: Record<string, string> = {
  daily: "日频",
  weekly: "周频",
  monthly: "月频",
};

/** Right-hand drawer listing saved analysis history with MD/PDF export and delete. */
export default function HistoryDrawer({ open, onClose, records, onDelete, onView }: HistoryDrawerProps): React.JSX.Element {
  return (
    <Drawer title="研究档案" placement="right" width={560} open={open} onClose={onClose}>
      {records.length === 0 ? (
        <Empty description={"暂无保存的研究，完成分析后点击「保存到研究档案」。"} />
      ) : (
        <List
          dataSource={records}
          renderItem={(record) => (
            <List.Item style={{ display: "block", padding: "16px 0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
                <div style={{ minWidth: 0 }}>
                  <div role="button" tabIndex={0} onClick={() => onView(record)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onView(record); }} title="查看当次研究看板" style={{ cursor: "pointer" }}>
                    <Tag color="blue" style={{ fontSize: 14, fontWeight: 600, marginInlineEnd: 8 }}>{record.product_name}</Tag>
                    <Text type="secondary" style={{ fontSize: 12 }}>{FREQUENCY_LABEL[record.frequency] ?? record.frequency} · {record.nav_count} 点</Text>
                  </div>
                  <Space size={6} wrap style={{ marginTop: 7 }}>
                    {record.strategy_profile && <Tag style={{ margin: 0 }}>画像</Tag>}
                    {record.ai_report && <Tag style={{ margin: 0 }}>AI 分析</Tag>}
                    <Text type="secondary" style={{ fontSize: 12 }}>{record.saved_at}</Text>
                  </Space>
                </div>
                <Button size="small" type="primary" onClick={() => onView(record)}>查看详情</Button>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8, marginTop: 14 }}>
                <HistoryMetric label="年化收益" value={percentage(record.metrics.annualized_return)} />
                <HistoryMetric label="最大回撤" value={percentage(record.metrics.maximum_drawdown)} />
                <HistoryMetric label="夏普比率" value={decimal(record.metrics.sharpe_ratio)} />
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
                <Button size="small" onClick={() => {
                  const safeName = (record.product_name || "产品").replace(/[\\/:*?"<>|]/g, "_");
                  downloadFile(generateMarkdownReport(record), `${safeName}_报告.md`, "text/markdown;charset=utf-8");
                }}>导出 MD</Button>
                <Button size="small" onClick={() => { void exportPdfReport(record); }}>导出 PDF</Button>
                <Popconfirm title="确定删除此记录？" onConfirm={() => onDelete(record.id)} okText="删除" cancelText="取消">
                  <Button size="small" danger>删除</Button>
                </Popconfirm>
              </div>
            </List.Item>
          )}
        />
      )}
    </Drawer>
  );
}

function HistoryMetric({ label, value }: { label: string; value: string }): React.JSX.Element {
  return <div style={{ minWidth: 0, padding: "9px 10px", border: "1px solid var(--serif-border)", borderRadius: 7, background: "var(--serif-muted)" }}>
    <Text type="secondary" style={{ display: "block", fontSize: 11 }}>{label}</Text>
    <Text strong style={{ display: "block", marginTop: 3, fontSize: 14, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{value}</Text>
  </div>;
}
