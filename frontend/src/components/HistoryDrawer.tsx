import React from "react";
import { Button, Drawer, Empty, List, Popconfirm, Tag, Typography } from "antd";
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
    <Drawer title="历史记录" placement="right" width={480} open={open} onClose={onClose}>
      {records.length === 0 ? (
        <Empty description={"暂无保存的记录，计算完成后点击「保存到历史记录」。"} />
      ) : (
        <List
          dataSource={records}
          renderItem={(record) => (
            <List.Item
              actions={[
                <Button key="view" size="small" type="primary" onClick={() => onView(record)}>
                  查看详情
                </Button>,
                <Button
                  key="md"
                  size="small"
                  onClick={() => {
                    const safeName = (record.product_name || "产品").replace(/[\\/:*?"<>|]/g, "_");
                    downloadFile(generateMarkdownReport(record), `${safeName}_报告.md`, "text/markdown;charset=utf-8");
                  }}
                >
                  MD
                </Button>,
                <Button key="pdf" size="small" onClick={() => { void exportPdfReport(record); }}>
                  导出 PDF
                </Button>,
                <Popconfirm
                  key="del"
                  title="确定删除此记录？"
                  onConfirm={() => onDelete(record.id)}
                  okText="删除"
                  cancelText="取消"
                >
                  <Button size="small" danger>
                    删除
                  </Button>
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                title={
                  <span
                    role="button"
                    tabIndex={0}
                    onDoubleClick={() => onView(record)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") onView(record);
                    }}
                    title="双击查看当次研究看板"
                    style={{ cursor: "pointer" }}
                  >
                    <Tag color="blue">{record.product_name}</Tag>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {FREQUENCY_LABEL[record.frequency] ?? record.frequency} · {record.nav_count} 点
                    </Text>
                    {record.strategy_profile && <Tag style={{ marginLeft: 6 }}>画像</Tag>}
                    {record.ai_report && <Tag style={{ marginLeft: 6 }}>AI 分析</Tag>}
                  </span>
                }
                description={
                  <span style={{ fontSize: 12 }}>
                    {record.saved_at}
                    <br />
                    年化 {percentage(record.metrics.annualized_return)} · 回撤 {percentage(record.metrics.maximum_drawdown)} · 夏普 {decimal(record.metrics.sharpe_ratio)}
                  </span>
                }
              />
            </List.Item>
          )}
        />
      )}
    </Drawer>
  );
}
