/** Right panel: candidate pool, comparison and investment-committee review. */

import React, { useCallback, useEffect, useState } from "react";
import { Button, Card, Descriptions, Empty, List, Space, Statistic, Tag, Typography, message } from "antd";
import {
  AuditOutlined,
  DeleteOutlined,
  FundOutlined,
} from "@ant-design/icons";
import { kbGetProduct, type KbProductDetail } from "../api";

const { Text } = Typography;

interface Props {
  selectedProductIds: string[];
}

export default function CandidatePanel({ selectedProductIds }: Props): React.JSX.Element {
  const [details, setDetails] = useState<KbProductDetail[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedProductIds.length === 0) {
      setDetails([]);
      return;
    }
    setLoading(true);
    Promise.all(selectedProductIds.map((id) => kbGetProduct(id).catch(() => null)))
      .then((results) => setDetails(results.filter((r): r is KbProductDetail => r !== null)))
      .finally(() => setLoading(false));
  }, [selectedProductIds]);

  return (
    <Card
      size="small"
      style={{ height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}
      styles={{ body: { flex: 1, overflow: "auto", minHeight: 0, padding: "12px 14px" } }}
      title={
        <Space size={8}>
          <FundOutlined style={{ color: "var(--serif-accent, #1677ff)" }} />
          <span>候选池</span>
          {details.length > 0 && <Tag style={{ fontSize: 10 }}>{details.length}</Tag>}
        </Space>
      }
      extra={
        <Button size="small" icon={<AuditOutlined />} disabled={details.length === 0}>
          提交审核
        </Button>
      }
    >
      {details.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <Space direction="vertical" size={4}>
              <Text type="secondary">从左侧选择产品加入候选池</Text>
              <Text type="secondary" style={{ fontSize: 11 }}>
                或等待 Agent 推荐结果
              </Text>
            </Space>
          }
          style={{ marginTop: 60 }}
        />
      ) : (
        <List
          size="small"
          loading={loading}
          dataSource={details}
          renderItem={(product) => (
            <CandidateCard key={product.id} product={product} />
          )}
        />
      )}
    </Card>
  );
}

function CandidateCard({ product }: { product: KbProductDetail }): React.JSX.Element {
  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 6,
        border: "1px solid var(--serif-border, #f0f0f0)",
        marginBottom: 8,
        background: "var(--serif-card-bg, #fff)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Text strong style={{ fontSize: 13 }}>{product.standard_name}</Text>
        <Tag
          color={product.confirmation_status === "confirmed" ? "success" : "processing"}
          style={{ fontSize: 10, margin: 0 }}
        >
          {product.confirmation_status === "confirmed" ? "已确认" : "待确认"}
        </Tag>
      </div>

      <div style={{ marginTop: 6, fontSize: 11, color: "var(--serif-text-secondary, #888)" }}>
        {product.manager_name && <span>{product.manager_name}</span>}
        {product.strategy && <span> · {product.strategy}</span>}
      </div>

      <div style={{ marginTop: 8, display: "flex", gap: 16 }}>
        <Statistic
          title="净值点"
          value={product.nav_count}
          valueStyle={{ fontSize: 14, fontVariantNumeric: "tabular-nums" }}
        />
        <Statistic
          title="事实"
          value={product.fact_count}
          valueStyle={{ fontSize: 14, fontVariantNumeric: "tabular-nums" }}
        />
        {product.aliases.length > 0 && (
          <Statistic
            title="别名"
            value={product.aliases.length}
            valueStyle={{ fontSize: 14, fontVariantNumeric: "tabular-nums" }}
          />
        )}
      </div>

      {product.aliases.length > 0 && (
        <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 4 }}>
          {product.aliases.map((a) => (
            <Tag key={a.id} style={{ fontSize: 10, margin: 0 }}>{a.alias}</Tag>
          ))}
        </div>
      )}
    </div>
  );
}
