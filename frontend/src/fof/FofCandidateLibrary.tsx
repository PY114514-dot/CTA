import React, { useCallback, useEffect, useState } from "react";
import { Button, Card, Empty, Tag, Typography } from "antd";
import { ReloadOutlined, SafetyCertificateOutlined } from "@ant-design/icons";
import { listFofLibraryProducts, type FofLibraryProduct } from "../api";

const { Text } = Typography;

export default function FofCandidateLibrary({ refreshKey }: { refreshKey: number }): React.JSX.Element {
  const [products, setProducts] = useState<FofLibraryProduct[]>([]);
  const [loading, setLoading] = useState(false);
  const refresh = useCallback(async () => {
    setLoading(true);
    try { setProducts(await listFofLibraryProducts()); } finally { setLoading(false); }
  }, []);

  useEffect(() => { void refresh(); }, [refresh, refreshKey]);

  return <Card
    size="small"
    className="card-accent-top"
    title={<span><SafetyCertificateOutlined style={{ marginRight: 6 }} />已识别候选</span>}
    extra={<Button type="text" size="small" icon={<ReloadOutlined />} aria-label="刷新已识别候选" onClick={() => void refresh()} loading={loading} />}
    styles={{ body: { padding: 8 } }}
  >
    {products.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="上传周报后显示候选" /> : (
      <div style={{ display: "flex", flexDirection: "column", gap: 7, maxHeight: 260, overflowY: "auto" }}>
        {products.map((product) => <div key={product.product_id} style={{ padding: "7px 8px", border: "1px solid var(--serif-border)", borderRadius: 6 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 5 }}>
            <Text strong ellipsis style={{ fontSize: 12, maxWidth: 170 }}>{product.name}</Text>
            <Tag color="processing" style={{ margin: 0, fontSize: 10 }}>待复核</Tag>
          </div>
          <Text type="secondary" style={{ display: "block", fontSize: 10, marginTop: 3 }}>{product.manager_name ?? "管理人待确认"}</Text>
          <Text type="secondary" style={{ display: "block", fontSize: 10 }}>证据 {product.evidence_count} 条 · 素材 {product.material_count} 份</Text>
        </div>)}
      </div>
    )}
  </Card>;
}
