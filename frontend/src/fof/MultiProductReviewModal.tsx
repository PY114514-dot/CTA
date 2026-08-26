/** Review curve candidates on their original page before entering NAV calibration. */

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Alert, Button, Card, Input, InputNumber, Modal, Select, Space, Tag, Typography, message } from "antd";
import { kbGetSourcePreview, type KbFile, type KbProduct } from "../api";

const { Text } = Typography;

export interface CropRegion {
  left_ratio: number;
  top_ratio: number;
  right_ratio: number;
  bottom_ratio: number;
}

export interface CurveReviewChoice {
  fragment_id: string;
  product_id: string;
  region: CropRegion;
}

type Candidate = NonNullable<KbFile["curve_bindings"]>[number];
type CandidateGroup = { id: string; candidates: Candidate[]; page: number; region: CropRegion };

function defaultRegion(candidate: Candidate): CropRegion {
  const box = candidate.bbox ?? {};
  return {
    left_ratio: Math.max(0, Math.min(1, box.left_ratio ?? 0.1)),
    top_ratio: Math.max(0, Math.min(1, box.top_ratio ?? 0.1)),
    right_ratio: Math.max(0, Math.min(1, box.right_ratio ?? 0.9)),
    bottom_ratio: Math.max(0, Math.min(1, box.bottom_ratio ?? 0.9)),
  };
}

function clampRegion(region: CropRegion): CropRegion {
  const left = Math.max(0, Math.min(0.98, region.left_ratio));
  const top = Math.max(0, Math.min(0.98, region.top_ratio));
  return {
    left_ratio: left,
    top_ratio: top,
    right_ratio: Math.max(left + 0.02, Math.min(1, region.right_ratio)),
    bottom_ratio: Math.max(top + 0.02, Math.min(1, region.bottom_ratio)),
  };
}

function overlapRatio(left: CropRegion, right: CropRegion): number {
  const width = Math.max(0, Math.min(left.right_ratio, right.right_ratio) - Math.max(left.left_ratio, right.left_ratio));
  const height = Math.max(0, Math.min(left.bottom_ratio, right.bottom_ratio) - Math.max(left.top_ratio, right.top_ratio));
  const overlap = width * height;
  return overlap / Math.max(Math.min(
    (left.right_ratio - left.left_ratio) * (left.bottom_ratio - left.top_ratio),
    (right.right_ratio - right.left_ratio) * (right.bottom_ratio - right.top_ratio),
  ), 0.0001);
}

function groupCandidates(candidates: Candidate[]): CandidateGroup[] {
  const groups: CandidateGroup[] = [];
  for (const candidate of candidates) {
    const page = candidate.page_number && candidate.page_number > 0 ? candidate.page_number - 1 : 0;
    const region = defaultRegion(candidate);
    const existing = groups.find((group) => group.page === page && overlapRatio(group.region, region) >= 0.55);
    if (existing) {
      existing.candidates.push(candidate);
      existing.region = clampRegion({
        left_ratio: Math.min(existing.region.left_ratio, region.left_ratio),
        top_ratio: Math.min(existing.region.top_ratio, region.top_ratio),
        right_ratio: Math.max(existing.region.right_ratio, region.right_ratio),
        bottom_ratio: Math.max(existing.region.bottom_ratio, region.bottom_ratio),
      });
    } else {
      groups.push({ id: candidate.fragment_id, candidates: [candidate], page, region });
    }
  }
  return groups.map((group) => ({
    ...group,
    // Keep the axes and tick labels in the crop; VLM boxes often start at
    // the curve itself and otherwise cut off the left side of the chart.
    region: clampRegion({ ...group.region, left_ratio: Math.max(0, group.region.left_ratio - 0.05), right_ratio: Math.min(1, group.region.right_ratio + 0.02) }),
  }));
}

export default function MultiProductReviewModal({
  file, products, open, submitting, onCancel, onSubmit, onCreateProduct,
}: {
  file: KbFile | null;
  products: KbProduct[];
  open: boolean;
  submitting: boolean;
  onCancel: () => void;
  onSubmit: (choices: CurveReviewChoice[], openAfterSubmit?: CurveReviewChoice, startQueue?: boolean) => Promise<void>;
  onCreateProduct: (standardName: string) => Promise<KbProduct>;
}): React.JSX.Element {
  const candidates = useMemo(() => {
    const latestByCurve = new Map<string, Candidate>();
    for (const candidate of (file?.curve_bindings ?? [])) {
      if (candidate.binding_status === "matched") continue;
      const page = candidate.page_number && candidate.page_number > 0 ? candidate.page_number : 1;
      // Re-parses retain the audit trail.  Only the newest candidate for the
      // same line belongs in the reviewer UI.
      latestByCurve.set(`${page}:${candidate.curve_index}`, candidate);
    }
    return [...latestByCurve.values()].sort((left, right) => left.curve_index - right.curve_index);
  }, [file]);
  const groups = useMemo(() => groupCandidates(candidates), [candidates]);
  const groupByCandidateId = useMemo(() => new Map(groups.flatMap((group) => group.candidates.map((candidate) => [candidate.fragment_id, group.id]))), [groups]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [createdProducts, setCreatedProducts] = useState<KbProduct[]>([]);
  const [regions, setRegions] = useState<Record<string, CropRegion>>({});
  const [newNames, setNewNames] = useState<Record<string, string>>({});
  const [creatingId, setCreatingId] = useState<string>();
  const [activeId, setActiveId] = useState<string>();
  const [previewUrl, setPreviewUrl] = useState<string>();
  const [previewError, setPreviewError] = useState<string>();
  const dragRef = useRef<{ id: string; x: number; y: number; region: CropRegion } | undefined>(undefined);

  useEffect(() => {
    const initialValues: Record<string, string> = {};
    const initialRegions: Record<string, CropRegion> = {};
    groups.forEach((group) => {
      initialRegions[group.id] = group.region;
      group.candidates.forEach((candidate) => {
      if (candidate.product_id) initialValues[candidate.fragment_id] = candidate.product_id;
      });
    });
    setValues(initialValues);
    setCreatedProducts([]);
    setRegions(initialRegions);
    setNewNames({});
    setActiveId(groups[0]?.id);
  }, [groups]);

  const active = groups.find((group) => group.id === activeId) ?? groups[0];
  const activePage = active?.page ?? 0;

  useEffect(() => {
    if (!open || !file) return;
    let disposed = false;
    let objectUrl: string | undefined;
    setPreviewUrl(undefined);
    setPreviewError(undefined);
    void kbGetSourcePreview(file.id, activePage)
      .then((preview) => {
        if (disposed) return;
        objectUrl = URL.createObjectURL(preview);
        setPreviewUrl(objectUrl);
      })
      .catch(() => { if (!disposed) setPreviewError("无法读取原始页面预览；仍可先确认产品归属后进入复核。 "); });
    return () => {
      disposed = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [activePage, file, open]);

  const activeRegion = active ? regions[active.id] ?? active.region : undefined;
  const updateActiveRegion = (next: Partial<CropRegion>) => {
    if (!active || !activeRegion) return;
    setRegions((current) => ({ ...current, [active.id]: clampRegion({ ...activeRegion, ...next }) }));
  };
  const selections = candidates
    .filter((candidate) => values[candidate.fragment_id])
    .map((candidate) => ({
      fragment_id: candidate.fragment_id,
      product_id: values[candidate.fragment_id]!,
      region: regions[groupByCandidateId.get(candidate.fragment_id) ?? ""] ?? defaultRegion(candidate),
    }));
  const createAndSelect = async (candidate: Candidate) => {
    const standardName = (newNames[candidate.fragment_id] || candidate.layout_product_name || "").trim();
    if (!standardName) return;
    setCreatingId(candidate.fragment_id);
    try {
      const product = await onCreateProduct(standardName);
      setCreatedProducts((current) => current.some((item) => item.id === product.id) ? current : [...current, product]);
      setValues((current) => ({ ...current, [candidate.fragment_id]: product.id }));
    } catch (error) {
      message.error(error instanceof Error ? error.message : "新建产品失败");
    } finally {
      setCreatingId(undefined);
    }
  };

  return <Modal
    title={file ? `多产品切图复核 · ${file.filename}` : "多产品切图复核"}
    open={open}
    onCancel={onCancel}
    footer={null}
    width={1120}
    destroyOnClose
  >
    <Alert showIcon type="info" style={{ marginBottom: 12 }} message="先确认每条曲线对应的产品，再进入单产品校准" description="框选只是确定本次复核的图表区域；净值不会自动写入，仍须在下一页核对日期、坐标和曲线后确认。" />
    {!candidates.length ? <Alert showIcon type="warning" message="当前资料没有待确认的曲线候选" /> : <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.4fr) minmax(300px, 0.8fr)", gap: 16, alignItems: "start" }}>
      <div>
        {previewError && <Alert type="warning" showIcon message={previewError} style={{ marginBottom: 8 }} />}
        <div style={{ position: "relative", minHeight: 250, border: "1px solid var(--serif-border)", borderRadius: 8, overflow: "hidden", background: "var(--serif-muted)" }}>
          {previewUrl ? <img src={previewUrl} alt="多产品来源页面" style={{ display: "block", width: "100%" }} /> : <div style={{ height: 300, display: "grid", placeItems: "center" }}><Text type="secondary">正在加载原始页面…</Text></div>}
          {previewUrl && groups.filter((group) => group.page === activePage).map((group, index) => {
            const region = regions[group.id] ?? group.region;
            const selected = group.id === active?.id;
            return <button
              key={group.id}
              type="button"
              aria-label={`选择图表区域 ${index + 1}`}
              onClick={() => setActiveId(group.id)}
              onPointerDown={(event) => {
                if (!selected) return;
                event.currentTarget.setPointerCapture(event.pointerId);
                dragRef.current = { id: group.id, x: event.clientX, y: event.clientY, region };
              }}
              onPointerMove={(event) => {
                const drag = dragRef.current;
                const parent = event.currentTarget.parentElement;
                if (!drag || drag.id !== group.id || !parent) return;
                const bounds = parent.getBoundingClientRect();
                const dx = (event.clientX - drag.x) / bounds.width;
                const dy = (event.clientY - drag.y) / bounds.height;
                const width = drag.region.right_ratio - drag.region.left_ratio;
                const height = drag.region.bottom_ratio - drag.region.top_ratio;
                const left = Math.max(0, Math.min(1 - width, drag.region.left_ratio + dx));
                const top = Math.max(0, Math.min(1 - height, drag.region.top_ratio + dy));
                setRegions((current) => ({ ...current, [drag.id]: { left_ratio: left, top_ratio: top, right_ratio: left + width, bottom_ratio: top + height } }));
              }}
              onPointerUp={() => { dragRef.current = undefined; }}
              style={{ position: "absolute", left: `${region.left_ratio * 100}%`, top: `${region.top_ratio * 100}%`, width: `${(region.right_ratio - region.left_ratio) * 100}%`, height: `${(region.bottom_ratio - region.top_ratio) * 100}%`, border: `2px solid ${selected ? "#1677ff" : "#faad14"}`, background: selected ? "rgba(22,119,255,.08)" : "rgba(250,173,20,.05)", cursor: selected ? "move" : "pointer", padding: 0 }}
            ><span style={{ position: "absolute", top: -25, left: -2, whiteSpace: "nowrap", color: "white", background: selected ? "#1677ff" : "#d48806", borderRadius: 3, padding: "2px 6px", fontSize: 11 }}>图表区域 {index + 1} · {group.candidates.length} 条曲线</span></button>;
          })}
        </div>
        <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 8 }}>当前为 {active?.page ? `第 ${active.page + 1} 页` : "来源原图"}。同一图内的多条线共用一个图表范围；拖动框只调整这张图，不会把曲线混在一起。</Text>
      </div>
      <div style={{ maxHeight: "66vh", overflow: "auto", paddingRight: 2 }}>
        <Space direction="vertical" size={10} style={{ width: "100%" }}>
          {active && activeRegion && <Card size="small" title="当前图表范围" style={{ borderColor: "#1677ff" }}>
            <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>左侧含纵轴刻度；右侧含图例；上、下侧分别保留图顶和横轴日期。数值是相对原图的位置（0 到 1），通常只需拖动蓝框。</Text>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {([['left_ratio', '左侧（含纵轴）'], ['top_ratio', '上侧'], ['right_ratio', '右侧（含图例）'], ['bottom_ratio', '下侧（含横轴）']] as const).map(([field, label]) => <div key={field}><Text type="secondary" style={{ fontSize: 11 }}>{label}</Text><InputNumber min={0} max={1} step={0.01} precision={2} value={activeRegion[field]} onChange={(value) => updateActiveRegion({ [field]: Number(value) } as Partial<CropRegion>)} style={{ width: "100%" }} /></div>)}
            </div>
          </Card>}
          {candidates.map((candidate) => {
            const selected = groupByCandidateId.get(candidate.fragment_id) === active?.id;
            return <Card key={candidate.fragment_id} size="small" onClick={() => setActiveId(groupByCandidateId.get(candidate.fragment_id))} style={{ cursor: "pointer", borderColor: selected ? "#1677ff" : undefined }}>
              <Space size={6} wrap><Tag color={selected ? "blue" : "default"}>曲线 {candidate.curve_index}</Tag>{candidate.page_number && <Tag style={{ margin: 0 }}>第 {candidate.page_number} 页</Tag>}<Text type="secondary" style={{ fontSize: 11 }}>识别 {Math.round((candidate.binding_confidence ?? 0) * 100)}%</Text></Space>
              <Select allowClear placeholder="选择对应产品" value={values[candidate.fragment_id]} onChange={(value) => setValues((current) => ({ ...current, [candidate.fragment_id]: value }))} style={{ width: "100%", marginTop: 8 }} options={[...products, ...createdProducts.filter((created) => !products.some((product) => product.id === created.id))].map((product) => ({ value: product.id, label: product.standard_name }))} />
              {!values[candidate.fragment_id] && <Space.Compact style={{ display: "flex", marginTop: 8 }} onClick={(event) => event.stopPropagation()}>
                <Input placeholder="或输入新产品名称" value={newNames[candidate.fragment_id] ?? candidate.layout_product_name ?? ""} onChange={(event) => setNewNames((current) => ({ ...current, [candidate.fragment_id]: event.target.value }))} />
                <Button loading={creatingId === candidate.fragment_id} disabled={submitting} onClick={() => void createAndSelect(candidate)}>新增</Button>
              </Space.Compact>}
              {!!candidate.binding_evidence?.length && <Text type="secondary" style={{ display: "block", marginTop: 8, fontSize: 11 }}>{candidate.binding_evidence.join("；")}</Text>}
            </Card>;
          })}
        </Space>
      </div>
    </div>}
    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
      <Button onClick={onCancel} disabled={submitting}>稍后处理</Button>
      <Space wrap>
        <Button disabled={!selections.length || submitting} loading={submitting} onClick={() => void onSubmit(selections)}>确认归属</Button>
        <Button disabled={!selections.length || submitting} loading={submitting} onClick={() => void onSubmit(selections, undefined, true)}>确认并逐条校准</Button>
        <Button type="primary" disabled={!active || !active.candidates.some((candidate) => values[candidate.fragment_id]) || submitting} loading={submitting} onClick={() => {
          const selectedChoice = selections.find((choice) => active?.candidates.some((candidate) => candidate.fragment_id === choice.fragment_id));
          if (selectedChoice) void onSubmit(selections, selectedChoice);
        }}>确认并校准当前曲线</Button>
      </Space>
    </div>
  </Modal>;
}
