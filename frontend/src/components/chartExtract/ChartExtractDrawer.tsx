/**
 * Chart extraction drawer - top-level flow: upload -> configure -> results.
 * TracedChartPreview / ResultReview sub-components and shared helpers were
 * extracted into this chartExtract/ package from the original single file.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  ColorPicker,
  Divider,
  Drawer,
  Empty,
  Input,
  Progress,
  Space,
  Spin,
  Steps,
  Tag,
  Tooltip,
  Typography,
  Upload,
} from "antd";
import type { Color } from "antd/es/color-picker";
import {
  extractPdf,
  extractRegionImageUrl,
  extractSingleImage,
  getExtractJobPreview,
  getExtractJobStatus,
  getVlmConfig,
  sampleColorAtPixel,
  type ExtractChartResult,
  type ExtractJobPreview,
  type ExtractJobStatus,
  type VlmConfig,
} from "../../api";
import {
  IMAGE_ACCEPT,
  PDF_ACCEPT,
  nextCurveId,
  parseYTicksInput,
  type CurveRow,
} from "./utils";
import { FONT_MONO } from "../../theme";
import { TracedChartPreview } from "./TracedChartPreview";
import { ResultReview } from "./ResultReview";

const { Text, Paragraph } = Typography;

// ---------------------------------------------------------------------------
// Main drawer
// ---------------------------------------------------------------------------

interface ChartExtractDrawerProps {
  open: boolean;
  onClose: () => void;
  /** Write an extracted "date,value" series into the main workspace textarea. */
  onApplyToWorkspace: (navText: string, reviewConfirmed: boolean) => void;
}

type StepKey = "upload" | "configure" | "results";

export default function ChartExtractDrawer({
  open,
  onClose,
  onApplyToWorkspace,
}: ChartExtractDrawerProps): React.JSX.Element {
  const [step, setStep] = useState<StepKey>("upload");

  // --- image intake ---
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");

  // --- curve configuration ---
  const [curves, setCurves] = useState<CurveRow[]>([]);
  const [pickingId, setPickingId] = useState<string | null>(null);
  const [isSampling, setIsSampling] = useState(false);
  const [yTicksText, setYTicksText] = useState<string>("");

  // --- extraction result (single image) ---
  const [result, setResult] = useState<ExtractChartResult | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState<string>("");

  // --- pdf flow ---
  const [pdfName, setPdfName] = useState<string>("");
  const [pdfJob, setPdfJob] = useState<ExtractJobStatus | null>(null);
  const [pdfPreview, setPdfPreview] = useState<ExtractJobPreview | null>(null);
  const [expandedPdfResult, setExpandedPdfResult] = useState<number | null>(null);

  const [vlmConfig, setVlmConfig] = useState<VlmConfig | null>(null);

  const pollTimerRef = useRef<number | null>(null);

  // Load VLM config whenever the drawer opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void getVlmConfig()
      .then((config) => {
        if (!cancelled) setVlmConfig(config);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [open]);

  // Clean up the object URL and poll timer on close / unmount.
  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current);
    };
  }, [imageUrl]);

  // ESC cancels color-picking mode (matches the hint shown while picking).
  useEffect(() => {
    if (!pickingId) return;
    const onKeyDown = (e: KeyboardEvent): void => {
      if (e.key === "Escape") setPickingId(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [pickingId]);

  const resetAll = useCallback(() => {
    setStep("upload");
    setImageFile(null);
    setImageUrl("");
    setCurves([]);
    setPickingId(null);
    setYTicksText("");
    setResult(null);
    setIsExtracting(false);
    setError("");
    setPdfName("");
    setPdfJob(null);
    setPdfPreview(null);
    setExpandedPdfResult(null);
  }, []);

  // --- image selection ---
  const handleSelectImage = useCallback((file: File) => {
    setImageFile(file);
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    setResult(null);
    setError("");
    setCurves([
      { id: nextCurveId(), name: "产品净值", colorHex: "#E03030", isBenchmark: false },
    ]);
    setStep("configure");
  }, []);

  // --- curve row editing ---
  const updateCurve = useCallback((id: string, patch: Partial<CurveRow>) => {
    setCurves((rows) => rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }, []);

  const addCurve = useCallback(() => {
    setCurves((rows) => [
      ...rows,
      { id: nextCurveId(), name: `曲线${rows.length + 1}`, colorHex: "#1E5799", isBenchmark: false },
    ]);
  }, []);

  const removeCurve = useCallback((id: string) => {
    setCurves((rows) => rows.filter((row) => row.id !== id));
    setPickingId((current) => (current === id ? null : current));
  }, []);

  // --- click-to-pick color ---
  const handlePickColor = useCallback(
    async (x: number, y: number) => {
      if (!pickingId || !imageFile) return;
      setIsSampling(true);
      try {
        const sampled = await sampleColorAtPixel(imageFile, x, y);
        updateCurve(pickingId, { colorHex: sampled.color_hex });
        setPickingId(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "取色失败");
      } finally {
        setIsSampling(false);
      }
    },
    [pickingId, imageFile, updateCurve],
  );

  // --- manual extraction (deterministic CV, no VLM) ---
  const handleManualExtract = useCallback(async () => {
    if (!imageFile) return;
    const valid = curves.filter((c) => c.colorHex.trim());
    if (valid.length === 0) {
      setError("请至少配置一条曲线颜色（可输入色值或在图上点击取色）。");
      return;
    }
    setIsExtracting(true);
    setError("");
    try {
      const yTicks = parseYTicksInput(yTicksText);
      const response = await extractSingleImage(
        imageFile,
        valid.map((c) => ({
          name: c.name,
          color_hex: c.colorHex,
          is_benchmark: c.isBenchmark,
        })),
        { useVlm: false, yTicks: yTicks.length > 0 ? yTicks : undefined },
      );
      setResult(response.result);
      setStep("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "提取失败");
    } finally {
      setIsExtracting(false);
    }
  }, [imageFile, curves, yTicksText]);

  // --- VLM auto detection (structure + trace in one call) ---
  const handleAutoExtract = useCallback(async () => {
    if (!imageFile) return;
    setIsExtracting(true);
    setError("");
    try {
      const response = await extractSingleImage(imageFile, undefined, { useVlm: true });
      setResult(response.result);
      // Pre-fill the curve editor from the VLM structure for later tweaking.
      if (response.result.structure) {
        const detected = response.result.structure.curves
          .map((c) => ({
            id: nextCurveId(),
            name: String(c.name ?? "曲线"),
            colorHex: String(c.color_hex ?? "#888888"),
            isBenchmark: Boolean(c.is_benchmark),
          }));
        if (detected.length > 0) setCurves(detected);
      }
      setStep("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "VLM 识别失败");
    } finally {
      setIsExtracting(false);
    }
  }, [imageFile]);

  // --- pdf flow ---
  const pollPdfJob = useCallback((jobId: string) => {
    const tick = async (): Promise<void> => {
      try {
        const status = await getExtractJobStatus(jobId);
        setPdfJob(status);
        if (status.status === "completed" || status.status === "failed") {
          const preview = await getExtractJobPreview(jobId);
          setPdfPreview(preview);
          return;
        }
        pollTimerRef.current = window.setTimeout(() => void tick(), 1500);
      } catch (err) {
        setError(err instanceof Error ? err.message : "查询任务失败");
      }
    };
    void tick();
  }, []);

  const handleSelectPdf = useCallback(
    async (file: File) => {
      setPdfName(file.name);
      setPdfPreview(null);
      setPdfJob(null);
      setError("");
      setStep("results");
      try {
        const response = await extractPdf(file, { useVlm: true });
        pollPdfJob(response.job_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "PDF 提取失败");
      }
    },
    [pollPdfJob],
  );

  const stepIndex = step === "upload" ? 0 : step === "configure" ? 1 : 2;
  const vlmReady = Boolean(vlmConfig?.api_key_set);

  return (
    <Drawer
      title="图表净值提取（CV 像素追踪）"
      open={open}
      onClose={onClose}
      width={760}
      destroyOnHidden
      extra={
        <Button size="small" onClick={resetAll}>
          重新开始
        </Button>
      }
    >
      <Steps
        size="small"
        current={stepIndex}
        style={{ marginBottom: 20 }}
        items={[
          { title: "上传", description: "图片或 PDF" },
          { title: "配置曲线", description: "颜色 / 取色" },
          { title: "结果导出", description: "核验 / 下载" },
        ]}
      />

      {error && (
        <Alert
          type="error"
          showIcon
          closable
          message={error}
          onClose={() => setError("")}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* ---------------- Step 1: upload ---------------- */}
      {step === "upload" && (
        <div>
          <Paragraph type="secondary">
            私募产品净值大多只以路演 PDF 中的曲线图形式存在。这里从图片像素中还原数值序列：
            视觉模型（可选）只负责读取图表结构（曲线颜色、坐标刻度），真正的数值由确定性的像素追踪给出。
          </Paragraph>

          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            <div
              style={{
                border: "1px dashed var(--serif-border)",
                borderRadius: 8,
                padding: 20,
                background: "var(--serif-card)",
              }}
            >
              <Text strong>方式一：单张曲线图片（推荐，可手动取色）</Text>
              <Paragraph type="secondary" style={{ fontSize: 13, margin: "6px 0 12px" }}>
                上传净值曲线截图，随后在图上点击即可为每条曲线取色，零 API 成本。
              </Paragraph>
              <Upload
                accept={IMAGE_ACCEPT}
                maxCount={1}
                showUploadList={false}
                beforeUpload={(file) => {
                  handleSelectImage(file);
                  return false;
                }}
              >
                <Button type="primary">选择曲线图片</Button>
              </Upload>
            </div>

            <div
              style={{
                border: "1px dashed var(--serif-border)",
                borderRadius: 8,
                padding: 20,
                background: "var(--serif-card)",
              }}
            >
              <Text strong>方式二：整份 PDF（自动检测图表）</Text>
              <Paragraph type="secondary" style={{ fontSize: 13, margin: "6px 0 12px" }}>
                自动渲染、定位图表区域并用 VLM 读取结构后逐图追踪。
                {!vlmReady && (
                  <Tag color="warning" style={{ marginLeft: 8 }}>未配置 VLM API Key</Tag>
                )}
              </Paragraph>
              <Upload
                accept={PDF_ACCEPT}
                maxCount={1}
                showUploadList={false}
                beforeUpload={(file) => {
                  void handleSelectPdf(file);
                  return false;
                }}
              >
                <Button>选择 PDF 文件</Button>
              </Upload>
              {vlmConfig && (
                <div style={{ marginTop: 10, fontSize: 12, color: "var(--serif-muted-foreground)" }}>
                  VLM：{vlmConfig.provider} / {vlmConfig.model}
                  {vlmReady ? "（已配置 Key）" : "（未配置 Key，请在后端设置 DASHSCOPE_API_KEY）"}
                </div>
              )}
            </div>
          </Space>
        </div>
      )}

      {/* ---------------- Step 2: configure curves (image) ---------------- */}
      {step === "configure" && imageFile && (
        <div>
          <Space style={{ marginBottom: 10 }} wrap>
            <Button size="small" onClick={() => setStep("upload")}>← 重新选择</Button>
            <Text type="secondary" style={{ fontSize: 13 }}>{imageFile.name}</Text>
          </Space>

          <TracedChartPreview
            src={imageUrl}
            curves={[]}
            picking={pickingId !== null}
            onPick={(x, y) => void handlePickColor(x, y)}
            maxHeight={360}
          />

          {pickingId && (
            <Alert
              type="info"
              showIcon
              style={{ marginTop: 10 }}
              message={
                isSampling
                  ? "正在取色…"
                  : "取色模式：在上方图片中点击目标曲线像素即可自动填入颜色，ESC 可取消。"
              }
            />
          )}

          <Divider titlePlacement="left" style={{ margin: "18px 0 12px" }}>曲线配置</Divider>

          {curves.map((row) => (
            <div
              key={row.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "8px 10px",
                border: "1px solid var(--serif-border)",
                borderRadius: 6,
                marginBottom: 8,
                background: "var(--serif-card)",
                flexWrap: "wrap",
              }}
            >
              <Input
                value={row.name}
                onChange={(e) => updateCurve(row.id, { name: e.target.value })}
                placeholder="曲线名称"
                style={{ width: 150 }}
                size="middle"
              />
              <ColorPicker
                value={row.colorHex}
                onChange={(color: Color) => updateCurve(row.id, { colorHex: color.toHexString() })}
                showText
                size="middle"
              />
              <Input
                value={row.colorHex}
                onChange={(e) => updateCurve(row.id, { colorHex: e.target.value })}
                placeholder="#RRGGBB"
                style={{ width: 110, fontFamily: FONT_MONO }}
              />
              <Tooltip title="点击后在图上选取该曲线颜色">
                <Button
                  type={pickingId === row.id ? "primary" : "default"}
                  onClick={() => setPickingId(pickingId === row.id ? null : row.id)}
                >
                  {pickingId === row.id ? "取消取色" : "图上取色"}
                </Button>
              </Tooltip>
              <Checkbox
                checked={row.isBenchmark}
                onChange={(e) => updateCurve(row.id, { isBenchmark: e.target.checked })}
              >
                基准
              </Checkbox>
              <Button danger size="small" onClick={() => removeCurve(row.id)} style={{ marginLeft: "auto" }}>
                删除
              </Button>
            </div>
          ))}

          <Space wrap style={{ marginTop: 4 }}>
            <Button onClick={addCurve}>＋ 添加曲线</Button>
          </Space>

          <div style={{ marginTop: 14 }}>
            <Text strong style={{ fontSize: 13 }}>Y 轴刻度（自下而上）</Text>
            <Input
              value={yTicksText}
              onChange={(e) => setYTicksText(e.target.value)}
              placeholder="逐个填 0.95, 1.00, 1.05…  或简写 最小, 最大, 间距（如 0.995, 1.075, 0.01）"
              style={{ marginTop: 6, fontFamily: FONT_MONO }}
              allowClear
            />
            <Paragraph type="secondary" style={{ fontSize: 12, margin: "4px 0 0" }}>
              按图上 Y 轴刻度自下而上填写（逗号分隔）；也可只填「最小值, 最大值, 间距」三个数自动展开。
              像素位置由网格线 / 等距自动定位并校准为真实净值；留空则只返回像素位置。
              {yTicksText.trim() !== "" && (
                <span style={{ color: "var(--serif-accent)" }}>
                  {" "}（当前解析为 {parseYTicksInput(yTicksText).length} 个刻度）
                </span>
              )}
            </Paragraph>
          </div>

          <Divider style={{ margin: "18px 0 12px" }} />

          <Space wrap>
            <Button
              type="primary"
              loading={isExtracting}
              onClick={() => void handleManualExtract()}
              disabled={curves.filter((c) => c.colorHex.trim()).length === 0}
            >
              按指定颜色提取（CV，零成本）
            </Button>
            <Tooltip title={vlmReady ? "用视觉模型自动识别曲线结构后追踪" : "未配置 VLM API Key"}>
              <Button
                loading={isExtracting}
                onClick={() => void handleAutoExtract()}
                disabled={!vlmReady}
              >
                VLM 自动识别并提取
              </Button>
            </Tooltip>
          </Space>
          <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 10, marginBottom: 0 }}>
            提取后可在结果页核对数值范围；如需更高精度，可填写更密的 Y 轴刻度。
          </Paragraph>
        </div>
      )}

      {/* ---------------- Step 3: results ---------------- */}
      {step === "results" && (
        <div>
          {imageFile && result && (
            <>
              <Space style={{ marginBottom: 12 }}>
                <Button size="small" onClick={() => setStep("configure")}>← 返回调整曲线</Button>
                <Text type="secondary" style={{ fontSize: 13 }}>{imageFile.name}</Text>
              </Space>
              <ResultReview
                result={result}
                imageSrc={imageUrl}
                baseName={imageFile.name.replace(/\.[^.]+$/, "")}
                onApplyToWorkspace={(text, confirmed) => {
                  onApplyToWorkspace(text, confirmed);
                  onClose();
                }}
              />
            </>
          )}

          {pdfName && (
            <div>
              <Space style={{ marginBottom: 12 }} wrap>
                <Button size="small" onClick={() => setStep("upload")}>← 重新选择</Button>
                <Text type="secondary" style={{ fontSize: 13 }}>{pdfName}</Text>
              </Space>

              {pdfJob && pdfJob.status !== "completed" && pdfJob.status !== "failed" && (
                <div style={{ padding: "24px 0", textAlign: "center" }}>
                  <Spin />
                  <Progress
                    percent={Math.round((pdfJob.progress ?? 0) * 100)}
                    size="small"
                    style={{ maxWidth: 320, margin: "12px auto 0" }}
                  />
                  <div style={{ fontSize: 12, color: "var(--serif-muted-foreground)", marginTop: 6 }}>
                    正在处理（{pdfJob.status}）…
                  </div>
                </div>
              )}

              {pdfJob?.status === "failed" && (
                <Alert type="error" showIcon message={pdfJob.error ?? "提取失败"} />
              )}

              {pdfPreview && pdfPreview.results.length > 0 && (
                <div>
                  <Paragraph type="secondary" style={{ fontSize: 13 }}>
                    共检测到 {pdfPreview.regions.length} 个图表区域，成功追踪如下。
                  </Paragraph>
                  {pdfPreview.results.map((res, index) => {
                    const region = pdfPreview.regions[index];
                    const nCurves = res.curves.length;
                    const isOpen = expandedPdfResult === index;
                    return (
                      <div
                        key={index}
                        style={{
                          border: "1px solid var(--serif-border)",
                          borderRadius: 6,
                          marginBottom: 10,
                          background: "var(--serif-card)",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 10,
                            padding: "10px 14px",
                            cursor: "pointer",
                          }}
                          onClick={() => setExpandedPdfResult(isOpen ? null : index)}
                        >
                          <Text strong>
                            第 {(region?.page_index ?? 0) + 1} 页 · 区域 {(region?.region_index ?? 0) + 1}
                          </Text>
                          <Tag>{nCurves} 条曲线</Tag>
                          <Tag color={res.error ? "warning" : "success"}>
                            {res.error ? "部分失败" : `置信 ${(res.confidence * 100).toFixed(0)}%`}
                          </Tag>
                          <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--serif-muted-foreground)" }}>
                            {isOpen ? "收起 ▴" : "展开 ▾"}
                          </span>
                        </div>
                        {isOpen && region && (
                          <div style={{ padding: "0 14px 14px" }}>
                            <ResultReview
                              result={res}
                              imageSrc={extractRegionImageUrl(
                                pdfPreview.job_id,
                                region.page_index,
                                region.region_index,
                              )}
                              baseName={`${pdfName.replace(/\.[^.]+$/, "")}_p${region.page_index + 1}r${region.region_index + 1}`}
                              onApplyToWorkspace={(text, confirmed) => {
                                onApplyToWorkspace(text, confirmed);
                                onClose();
                              }}
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {pdfPreview && pdfPreview.results.length === 0 && pdfJob?.status === "completed" && (
                <Empty description="未检测到可追踪的图表曲线" />
              )}
            </div>
          )}

          {!imageFile && !pdfName && (
            <Empty description="暂无提取结果" />
          )}
        </div>
      )}
    </Drawer>
  );
}
