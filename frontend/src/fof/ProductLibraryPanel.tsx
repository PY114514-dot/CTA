/** Left panel: batch upload queue + product library with confirmation workflow. */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Checkbox,
  Dropdown,
  Empty,
  Form,
  Input,
  List,
  Modal,
  Progress,
  Segmented,
  Select,
  Space,
  Tag,
  Tooltip,
  Typography,
  Upload,
  message,
} from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  FileOutlined,
  InboxOutlined,
  LinkOutlined,
  MoreOutlined,
  ReloadOutlined,
  SearchOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import {
  kbConfirmProduct,
  kbBindFile,
  kbBindFileCurves,
  kbDeleteFile,
  kbDeleteProduct,
  kbListFiles,
  kbListProducts,
  kbMachineReviewProducts,
  kbRejectProduct,
  kbReparseFile,
  kbUpdateProduct,
  kbUploadFile,
  type KbFile,
  type KbProduct,
} from "../api";
import { hasTraceableSource, isDemoFileName, isDemoProduct, needsManualNaming, strategyLabel } from "./productDisplay";

const { Text } = Typography;
const { Dragger } = Upload;

type ResearchFilter = "all" | "actionable" | "ready";

type UploadBatch = {
  total: number;
  uploaded: number;
  completed: number;
  failed: number;
  fileIds: string[];
};

const PARSING_TAG: Record<string, { color: string; label: string }> = {
  pending: { color: "default", label: "待解析" },
  processing: { color: "processing", label: "解析中" },
  completed: { color: "success", label: "解析完成" },
  completed_no_nav: { color: "warning", label: "待校准曲线" },
  failed: { color: "error", label: "失败" },
};

/** Product-library lanes describe data readiness, never expected performance. */
interface Props {
  refreshKey?: number;
  selectedIds: string[];
  onToggleSelect: (id: string) => void;
  onFocusProduct: (product: KbProduct) => void;
  onCalibrateProduct: (product: KbProduct, sourceFileId?: string) => void;
}

export default function ProductLibraryPanel({ refreshKey = 0, selectedIds, onToggleSelect, onFocusProduct, onCalibrateProduct }: Props): React.JSX.Element {
  const [products, setProducts] = useState<KbProduct[]>([]);
  const [files, setFiles] = useState<KbFile[]>([]);
  const [researchFilter, setResearchFilter] = useState<ResearchFilter>("all");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sortBy, setSortBy] = useState<"recent" | "name" | "status">("recent");
  const [loading, setLoading] = useState(false);
  const [uploadCount, setUploadCount] = useState(0);
  const [uploadBatch, setUploadBatch] = useState<UploadBatch | null>(null);
  const [uploadExpanded, setUploadExpanded] = useState(true);
  const stagedUploadsRef = useRef<File[]>([]);
  const uploadStartTimerRef = useRef<number | null>(null);
  const refreshRequestIdRef = useRef(0);
  const listRef = useRef<HTMLDivElement>(null);
  const initialLoadDone = useRef(false);
  const [editingProduct, setEditingProduct] = useState<KbProduct | null>(null);
  const [confirmAfterEdit, setConfirmAfterEdit] = useState(false);
  const [editForm] = Form.useForm();
  const [bindingFile, setBindingFile] = useState<KbFile | null>(null);
  const [bindingMode, setBindingMode] = useState<"existing" | "new">("existing");
  const [bindingSubmitting, setBindingSubmitting] = useState(false);
  const [curveBindingFile, setCurveBindingFile] = useState<KbFile | null>(null);
  const [curveBindingValues, setCurveBindingValues] = useState<Record<string, string>>({});
  const [machineReviewing, setMachineReviewing] = useState(false);
  const [bindingForm] = Form.useForm();

  // Debounce search input (300ms)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search), 300);
    return () => window.clearTimeout(timer);
  }, [search]);

  const refresh = useCallback(async (options?: { silent?: boolean }) => {
    const requestId = ++refreshRequestIdRef.current;
    setLoading(true);
    try {
      const params: { search?: string } = {};
      if (debouncedSearch.trim()) params.search = debouncedSearch.trim();
      const [productList, fileList] = await Promise.all([kbListProducts(params), kbListFiles()]);
      if (requestId !== refreshRequestIdRef.current) return;
      setProducts(productList);
      setFiles(fileList);
      if (!initialLoadDone.current) {
        initialLoadDone.current = true;
        if (fileList.some((file) => !isDemoFileName(file.filename))) setUploadExpanded(false);
      }
    } catch {
      if (requestId !== refreshRequestIdRef.current) return;
      if (!options?.silent) message.error("加载产品库失败");
    } finally {
      if (requestId === refreshRequestIdRef.current) setLoading(false);
    }
  }, [debouncedSearch]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { if (refreshKey > 0) void refresh({ silent: true }); }, [refresh, refreshKey]);

  // The single-product calibration workspace is a sibling view, so refresh
  // the list immediately after a NAV review is saved there.
  useEffect(() => {
    const handler = () => { void refresh({ silent: true }); };
    window.addEventListener("kb:product-updated", handler);
    return () => window.removeEventListener("kb:product-updated", handler);
  }, [refresh]);

  useEffect(() => {
    if (!files.some((file) => file.parsing_status === "processing")) return;
    const timer = window.setInterval(() => { void refresh({ silent: true }); }, 3000);
    return () => window.clearInterval(timer);
  }, [files, refresh]);

  // Reflect the server-side parser state in the upload batch.  Uploading a
  // file only means it reached storage; a batch is complete only when each
  // source has a terminal parsing result.
  useEffect(() => {
    if (!uploadBatch?.fileIds.length) return;
    const batchFiles = files.filter((file) => uploadBatch.fileIds.includes(file.id));
    const completed = batchFiles.filter((file) => file.parsing_status === "completed" || file.parsing_status === "completed_no_nav").length;
    const failed = batchFiles.filter((file) => file.parsing_status === "failed").length;
    if (completed !== uploadBatch.completed || failed !== uploadBatch.failed) {
      setUploadBatch((previous) => previous ? { ...previous, completed, failed } : previous);
    }
  }, [files, uploadBatch?.fileIds, uploadBatch?.completed, uploadBatch?.failed]);

  // Demo records may remain in a local development database, but must never
  // appear in the formal workbench or be selectable for research.
  const visibleProducts = React.useMemo(
    // A product card is a reviewable research object, never an untraceable
    // NAV row.  Records without an original file remain out of the visible
    // workbench until a source is uploaded and bound to them.
    () => {
      const query = search.trim().toLocaleLowerCase("zh-CN");
      return products.filter((product) => (
        !isDemoProduct(product)
        && hasTraceableSource(product)
        && (!query || product.standard_name.toLocaleLowerCase("zh-CN").includes(query))
      ));
    },
    [products, search],
  );
  const filenameById = React.useMemo(
    () => new Map(files.map((file) => [file.id, file.filename])),
    [files],
  );
  const visibleFiles = React.useMemo(
    () => files.filter((file) => !isDemoFileName(file.filename)),
    [files],
  );
  // The queue is action-oriented: keep a bound source visible while parsing,
  // calibration or curve ownership still needs attention.
  const queueFiles = React.useMemo(
    () => visibleFiles.filter((file) => (
      !(file.linked_product_ids?.length)
      || file.parsing_status === "pending"
      || file.parsing_status === "processing"
      || file.parsing_status === "failed"
      || file.parsing_status === "completed_no_nav"
      || (file.unbound_curve_count ?? 0) > 0
      || (file.unbound_fragment_count ?? 0) > 0
    )),
    [visibleFiles],
  );

  const researchFilteredProducts = React.useMemo(
    () => researchFilter === "all"
      ? visibleProducts
      : researchFilter === "ready"
        ? visibleProducts.filter((product) => product.research_ready)
        : visibleProducts.filter((product) => !product.research_ready),
    [researchFilter, visibleProducts],
  );

  const sortedProducts = React.useMemo(() => {
    const STATUS_ORDER: Record<string, number> = { pending: 0, confirmed: 1, rejected: 2 };
    const arr = [...researchFilteredProducts];
    if (sortBy === "name") arr.sort((a, b) => a.standard_name.localeCompare(b.standard_name, "zh"));
    else if (sortBy === "status") arr.sort((a, b) => (STATUS_ORDER[a.confirmation_status] ?? 9) - (STATUS_ORDER[b.confirmation_status] ?? 9));
    else arr.sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
    return arr;
  }, [researchFilteredProducts, sortBy]);

  const uploadFiles = useCallback(async (batch: File[]) => {
    if (batch.length === 0) return;
    setUploadCount(batch.length);
    setUploadExpanded(true);
    setUploadBatch({ total: batch.length, uploaded: 0, completed: 0, failed: 0, fileIds: [] });

    // Two concurrent uploads keep the UI responsive and avoid saturating the
    // local VLM/CV parser with a large browser selection.
    let nextIndex = 0;
    const worker = async () => {
      while (nextIndex < batch.length) {
        const file = batch[nextIndex++];
        if (!file) continue;
        try {
          const result = await kbUploadFile(file);
          setUploadBatch((previous) => previous ? {
            ...previous,
            uploaded: previous.uploaded + 1,
            fileIds: [...previous.fileIds, result.id],
          } : previous);
        } catch {
          setUploadBatch((previous) => previous ? {
            ...previous,
            uploaded: previous.uploaded + 1,
            failed: previous.failed + 1,
          } : previous);
          message.error(`${file.name} 上传失败`);
        } finally {
          setUploadCount((count) => Math.max(0, count - 1));
          void refresh({ silent: true });
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(2, batch.length) }, worker));
    message.success(`已提交 ${batch.length} 份资料，正在后台识别`);
  }, [refresh]);

  const stageUpload = useCallback((file: File) => {
    stagedUploadsRef.current.push(file);
    if (uploadStartTimerRef.current !== null) window.clearTimeout(uploadStartTimerRef.current);
    // Ant Design calls beforeUpload once per selected file.  Collect this
    // synchronous burst so click-select and drag-drop share one batch.
    uploadStartTimerRef.current = window.setTimeout(() => {
      const batch = stagedUploadsRef.current.splice(0);
      uploadStartTimerRef.current = null;
      void uploadFiles(batch);
    }, 50);
  }, [uploadFiles]);

  const handleReparse = useCallback(async (fileId: string) => {
    try {
      await kbReparseFile(fileId);
      message.success("已重新提交解析");
      void refresh();
    } catch {
      message.error("重新解析失败，请重新上传");
    }
  }, [refresh]);

  const handleDeleteFile = useCallback((file: KbFile) => {
    Modal.confirm({
      title: "删除文件",
      content: `确定删除「${file.filename}」吗？已提取的片段将一并移除。`,
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: async () => {
        try {
          await kbDeleteFile(file.id);
          message.success("文件已删除");
          void refresh();
        } catch {
          message.error("删除文件失败");
        }
      },
    });
  }, [refresh]);

  const openBinding = useCallback((file: KbFile) => {
    const hasExistingProduct = visibleProducts.some((product) => product.confirmation_status !== "rejected");
    bindingForm.resetFields();
    setBindingMode(hasExistingProduct ? "existing" : "new");
    setBindingFile(file);
  }, [bindingForm, visibleProducts]);

  const handleBindFile = useCallback(async () => {
    if (!bindingFile) return;
    setBindingSubmitting(true);
    try {
      const values = await bindingForm.validateFields();
      const payload = bindingMode === "existing"
        ? { product_id: values.product_id as string }
        : {
            product_name: (values.product_name as string).trim(),
            manager_name: (values.manager_name as string | undefined)?.trim() || undefined,
            strategy: (values.strategy as string | undefined)?.trim() || undefined,
          };
      const result = await kbBindFile(bindingFile.id, payload);
      message.success(`已绑定到「${result.product_name}」`);
      setBindingFile(null);
      bindingForm.resetFields();
      void refresh();
    } catch (error) {
      if (error instanceof Error) message.error(error.message || "绑定产品失败");
    } finally {
      setBindingSubmitting(false);
    }
  }, [bindingFile, bindingForm, bindingMode, refresh]);

  const openCurveBinding = useCallback((file: KbFile) => {
    const initial: Record<string, string> = {};
    (file.curve_bindings ?? []).filter((item) => item.binding_status !== "matched").forEach((item) => {
      if (item.product_id) initial[item.fragment_id] = item.product_id;
    });
    setCurveBindingValues(initial);
    setCurveBindingFile(file);
  }, []);

  const handleBindCurves = useCallback(async () => {
    if (!curveBindingFile) return;
    const bindings = (curveBindingFile.curve_bindings ?? [])
      .filter((item) => item.binding_status !== "matched" && curveBindingValues[item.fragment_id])
      .map((item) => ({ fragment_id: item.fragment_id, product_id: curveBindingValues[item.fragment_id]! }));
    if (!bindings.length) {
      message.warning("请至少为一条曲线选择产品");
      return;
    }
    setBindingSubmitting(true);
    try {
      const result = await kbBindFileCurves(curveBindingFile.id, bindings);
      message.success(`已确认 ${result.bound_curves} 条曲线归属`);
      setCurveBindingFile(null);
      void refresh();
    } catch (error) {
      message.error(error instanceof Error ? error.message : "批量绑定曲线失败");
    } finally {
      setBindingSubmitting(false);
    }
  }, [curveBindingFile, curveBindingValues, refresh]);

  const handleDelete = useCallback((product: KbProduct) => {
    Modal.confirm({
      title: "删除产品",
      content: `确定删除「${product.standard_name}」吗？其净值数据与指标将一并移除，来源文件资料保留。`,
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: async () => {
        try {
          await kbDeleteProduct(product.id);
          message.success("已删除");
          void refresh();
        } catch {
          message.error("删除失败");
        }
      },
    });
  }, [refresh]);

  const openEdit = useCallback((product: KbProduct, confirmAfter = false) => {
    editForm.setFieldsValue({
      standard_name: product.standard_name,
      manager_name: product.manager_name ?? "",
      strategy: product.strategy ?? "",
    });
    setConfirmAfterEdit(confirmAfter);
    setEditingProduct(product);
  }, [editForm]);

  const handleEditSave = useCallback(async () => {
    if (!editingProduct) return;
    try {
      const values = await editForm.validateFields();
      await kbUpdateProduct(editingProduct.id, {
        standard_name: (values.standard_name as string).trim(),
        manager_name: (values.manager_name as string)?.trim() || null,
        strategy: (values.strategy as string)?.trim() || null,
      });
      if (confirmAfterEdit) await kbConfirmProduct(editingProduct.id);
      message.success(confirmAfterEdit ? "已保存并确认" : "已保存");
      setEditingProduct(null);
      void refresh();
    } catch (error) {
      if (error instanceof Error) message.error("保存失败");
    }
  }, [editingProduct, editForm, refresh, confirmAfterEdit]);

  const handleConfirm = useCallback(async (id: string) => {
    try {
      await kbConfirmProduct(id);
      message.success("已确认");
      void refresh();
    } catch {
      message.error("确认失败");
    }
  }, [refresh]);

  const handleReject = useCallback(async (id: string) => {
    const product = visibleProducts.find((item) => item.id === id);
    Modal.confirm({
      title: "标记为非产品",
      content: `确定将「${product?.standard_name ?? "该记录"}」标记为非产品吗？来源资料会保留，但不会参与 Agent 研究。`,
      okText: "标记为非产品",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: async () => {
        try {
          await kbRejectProduct(id);
          message.success("已标记为非产品");
          void refresh();
        } catch {
          message.error("操作失败");
        }
      },
    });
  }, [refresh, visibleProducts]);

  const handleMachineReview = useCallback(async () => {
    const candidates = visibleProducts
      .filter((product) => product.confirmation_status !== "rejected" && product.nav_count >= 2 && product.reviewed_nav_count < product.nav_count)
      .map((product) => product.id);
    if (candidates.length === 0) {
      message.info("没有待机器复核的净值曲线");
      return;
    }
    setMachineReviewing(true);
    try {
      const outcome = await kbMachineReviewProducts(candidates);
      message.info(`机器核验：${outcome.machine_reviewed} 个已标记为“机器核验通过”，仍需人工复核；${outcome.human_review_required} 个需要进一步处理`);
      await refresh({ silent: true });
    } catch (error) {
      message.error(error instanceof Error ? error.message : "机器复核失败");
    } finally {
      setMachineReviewing(false);
    }
  }, [refresh, visibleProducts]);

  const pendingCount = visibleProducts.filter((p) => p.confirmation_status === "pending").length;
  const pendingFileCount = queueFiles.filter((f) => f.parsing_status === "pending" || f.parsing_status === "processing").length;
  const isResearchReady = (product: KbProduct) => product.research_ready ?? (
    !needsManualNaming(product)
    && product.nav_count >= 2
    && product.confirmation_status === "confirmed"
    && (product.source_file_ids?.length ?? 0) > 0
    && product.reviewed_nav_count >= product.nav_count
  );
  const bindingProducts = visibleProducts.filter((product) => product.confirmation_status !== "rejected");
  const readyCount = visibleProducts.filter(isResearchReady).length;
  const actionableCount = Math.max(0, visibleProducts.length - readyCount);
  const needsNamingCount = visibleProducts.filter(needsManualNaming).length;
  const reviewCandidateFileCount = queueFiles.filter((file) => file.workflow?.next_action === "review_candidate").length;
  const bindingTaskCount = queueFiles.filter((file) => file.workflow?.next_action === "confirm_product_binding").length;
  const calibrationTaskCount = queueFiles.filter((file) => file.workflow?.next_action === "recover_chart_calibration").length;
  // An uploaded file is not automatically a research-ready product. Keep the
  // compact summary explicit so “可研究 0” is not mistaken for a failed upload.
  const uploadSummary = queueFiles.length === 0
    ? "暂无待处理资料 · 已绑定资料已归档到产品"
    : readyCount > 0
    ? `待处理 ${queueFiles.length} 份资料 · 可研究 ${readyCount} 个产品`
    : pendingFileCount > 0
      ? `待处理 ${queueFiles.length} 份资料 · ${pendingFileCount} 份正在解析`
      : needsNamingCount > 0
        ? `待处理 ${queueFiles.length} 份资料 · ${needsNamingCount} 个产品待命名`
        : visibleProducts.length > 0
          ? `待处理 ${queueFiles.length} 份资料 · ${visibleProducts.length} 个产品待复核`
          : `待处理 ${queueFiles.length} 份资料 · 尚未形成可研究产品`;
  const batchTerminalCount = uploadBatch ? uploadBatch.completed + uploadBatch.failed : 0;
  const batchPercent = uploadBatch ? Math.round((batchTerminalCount / uploadBatch.total) * 100) : 0;
  const batchActiveCount = uploadBatch ? Math.max(0, uploadBatch.uploaded - batchTerminalCount) : 0;
  const batchNavExtractedCount = uploadBatch
    ? files.filter((file) => uploadBatch.fileIds.includes(file.id) && (file.nav_count ?? 0) >= 2).length
    : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 10, overflow: "hidden" }}>
      {/* Upload area */}
      <Card size="small" style={{ flexShrink: 0 }} styles={{ body: { padding: queueFiles.length > 0 && !uploadExpanded ? "6px 8px" : 10 } }}>
        {queueFiles.length > 0 && !uploadExpanded ? (
          <div style={{ display: "flex", alignItems: "center", gap: 8, minHeight: 24 }}>
            <FileOutlined style={{ color: "var(--serif-accent, #1677ff)" }} />
            <Text ellipsis style={{ flex: 1, fontSize: 12 }}>{uploadSummary}</Text>
            <Button size="small" type="text" icon={<UploadOutlined />} onClick={() => setUploadExpanded(true)}>上传</Button>
          </div>
        ) : <>
          <Dragger
            multiple
            showUploadList={false}
            beforeUpload={(file) => { stageUpload(file); return false; }}
            disabled={uploadCount > 0}
            style={{ padding: "6px 0" }}
          >
            <p className="ant-upload-drag-icon" style={{ marginBottom: 2 }}>
              <InboxOutlined style={{ color: "var(--serif-accent, #1677ff)" }} />
            </p>
            <p style={{ margin: 0, fontSize: 12, color: "var(--serif-text-secondary, #666)" }}>
              {uploadCount > 0 ? `上传中 (${uploadCount})...` : "拖拽或点击上传 PDF / 图片 / XLSX / CSV / DOCX / PPTX"}
            </p>
          </Dragger>
          {uploadBatch && (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, fontSize: 11 }}>
                <Text style={{ fontSize: 11 }}>
                  本批 {uploadBatch.total} 份：已上传 {uploadBatch.uploaded}/{uploadBatch.total}
                  {batchActiveCount > 0 ? ` · 识别中 ${batchActiveCount}` : ""}
                  {uploadBatch.completed > 0 ? ` · 已完成 ${uploadBatch.completed}` : ""}
                  {uploadBatch.failed > 0 ? ` · 失败 ${uploadBatch.failed}` : ""}
                  {batchTerminalCount > 0 ? ` · 净值已提取 ${batchNavExtractedCount}` : ""}
                </Text>
                {batchTerminalCount === uploadBatch.total && <Tag color={uploadBatch.failed ? "warning" : "success"} style={{ margin: 0, fontSize: 10 }}>
                  {uploadBatch.failed ? "解析完成，含失败" : "解析完成"}
                </Tag>}
              </div>
              <Progress
                percent={batchPercent}
                size="small"
                status={uploadBatch.failed > 0 ? "exception" : batchTerminalCount === uploadBatch.total ? "success" : "active"}
                showInfo={false}
                style={{ margin: "3px 0 0" }}
              />
            </div>
          )}
          {queueFiles.length > 0 && (
            <div style={{ marginTop: 6, maxHeight: 88, overflow: "auto" }}>
              <div style={{ display: "flex", gap: 5, marginBottom: 4 }}>
                <Tag color="success" style={{ margin: 0, fontSize: 10 }}>可研究 {readyCount}</Tag>
                <Tag color="processing" style={{ margin: 0, fontSize: 10 }}>待确认 {pendingCount}</Tag>
                 {needsNamingCount > 0 && <Tag color="warning" style={{ margin: 0, fontSize: 10 }}>待命名 {needsNamingCount}</Tag>}
                 {reviewCandidateFileCount > 0 && <Tag color="success" style={{ margin: 0, fontSize: 10 }}>审核候选 {reviewCandidateFileCount}</Tag>}
                 {bindingTaskCount > 0 && <Tag color="warning" style={{ margin: 0, fontSize: 10 }}>确认绑定 {bindingTaskCount}</Tag>}
                 {calibrationTaskCount > 0 && <Tag color="processing" style={{ margin: 0, fontSize: 10 }}>校准恢复 {calibrationTaskCount}</Tag>}
              </div>
              {queueFiles.map((f) => {
                const tag = PARSING_TAG[f.parsing_status] ?? { color: "default", label: "未知" };
                const linkedProduct = f.linked_product_ids?.[0]
                  ? products.find((item) => item.id === f.linked_product_ids?.[0])
                  : undefined;
                const needsBinding = !f.linked_product_ids?.length;
                const needsCurveBinding = !needsBinding && (f.unbound_curve_count ?? 0) > 0;
                const needsCalibration = f.parsing_status === "completed_no_nav" && linkedProduct;
                const workflowAction = f.workflow?.next_action;
                const filePrimaryAction = workflowAction === "confirm_product_binding"
                  ? { label: "确认绑定", run: () => openBinding(f) }
                  : workflowAction === "recover_chart_calibration" && linkedProduct
                    ? { label: "恢复校准", run: () => onCalibrateProduct(linkedProduct, f.id) }
                    : needsBinding
                  ? { label: "绑定产品", run: () => openBinding(f) }
                  : needsCurveBinding
                    ? { label: "确认曲线", run: () => openCurveBinding(f) }
                    : needsCalibration
                      ? { label: "校准净值", run: () => onCalibrateProduct(linkedProduct, f.id) }
                      : f.parsing_status === "failed"
                        ? { label: "重新解析", run: () => void handleReparse(f.id) }
                        : null;
                return (
                  <div key={f.id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, padding: "2px 0" }}>
                    <FileOutlined style={{ flexShrink: 0 }} />
                    <Text ellipsis style={{ flex: 1, fontSize: 11 }}>{f.filename}</Text>
                    <Tag color={tag.color} style={{ margin: 0, fontSize: 10, lineHeight: "16px" }}>{tag.label}</Tag>
                    {f.parsing_error && (
                      <Tooltip title={f.parsing_error}>
                        <Text type="warning" style={{ fontSize: 10, flexShrink: 0 }}>原因</Text>
                      </Tooltip>
                    )}
                    <Text type="secondary" ellipsis style={{ maxWidth: 140, fontSize: 10 }}>
                      {f.linked_product_names?.length ? f.linked_product_names.join("、") : "尚未绑定产品"}
                    </Text>
                    {(f.unbound_curve_count ?? 0) > 0 && (
                      <Tooltip title={
                        <div>
                          <div>曲线未按位置自动绑定；请根据图例或原图确认。</div>
                          {(f.curve_bindings ?? []).filter((item) => item.binding_status !== "matched").map((item) => (
                            <div key={item.fragment_id}>
                              曲线 {item.curve_index} · {item.legend_label || "无图例"} · 置信度 {Math.round((item.binding_confidence ?? 0) * 100)}%
                            </div>
                          ))}
                        </div>
                      }>
                        <Tag color="warning" style={{ margin: 0, fontSize: 10, lineHeight: "16px" }}>待绑定 {f.unbound_curve_count}</Tag>
                      </Tooltip>
                    )}
                    {filePrimaryAction && (
                      <Button type="link" size="small" icon={needsBinding ? <LinkOutlined /> : undefined} style={{ padding: 0, height: 24, fontSize: 11, flexShrink: 0 }} onClick={filePrimaryAction.run}>
                        {filePrimaryAction.label}
                      </Button>
                    )}
                    <Dropdown trigger={["click"]} menu={{ items: [
                      ...(filePrimaryAction?.label !== "重新解析" ? [{ key: "reparse", icon: <ReloadOutlined />, label: "重新解析", onClick: () => void handleReparse(f.id) }] : []),
                      { key: "delete", icon: <DeleteOutlined />, label: "删除资料", danger: true, onClick: () => handleDeleteFile(f) },
                    ] }}>
                      <Button type="text" size="small" icon={<MoreOutlined />} aria-label={`${f.filename} 更多操作`} style={{ width: 24, height: 24, minWidth: 24 }} />
                    </Dropdown>
                  </div>
                );
              })}
              <Button type="link" size="small" style={{ padding: 0, height: 18, fontSize: 11 }} onClick={() => setUploadExpanded(false)}>收起资料</Button>
            </div>
          )}
        </>}
      </Card>

      <Modal
        title={curveBindingFile ? `分别绑定曲线 · ${curveBindingFile.filename}` : "分别绑定曲线"}
        open={curveBindingFile !== null}
        onCancel={() => { if (!bindingSubmitting) setCurveBindingFile(null); }}
        onOk={() => void handleBindCurves()}
        okText="确认所选归属"
        cancelText="取消"
        confirmLoading={bindingSubmitting}
        width={640}
      >
        <Text type="secondary" style={{ fontSize: 12, display: "block", marginBottom: 12 }}>
          系统不会按曲线位置猜测归属。仅确认您选中的曲线；未选择的仍保持待绑定。
        </Text>
        <Space direction="vertical" size={10} style={{ width: "100%" }}>
          {(curveBindingFile?.curve_bindings ?? []).filter((item) => item.binding_status !== "matched").map((item) => (
            <Card key={item.fragment_id} size="small" styles={{ body: { padding: "8px 10px" } }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Tag color="warning" style={{ margin: 0 }}>曲线 {item.curve_index}</Tag>
                <Text style={{ flex: 1 }} ellipsis>{item.legend_label || "未读取到图例名称"}</Text>
                <Text type="secondary" style={{ fontSize: 11 }}>识别 {Math.round((item.binding_confidence ?? 0) * 100)}%</Text>
              </div>
              <Select
                allowClear
                placeholder="选择该曲线对应的产品"
                value={curveBindingValues[item.fragment_id]}
                onChange={(value) => setCurveBindingValues((current) => ({ ...current, [item.fragment_id]: value }))}
                style={{ width: "100%", marginTop: 8 }}
                options={bindingProducts.map((product) => ({ value: product.id, label: product.standard_name }))}
              />
              {!!item.binding_evidence?.length && <Text type="secondary" style={{ fontSize: 11 }}>{item.binding_evidence.join("；")}</Text>}
            </Card>
          ))}
        </Space>
      </Modal>

      {/* Filter + search */}
      <div style={{ flexShrink: 0, display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
          <Text strong>产品库</Text>
          <Dropdown trigger={["click"]} menu={{ items: [
            { key: "machine-review", label: machineReviewing ? "正在自动检查…" : "自动检查待复核曲线", disabled: machineReviewing, onClick: () => void handleMachineReview() },
            { key: "refresh", icon: <ReloadOutlined />, label: "刷新产品库", onClick: () => void refresh() },
          ] }}>
            <Button size="small" type="text" icon={<MoreOutlined />} aria-label="产品库更多操作" />
          </Dropdown>
        </div>
        <Segmented
          block
          size="small"
          value={researchFilter}
          onChange={(value) => setResearchFilter(value as ResearchFilter)}
          options={[
            { value: "all", label: "全部" },
            { value: "actionable", label: <Badge count={actionableCount} size="small" offset={[7, 0]}><span>待处理</span></Badge> },
            { value: "ready", label: <Badge count={readyCount} size="small" offset={[7, 0]}><span>可研究</span></Badge> },
          ]}
        />
        <div style={{ display: "flex", gap: 6 }}>
          <Input
            size="small"
            placeholder="搜索产品名称..."
            prefix={<SearchOutlined />}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            allowClear
            style={{ flex: 1 }}
          />
          <Select
            size="small"
            value={sortBy}
            onChange={setSortBy}
            style={{ width: 76 }}
            options={[
              { value: "recent", label: "最新" },
              { value: "name", label: "名称" },
              { value: "status", label: "状态" },
            ]}
          />
        </div>
      </div>

      {/* Product list */}
      <div ref={listRef} style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
        {sortedProducts.length === 0 && !loading ? (
          <Empty description={researchFilter === "all" ? "暂无产品" : "当前状态下暂无产品"} image={Empty.PRESENTED_IMAGE_SIMPLE} style={{ marginTop: 40 }} />
        ) : (
          <List
            size="small"
            loading={loading}
            dataSource={sortedProducts}
            renderItem={(product) => {
              const isSelected = selectedIds.includes(product.id);
              const requiresNaming = needsManualNaming(product);
              const canResearch = isResearchReady(product);
              const reviewConfidence = product.nav_count > 0 ? Math.round(product.reviewed_nav_count / product.nav_count * 100) : 0;
              const navReviewComplete = product.nav_count >= 2 && product.reviewed_nav_count === product.nav_count;
              const extractionConfidence = product.nav_confidence;
              const confidenceBand = product.nav_confidence_band
                ?? (extractionConfidence == null ? "unknown" : extractionConfidence >= 0.85 ? "high" : extractionConfidence >= 0.60 ? "medium" : "low");
              const confidenceLabel = confidenceBand === "high" ? "高" : confidenceBand === "medium" ? "中" : confidenceBand === "low" ? "低" : "未评估";
              const displayStrategy = strategyLabel(product.strategy);
              const dataStatus = requiresNaming
                ? { color: "warning", label: "待人工命名" }
                : navReviewComplete
                  ? { color: "success", label: "净值已复核" }
                : product.nav_count >= 2 && product.fact_count > 0
                  ? { color: "success", label: "可研究" }
                  : product.nav_count >= 2
                    ? { color: "processing", label: "待复核净值" }
                    : product.fact_count > 0
                      ? { color: "processing", label: "待补净值" }
                      : { color: "default", label: "待补资料" };
              const resolvedDataStatus = product.nav_quality?.blocking
                ? { color: "error", label: "曲线与披露冲突" }
                : product.research_ready
                ? { color: "success", label: "可研究" }
                : navReviewComplete
                ? { color: "success", label: "净值已复核" }
                : product.readiness_reason
                  ? { color: "processing", label: product.readiness_reason }
                  : dataStatus;
              const displayStatus = product.confirmation_status === "pending" && navReviewComplete
                ? { color: "processing", label: "待确认产品" }
                : resolvedDataStatus;
              const needsNavReview = !requiresNaming
                && product.confirmation_status !== "rejected"
                && (product.nav_count < 2 || reviewConfidence < 100 || product.nav_quality?.blocking);
              const primaryAction = requiresNaming
                ? { label: "完善产品信息", run: () => openEdit(product, true) }
                : needsNavReview
                  ? { label: product.nav_count < 2 ? "提取 / 补录净值" : "复核净值", run: () => onCalibrateProduct(product) }
                  : product.confirmation_status === "pending"
                    ? { label: "确认产品", run: () => void handleConfirm(product.id) }
                    : null;
              return (
                <div
                  key={product.id}
                  style={{
                    padding: "5px 10px",
                    borderRadius: 6,
                    border: `1px solid ${isSelected ? "var(--serif-accent, #1677ff)" : "var(--serif-border, #f0f0f0)"}`,
                    marginBottom: 4,
                    background: isSelected ? "var(--serif-accent-bg, #f0f5ff)" : "var(--serif-card-bg, #fff)",
                    cursor: "pointer",
                    transition: "border-color 0.2s",
                  }}
                  onClick={() => {
                    if (requiresNaming) {
                      message.info("该记录尚未形成可靠产品名称，请右键选择“编辑信息”后再加入候选池。");
                      return;
                    }
                    if (product.confirmation_status === "rejected") {
                      message.info("该记录已标为不是产品，不会进入 Agent 研究上下文。");
                      return;
                    }
                    if (!canResearch) {
                      message.info(product.readiness_reason ?? "请先完成产品身份与净值复核");
                      return;
                    }
                    onFocusProduct(product); onToggleSelect(product.id);
                  }}
                >
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                    <Checkbox
                      checked={isSelected}
                      disabled={!canResearch}
                      style={{ marginTop: 2 }}
                      onClick={(event) => event.stopPropagation()}
                      onChange={() => { onFocusProduct(product); onToggleSelect(product.id); }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <Text strong ellipsis style={{ flex: 1, fontSize: 13 }}>{product.standard_name}</Text>
                        <Tag color={displayStatus.color} title={extractionConfidence != null ? `识别置信度 ${(extractionConfidence * 100).toFixed(0)}%（${confidenceLabel}）` : "尚未获得识别置信度"} style={{ margin: 0, fontSize: 10, lineHeight: "16px" }}>{displayStatus.label}</Tag>
                        <Dropdown trigger={["click"]} menu={{ items: [
                          { key: "edit", icon: <EditOutlined />, label: "编辑信息", onClick: () => openEdit(product) },
                          ...(product.confirmation_status !== "rejected" ? [{ key: "reject", label: "标记为非产品", danger: true, onClick: () => void handleReject(product.id) }] : []),
                          { key: "delete", icon: <DeleteOutlined />, label: "删除产品", danger: true, onClick: () => handleDelete(product) },
                        ] }}>
                          <Button type="text" size="small" icon={<MoreOutlined />} aria-label={`${product.standard_name} 更多操作`} style={{ width: 24, height: 24, minWidth: 24 }} onClick={(event) => event.stopPropagation()} />
                        </Dropdown>
                      </div>
                      <div style={{ marginTop: 2, display: "flex", gap: 8, fontSize: 11, color: "var(--serif-text-secondary, #888)" }}>
                        {product.manager_name && <span>{product.manager_name}</span>}
                        {displayStrategy && <span>{displayStrategy}</span>}
                        {product.nav_methods?.[0] && <span title={product.nav_methods[0]}>来源：{product.nav_methods[0]}</span>}
                      </div>
                      {requiresNaming && <Text type="secondary" style={{ display: "block", marginTop: 2, fontSize: 11 }}>识别文本已保留，需核对并编辑正式名称</Text>}
                      {!requiresNaming && <Text type="secondary" style={{ display: "block", marginTop: 2, fontSize: 11 }}>
                        净值 {product.nav_count} 条 · 复核 {reviewConfidence}% · 识别置信 {extractionConfidence == null ? "未评估" : `${(extractionConfidence * 100).toFixed(0)}%（${confidenceLabel}）`} · 证据 {product.fact_count} 条
                      </Text>}
                      {!requiresNaming && product.machine_nav_review && (
                        <Text type={product.machine_nav_review.machine_reviewed ? "success" : "warning"} style={{ display: "block", marginTop: 2, fontSize: 11 }}>
                          {product.machine_nav_review.machine_reviewed
                            ? "机器复核通过：可进入初步研究（非人工事实核验）"
                            : `需人工处理：${product.machine_nav_review.reasons[0] ?? "机器复核未通过"}`}
                        </Text>
                      )}
                      {!requiresNaming && confidenceBand === "low" && reviewConfidence < 100 && (
                        <Text type="warning" style={{ display: "block", marginTop: 2, fontSize: 11 }}>识别置信度较低，必须复核后才能研究</Text>
                      )}
                      {product.confirmation_status === "pending" && !requiresNaming && (
                        <div style={{ marginTop: 3, fontSize: 11, lineHeight: 1.5, color: "var(--serif-text-secondary, #888)" }}>
                          <div>识别范围：{product.nav_start && product.nav_end ? `${product.nav_start} 至 ${product.nav_end}` : "未识别日期"} · {product.reviewed_nav_count}/{product.nav_count} 已复核</div>
                          <div>来源：{(() => {
                            const filenames = product.source_files.length > 0
                              ? product.source_files
                              : (product.source_file_ids ?? []).map((id) => filenameById.get(id)).filter((name): name is string => Boolean(name));
                            return filenames.length > 0 ? filenames.join("、") : "未关联原始文件";
                          })()}</div>
                        </div>
                      )}
                      {primaryAction && (
                        <Button type="primary" size="small" block style={{ marginTop: 7 }} onClick={(event) => { event.stopPropagation(); primaryAction.run(); }}>
                          {primaryAction.label}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              );
            }}
          />
        )}
      </div>

      {/* Footer / batch actions */}
      {selectedIds.length > 0 ? (
        <div style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 8, padding: "4px 0", minWidth: 0 }}>
          <Text style={{ fontSize: 11, flex: 1, minWidth: 0 }}>已加入对话上下文：{selectedIds.length} 个产品</Text>
          <Button type="link" size="small" style={{ paddingInline: 0 }} onClick={() => selectedIds.forEach((id) => onToggleSelect(id))}>清空选择</Button>
        </div>
      ) : (
        <Text type="secondary" style={{ flexShrink: 0, fontSize: 11 }}>{sortedProducts.length} 个产品 · 点击可研究产品即可加入对话</Text>
      )}

      <Modal
        title={bindingFile ? `绑定产品 · ${bindingFile.filename}` : "绑定产品"}
        open={bindingFile !== null}
        onOk={() => void handleBindFile()}
        onCancel={() => { if (!bindingSubmitting) { setBindingFile(null); bindingForm.resetFields(); } }}
        okText="确认绑定"
        cancelText="取消"
        confirmLoading={bindingSubmitting}
        destroyOnClose
      >
        <Segmented
          block
          value={bindingMode}
          onChange={(value) => {
            bindingForm.resetFields();
            setBindingMode(value as "existing" | "new");
          }}
          options={[
            { value: "existing", label: "绑定已有产品" },
            { value: "new", label: "创建待确认产品" },
          ]}
          style={{ marginBottom: 14 }}
        />
        <Text type="secondary" style={{ display: "block", marginBottom: 12, fontSize: 12 }}>
          绑定只会处理这份文件中尚未确认归属的片段，不会覆盖已经绑定到其他产品的证据。
        </Text>
        <Form form={bindingForm} layout="vertical" preserve={false}>
          {bindingMode === "existing" ? (
            <Form.Item name="product_id" label="目标产品" rules={[{ required: true, message: "请选择目标产品" }]}>
              <Select
                showSearch
                optionFilterProp="label"
                placeholder={bindingProducts.length ? "选择产品" : "暂无可用产品，请切换为创建待确认产品"}
                options={bindingProducts.map((product) => ({
                  value: product.id,
                  label: `${product.standard_name}${product.confirmation_status === "pending" ? "（待确认）" : ""}`,
                }))}
              />
            </Form.Item>
          ) : (
            <>
              <Form.Item name="product_name" label="产品名称" rules={[{ required: true, whitespace: true, message: "请输入产品名称" }]}>
                <Input placeholder="例如：致远2号" />
              </Form.Item>
              <Form.Item name="manager_name" label="管理人（可选）">
                <Input placeholder="公司或管理人名称" />
              </Form.Item>
              <Form.Item name="strategy" label="策略（可选）">
                <Input placeholder="先留空，确认产品时再补充也可以" />
              </Form.Item>
            </>
          )}
        </Form>
      </Modal>

      <Modal
        title={confirmAfterEdit ? "编辑并确认产品" : "编辑产品信息"}
        open={editingProduct !== null}
        onOk={() => void handleEditSave()}
        onCancel={() => setEditingProduct(null)}
        okText={confirmAfterEdit ? "保存并确认" : "保存"}
        cancelText="取消"
        destroyOnClose
      >
        <Form form={editForm} layout="vertical" preserve={false}>
          <Form.Item name="standard_name" label="产品名称" rules={[{ required: true, message: "请输入产品名称" }]}>
            <Input placeholder="产品全称" />
          </Form.Item>
          <Form.Item name="manager_name" label="管理人">
            <Input placeholder="管理人 / 公司（可留空）" />
          </Form.Item>
          <Form.Item name="strategy" label="策略">
            <Input placeholder="如 商品 CTA、趋势 CTA（可留空）" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
