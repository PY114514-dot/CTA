/** Left panel: product library with a compact material intake entry. */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Card,
  Dropdown,
  Empty,
  Form,
  Input,
  List,
  Modal,
  Segmented,
  Select,
  Space,
  Tag,
  Typography,
  message,
} from "antd";
import {
  MoreOutlined,
  ReloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import {
  kbConfirmProduct,
  kbBindFile,
  kbBindFileCurves,
  kbCreateProduct,
  kbDeleteFile,
  kbDeleteProduct,
  kbGetIngestionQueue,
  kbGetProduct,
  kbListFiles,
  kbListProducts,
  kbMachineReviewProducts,
  kbRejectProduct,
  kbReparseFile,
  kbUpdateProduct,
  kbUpdateFileMaterialNature,
  kbUploadFile,
  kbImportFuturesWeeklySqlite,
  type IngestionQueueSnapshot,
  type KbFile,
  type KbProduct,
} from "../api";
import { hasTraceableSource, isDemoFileName, isDemoProduct, needsManualNaming } from "./productDisplay";
import {
  findNextProductReviewFile,
  isBulkNavDataset,
  isProductReviewFile,
  isResearchReady,
  needsProductBinding,
  productWorkflowAction,
} from "./productWorkflow";
import MaterialUploadCard from "./MaterialUploadCard";
import ProductLibraryItem from "./ProductLibraryItem";
import MultiProductReviewModal, { type CropRegion, type CurveReviewChoice } from "./MultiProductReviewModal";
import { useWorkbenchActions } from "./FofWorkbench";

const { Text } = Typography;

type UploadBatch = {
  total: number;
  uploaded: number;
  completed: number;
  failed: number;
  fileIds: string[];
};

/** One pending workbench item, shared by the next-task button and its label.
 *
 * The button label and the click behaviour used to be computed by two
 * different rule sets, so the label could promise one task while the click
 * opened another.  ``decideNextTask`` is the single source of truth for both.
 */
type NextTaskDecision =
  | { kind: "review"; label: string; file: KbFile }
  | { kind: "bind"; label: string; file: KbFile }
  | { kind: "curves"; label: string; file: KbFile }
  | { kind: "recover"; label: string; file: KbFile }
  | { kind: "reparse"; label: string; file: KbFile }
  | { kind: "product"; label: string; product: KbProduct }
  | { kind: "waiting"; label: string; waitingCount: number }
  | { kind: "idle"; label: string };

/** True once a file has finished parsing (or failed), so binding never opens
 * for a half-parsed file whose product identity has not been extracted yet. */
function isTerminalFile(file: KbFile): boolean {
  return file.parsing_status !== "pending" && file.parsing_status !== "processing";
}

function decideNextTask(
  queueFiles: KbFile[],
  visibleProducts: KbProduct[],
  excludedIds: ReadonlySet<string>,
  preferredIds: readonly string[] | undefined,
  productLabel: (product: KbProduct) => string | null,
): NextTaskDecision {
  const inFlightCount = queueFiles.filter((file) => !isTerminalFile(file)).length;
  const reviewFile = findNextProductReviewFile(queueFiles, excludedIds, preferredIds);
  if (reviewFile) {
    return { kind: "review", label: `复核「${reviewFile.linked_product_names?.[0] ?? reviewFile.filename}」`, file: reviewFile };
  }
  const bindFile = queueFiles.find((file) => (
    !excludedIds.has(file.id)
    && needsProductBinding(file)
    && isTerminalFile(file)
    && file.parsing_status !== "failed"
  ));
  if (bindFile) return { kind: "bind", label: `绑定「${bindFile.filename}」`, file: bindFile };
  const curveFile = queueFiles.find((file) => !excludedIds.has(file.id) && (file.unbound_curve_count ?? 0) > 0);
  if (curveFile) return { kind: "curves", label: `确认曲线归属 · ${curveFile.filename}`, file: curveFile };
  const recoverFile = queueFiles.find((file) => (
    file.workflow?.next_action === "recover_chart_calibration" || file.parsing_status === "completed_no_nav"
  ));
  if (recoverFile) {
    return { kind: "recover", label: `校准「${recoverFile.linked_product_names?.[0] ?? recoverFile.filename}」`, file: recoverFile };
  }
  const failedFile = queueFiles.find((file) => file.parsing_status === "failed");
  if (failedFile) return { kind: "reparse", label: `重试「${failedFile.filename}」`, file: failedFile };
  const nextProduct = visibleProducts.find((product) => !isResearchReady(product) && product.confirmation_status !== "rejected");
  if (nextProduct) {
    const label = productLabel(nextProduct);
    if (label) return { kind: "product", label, product: nextProduct };
  }
  if (inFlightCount > 0) return { kind: "waiting", label: `等待解析 ${inFlightCount} 份`, waitingCount: inFlightCount };
  return { kind: "idle", label: "处理下一项" };
}

export type CalibrationQueueItem = {
  product: KbProduct;
  sourceFileId: string;
  sourceRegion: CropRegion;
  sourceFragmentId: string;
};

/** Product-library lanes describe data readiness, never expected performance.
 *
 * Workbench-level callbacks (focus / research / calibrate / queue) are no
 * longer drilled through this panel: they are provided by the
 * WorkbenchActionsContext owned by FofWorkbench and consumed directly here
 * and inside ProductLibraryItem.
 */
interface Props {
  refreshKey?: number;
  mode?: "queue" | "catalog";
  selectedIds: string[];
  onToggleSelect: (id: string) => void;
}

export default function ProductLibraryPanel({ refreshKey = 0, mode = "queue", selectedIds, onToggleSelect }: Props): React.JSX.Element {
  const { onCalibrateProduct, onOpenSelection, onStartCalibrationQueue } = useWorkbenchActions();
  const [products, setProducts] = useState<KbProduct[]>([]);
  const [files, setFiles] = useState<KbFile[]>([]);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sortBy, setSortBy] = useState<"recent" | "name" | "status">("recent");
  const [strategyFilter, setStrategyFilter] = useState<string>();
  const [frequencyFilter, setFrequencyFilter] = useState<string>();
  const [managerFilter, setManagerFilter] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [uploadCount, setUploadCount] = useState(0);
  const [uploadBatch, setUploadBatch] = useState<UploadBatch | null>(null);
  const [uploadExpanded, setUploadExpanded] = useState(true);
  const [uploadMaterialNature, setUploadMaterialNature] = useState<string>();
  const [sqliteImporting, setSqliteImporting] = useState(false);
  const stagedUploadsRef = useRef<File[]>([]);
  const uploadStartTimerRef = useRef<number | null>(null);
  const refreshRequestIdRef = useRef(0);
  const listRef = useRef<HTMLDivElement>(null);
  const initialLoadDone = useRef(false);
  const activeReviewFileIdRef = useRef<string | undefined>(undefined);
  const resumeCurveReviewFileIdRef = useRef<string | undefined>(undefined);
  const resumeCurveReviewAfterSaveFileIdRef = useRef<string | undefined>(undefined);
  const openingReviewFileIdRef = useRef<string | undefined>(undefined);
  const startedReviewFileIdsRef = useRef(new Set<string>());
  const completedReviewFileIdsRef = useRef(new Set<string>());
  const [autoAdvanceRequest, setAutoAdvanceRequest] = useState(0);
  const [editingProduct, setEditingProduct] = useState<KbProduct | null>(null);
  const [confirmAfterEdit, setConfirmAfterEdit] = useState(false);
  const [editForm] = Form.useForm();
  const [bindingFile, setBindingFile] = useState<KbFile | null>(null);
  const [bindingMode, setBindingMode] = useState<"existing" | "new">("existing");
  const [bindingSubmitting, setBindingSubmitting] = useState(false);
  const [curveBindingFile, setCurveBindingFile] = useState<KbFile | null>(null);
  const [machineReviewing, setMachineReviewing] = useState(false);
  const [bindingForm] = Form.useForm();
  const [ingestionQueue, setIngestionQueue] = useState<IngestionQueueSnapshot | null>(null);

  // Debounce search input (300ms)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search), 300);
    return () => window.clearTimeout(timer);
  }, [search]);

  const refresh = useCallback(async (options?: { silent?: boolean }) => {
    const requestId = ++refreshRequestIdRef.current;
    setLoading(true);
    try {
      const params: { search?: string; limit: number } = { limit: 50 };
      if (debouncedSearch.trim()) params.search = debouncedSearch.trim();
      const [productList, fileList] = await Promise.all([
        kbListProducts(params),
        mode === "queue" ? kbListFiles({ pendingOnly: true }) : Promise.resolve([] as KbFile[]),
      ]);
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
  }, [debouncedSearch, mode]);

  const importFuturesSqlite = useCallback(async (file: File) => {
    setSqliteImporting(true);
    try {
      const result = await kbImportFuturesWeeklySqlite(file);
      message.success(`已导入 ${result.products_selected} 个产品，新增 ${result.nav_observations_added} 条周频净值`);
      await refresh();
    } catch (error) {
      message.error(error instanceof Error ? error.message : "SQLite 导入失败");
    } finally {
      setSqliteImporting(false);
    }
  }, [refresh]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { if (refreshKey > 0) void refresh({ silent: true }); }, [refresh, refreshKey]);

  // The single-product calibration workspace is a sibling view, so refresh
  // the list immediately after a NAV review is saved there.
  useEffect(() => {
    const handler = () => {
      const resumeFileId = resumeCurveReviewAfterSaveFileIdRef.current;
      if (resumeFileId) {
        resumeCurveReviewAfterSaveFileIdRef.current = undefined;
        resumeCurveReviewFileIdRef.current = resumeFileId;
      }
      const fileId = activeReviewFileIdRef.current;
      if (fileId) {
        completedReviewFileIdsRef.current.add(fileId);
        activeReviewFileIdRef.current = undefined;
        setAutoAdvanceRequest((value) => value + 1);
      }
      void refresh({ silent: true });
    };
    window.addEventListener("kb:product-updated", handler);
    return () => window.removeEventListener("kb:product-updated", handler);
  }, [refresh]);

  useEffect(() => {
    const hasProcessing = files.some((file) => file.parsing_status === "processing");
    if (!hasProcessing) {
      setIngestionQueue(null);
      return;
    }
    let cancelled = false;
    const poll = () => {
      void refresh({ silent: true });
      void kbGetIngestionQueue()
        .then((queue) => { if (!cancelled) setIngestionQueue(queue); })
        .catch(() => { if (!cancelled) setIngestionQueue(null); });
    };
    const timer = window.setInterval(poll, 3000);
    return () => { cancelled = true; window.clearInterval(timer); };
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
        && (!strategyFilter || product.strategy === strategyFilter)
        && (!frequencyFilter || product.nav_frequency === frequencyFilter)
        && (!managerFilter || product.manager_name === managerFilter)
      ));
    },
    [frequencyFilter, managerFilter, products, search, strategyFilter],
  );
  const catalogFilterOptions = React.useMemo(() => ({
    strategies: Array.from(new Set(products.map((product) => product.strategy).filter((value): value is string => Boolean(value)))).sort((a, b) => a.localeCompare(b, "zh")),
    frequencies: Array.from(new Set(products.map((product) => product.nav_frequency).filter((value): value is string => Boolean(value)))).sort(),
    managers: Array.from(new Set(products.map((product) => product.manager_name).filter((value): value is string => Boolean(value)))).sort((a, b) => a.localeCompare(b, "zh")),
  }), [products]);
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
    () => visibleFiles.filter((file) => !isBulkNavDataset(file) && !["CTA策略介绍", "CTA研究/市场资料", "忽略"].includes(file.ingestion_context?.material_nature ?? "") && (
      !(file.linked_product_ids?.length)
      || file.parsing_status === "pending"
      || file.parsing_status === "processing"
      || file.parsing_status === "failed"
      || file.parsing_status === "completed_no_nav"
      || isProductReviewFile(file)
      || (file.unbound_curve_count ?? 0) > 0
      || (file.unbound_fragment_count ?? 0) > 0
    )),
    [visibleFiles],
  );

  const startProductReview = useCallback(async (file: KbFile, productIdOverride?: string): Promise<boolean> => {
    if (openingReviewFileIdRef.current) return false;
    const productId = file.linked_product_ids?.[0] ?? productIdOverride;
    if (!productId) return false;
    openingReviewFileIdRef.current = file.id;
    try {
      const product = products.find((candidate) => candidate.id === productId) ?? await kbGetProduct(productId);
      if (product.confirmation_status === "rejected") {
        message.warning("该文件已关联到被标记为非产品的记录，无法进入复核。");
        return false;
      }
      activeReviewFileIdRef.current = file.id;
      startedReviewFileIdsRef.current.add(file.id);
      onCalibrateProduct(product, file.id);
      return true;
    } catch (error) {
      message.error(error instanceof Error ? error.message : "打开产品复核失败");
      return false;
    } finally {
      openingReviewFileIdRef.current = undefined;
    }
  }, [onCalibrateProduct, products]);

  // As soon as a newly uploaded source has a defensible product target and a
  // parsed NAV candidate, open the review workspace without another click.
  useEffect(() => {
    if (autoAdvanceRequest > 0 || activeReviewFileIdRef.current || !uploadBatch?.fileIds.length) return;
    const nextFile = findNextProductReviewFile(
      queueFiles,
      startedReviewFileIdsRef.current,
      uploadBatch.fileIds,
    );
    if (nextFile) void startProductReview(nextFile);
  }, [autoAdvanceRequest, queueFiles, startProductReview, uploadBatch?.fileIds]);

  const researchFilteredProducts = React.useMemo(
    () => visibleProducts.filter((product) => mode === "catalog" ? isResearchReady(product) : !isResearchReady(product)),
    [mode, visibleProducts],
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
          const result = await kbUploadFile(file, uploadMaterialNature);
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
  }, [refresh, uploadMaterialNature]);

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

  const classifyQueueFile = useCallback(async (fileId: string, materialNature: "CTA策略介绍" | "CTA研究/市场资料" | "忽略") => {
    try {
      await kbUpdateFileMaterialNature(fileId, materialNature);
      message.success(materialNature === "忽略" ? "已移出待办，原始资料仍保留" : "已归档为研究证据，不再要求净值校准");
      void refresh({ silent: true });
    } catch (error) {
      message.error(error instanceof Error ? error.message : "更新资料用途失败");
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
      const boundFile = bindingFile;
      setBindingFile(null);
      bindingForm.resetFields();
      void refresh({ silent: true });
      // Binding resolves identity; it is not the end of the workflow.  Load
      // the resulting product and continue directly into source review.
      void startProductReview(boundFile, result.product_id);
    } catch (error) {
      if (error instanceof Error) message.error(error.message || "绑定产品失败");
    } finally {
      setBindingSubmitting(false);
    }
  }, [bindingFile, bindingForm, bindingMode, refresh, startProductReview]);

  const openCurveBinding = useCallback((file: KbFile) => {
    setCurveBindingFile(file);
  }, []);

  const createCurveProduct = useCallback(async (standardName: string): Promise<KbProduct> => {
    const created = await kbCreateProduct({ standard_name: standardName });
    const product = await kbGetProduct(created.id);
    setProducts((current) => current.some((item) => item.id === product.id) ? current : [product, ...current]);
    return product;
  }, []);

  const handleBindCurves = useCallback(async (bindings: CurveReviewChoice[], openAfterSubmit?: CurveReviewChoice, startQueue = false) => {
    if (!curveBindingFile) return;
    if (!bindings.length) {
      message.warning("请至少为一条曲线选择产品");
      return;
    }
    setBindingSubmitting(true);
    try {
      const result = await kbBindFileCurves(curveBindingFile.id, bindings);
      message.success(`已确认 ${result.bound_curves} 条曲线归属`);
      const sourceFile = curveBindingFile;
      setCurveBindingFile(null);
      await refresh({ silent: true });
      if (openAfterSubmit) {
        const product = products.find((item) => item.id === openAfterSubmit.product_id) ?? await kbGetProduct(openAfterSubmit.product_id);
        // Resume this source only after the reviewer saves the current NAV.
        // Until then the calibration workspace must remain unobstructed.
        resumeCurveReviewAfterSaveFileIdRef.current = sourceFile.id;
        onCalibrateProduct(product, sourceFile.id, openAfterSubmit.region, openAfterSubmit.fragment_id);
      } else if (startQueue) {
        const queue = await Promise.all(bindings.map(async (choice) => ({
          product: products.find((item) => item.id === choice.product_id) ?? await kbGetProduct(choice.product_id),
          sourceFileId: sourceFile.id,
          sourceRegion: choice.region,
          sourceFragmentId: choice.fragment_id,
        })));
        onStartCalibrationQueue(queue);
      }
    } catch (error) {
      message.error(error instanceof Error ? error.message : "批量绑定曲线失败");
    } finally {
      setBindingSubmitting(false);
    }
  }, [curveBindingFile, onCalibrateProduct, onStartCalibrationQueue, products, refresh]);

  useEffect(() => {
    const fileId = resumeCurveReviewFileIdRef.current;
    if (!fileId || curveBindingFile) return;
    const refreshedFile = files.find((file) => file.id === fileId);
    if (!refreshedFile) return;
    resumeCurveReviewFileIdRef.current = undefined;
    if ((refreshedFile.unbound_curve_count ?? 0) > 0) openCurveBinding(refreshedFile);
  }, [curveBindingFile, files, openCurveBinding]);

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
        } catch (error) {
          message.error(error instanceof Error ? error.message : "删除产品失败");
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

  const productActionLabel = useCallback((product: KbProduct): string | null => {
    const action = productWorkflowAction(product, {
      editIdentity: () => openEdit(product, true),
      calibrateNav: () => {
        activeReviewFileIdRef.current = undefined;
        onCalibrateProduct(product);
      },
      confirmIdentity: () => void handleConfirm(product.id),
    });
    return action?.label ?? null;
  }, [handleConfirm, onCalibrateProduct, openEdit]);

  const handleNextTask = useCallback(() => {
    const decision = decideNextTask(
      queueFiles,
      visibleProducts,
      completedReviewFileIdsRef.current,
      uploadBatch?.fileIds,
      productActionLabel,
    );
    switch (decision.kind) {
      case "review":
      case "recover":
        void startProductReview(decision.file);
        return;
      case "bind":
        openBinding(decision.file);
        return;
      case "curves":
        openCurveBinding(decision.file);
        return;
      case "reparse":
        void handleReparse(decision.file.id);
        return;
      case "product": {
        const action = productWorkflowAction(decision.product, {
          editIdentity: () => openEdit(decision.product, true),
          calibrateNav: () => {
            activeReviewFileIdRef.current = undefined;
            onCalibrateProduct(decision.product);
          },
          confirmIdentity: () => void handleConfirm(decision.product.id),
        });
        if (action) {
          action.run();
          return;
        }
        break;
      }
      case "waiting":
        setUploadExpanded(true);
        message.info(`还有 ${decision.waitingCount} 份资料在后台解析；完成后队列会自动刷新，无需重复上传。`);
        return;
      case "idle":
        message.info("当前没有可继续处理的待办。");
        return;
    }
  }, [handleConfirm, handleReparse, onCalibrateProduct, openBinding, openCurveBinding, openEdit, productActionLabel, queueFiles, startProductReview, uploadBatch?.fileIds, visibleProducts]);

  // The calibration page is rendered by the parent.  Once it broadcasts a
  // successful save, this mounted (but possibly hidden) workbench advances to
  // the next parsed product instead of making the user return and click again.
  useEffect(() => {
    if (!autoAdvanceRequest || activeReviewFileIdRef.current) return;
    const decision = decideNextTask(
      queueFiles,
      visibleProducts,
      completedReviewFileIdsRef.current,
      uploadBatch?.fileIds,
      // Auto-advance only walks the file queue; product-level actions stay
      // on the cards so the user remains in control of identity decisions.
      () => null,
    );
    switch (decision.kind) {
      case "review":
        void startProductReview(decision.file).then((opened) => {
          if (opened) setAutoAdvanceRequest(0);
        });
        return;
      case "bind":
        openBinding(decision.file);
        setAutoAdvanceRequest(0);
        return;
      case "curves":
        openCurveBinding(decision.file);
        setAutoAdvanceRequest(0);
        return;
      case "waiting":
        // Keep the request pending; the poll refresh re-runs this effect once
        // parsing finishes and a reviewable candidate appears.
        return;
      default:
        setAutoAdvanceRequest(0);
        return;
    }
  }, [autoAdvanceRequest, openBinding, openCurveBinding, queueFiles, startProductReview, uploadBatch?.fileIds, visibleProducts]);

  const pendingFileCount = queueFiles.filter((f) => f.parsing_status === "pending" || f.parsing_status === "processing").length;
  const bindingProducts = visibleProducts.filter((product) => product.confirmation_status !== "rejected");
  const readyCount = visibleProducts.filter(isResearchReady).length;
  const actionableCount = Math.max(0, visibleProducts.length - readyCount);
  const needsNamingCount = visibleProducts.filter(needsManualNaming).length;
  // The button label and handleNextTask share one decision, so the label can
  // never promise a task the click does not deliver.
  const nextTask = React.useMemo(
    () => decideNextTask(queueFiles, visibleProducts, completedReviewFileIdsRef.current, uploadBatch?.fileIds, productActionLabel),
    [queueFiles, visibleProducts, uploadBatch?.fileIds, productActionLabel],
  );
  const nextTaskLabel = nextTask.label;
  // Surface the async pipeline: which files are in flight, how many slots the
  // backend runs, and how many files are still queued behind the semaphore.
  const processingNames = React.useMemo(
    () => files.filter((file) => file.parsing_status === "processing").map((file) => file.filename),
    [files],
  );
  const activeNames = React.useMemo(() => {
    const activeFileIds = new Set(ingestionQueue?.active_file_ids ?? []);
    return files
      .filter((file) => file.parsing_status === "processing" && activeFileIds.has(file.id))
      .map((file) => file.filename);
  }, [files, ingestionQueue]);
  const processingLead = pendingFileCount > 0
    ? activeNames.length > 0
      ? `后台异步识别中「${activeNames[0]}」${processingNames.length > 1 ? ` 等 ${processingNames.length} 份` : ""}`
      : `后台异步识别中 ${processingNames.length} 份`
    : "";
  // An uploaded file is not automatically a research-ready product. Keep the
  // compact summary explicit so “可研究 0” is not mistaken for a failed upload.
  const uploadSummary = queueFiles.length === 0
    ? "暂无待处理资料 · 已绑定资料已归档到产品"
    : readyCount > 0
    ? `待处理 ${queueFiles.length} 份资料 · 可研究 ${readyCount} 个产品`
    : pendingFileCount > 0
      ? `待处理 ${queueFiles.length} 份资料 · ${processingLead}`
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
      {mode === "queue" && <MaterialUploadCard
        hasQueue={queueFiles.length > 0}
        expanded={uploadExpanded}
        onExpandedChange={setUploadExpanded}
        summary={uploadSummary}
        nextTaskLabel={nextTaskLabel}
        onNextTask={handleNextTask}
        uploadCount={uploadCount}
        onStageUpload={stageUpload}
        materialNature={uploadMaterialNature}
        onMaterialNatureChange={setUploadMaterialNature}
        batch={uploadBatch}
        batchPercent={batchPercent}
        batchActiveCount={batchActiveCount}
        batchNavExtractedCount={batchNavExtractedCount}
        processingNames={processingNames}
        maxConcurrency={ingestionQueue?.max_concurrency}
        queuedCount={ingestionQueue?.queued_count}
        queueFiles={queueFiles}
        activeFileIds={ingestionQueue?.active_file_ids}
        onClassifyFile={(fileId, materialNature) => void classifyQueueFile(fileId, materialNature)}
      />}

      {mode === "queue" && <MultiProductReviewModal
        file={curveBindingFile}
        products={bindingProducts}
        open={curveBindingFile !== null}
        submitting={bindingSubmitting}
        onCancel={() => { if (!bindingSubmitting) setCurveBindingFile(null); }}
        onSubmit={handleBindCurves}
        onCreateProduct={createCurveProduct}
      />}

      {/* Filter + search */}
      <div style={{ flexShrink: 0, display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
          <Text strong>{mode === "catalog" ? "产品列表" : "待处理队列"}</Text>
          {mode === "queue" && <Dropdown trigger={["click"]} menu={{ items: [
            { key: "sqlite-import", label: <label style={{ cursor: sqliteImporting ? "wait" : "pointer" }}>导入期货周频 SQLite<input type="file" accept=".sqlite,.sqlite3,.db" disabled={sqliteImporting} style={{ display: "none" }} onChange={(event) => { const file = event.target.files?.[0]; if (file) void importFuturesSqlite(file); event.currentTarget.value = ""; }} /></label> },
            { key: "machine-review", label: machineReviewing ? "正在自动检查…" : "自动检查待复核曲线", disabled: machineReviewing, onClick: () => void handleMachineReview() },
            { key: "refresh", icon: <ReloadOutlined />, label: "刷新列表", onClick: () => void refresh() },
          ] }}>
            <Button size="small" type="text" icon={<MoreOutlined />} aria-label="复核队列更多操作" />
          </Dropdown>}
        </div>
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
        {mode === "catalog" && <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          <Select
            size="small"
            allowClear
            placeholder="策略分类"
            value={strategyFilter}
            onChange={setStrategyFilter}
            style={{ width: 132 }}
            options={catalogFilterOptions.strategies.map((value) => ({ value, label: value }))}
          />
          <Select
            size="small"
            allowClear
            placeholder="净值频率"
            value={frequencyFilter}
            onChange={setFrequencyFilter}
            style={{ width: 112 }}
            options={catalogFilterOptions.frequencies.map((value) => ({ value, label: value === "daily" ? "日频" : value === "weekly" ? "周频" : value === "monthly" ? "月频" : value }))}
          />
          <Select
            size="small"
            allowClear
            showSearch
            optionFilterProp="label"
            placeholder="管理人"
            value={managerFilter}
            onChange={setManagerFilter}
            style={{ width: 150 }}
            options={catalogFilterOptions.managers.map((value) => ({ value, label: value }))}
          />
          {(strategyFilter || frequencyFilter || managerFilter) && <Button size="small" type="link" onClick={() => { setStrategyFilter(undefined); setFrequencyFilter(undefined); setManagerFilter(undefined); }}>清除筛选</Button>}
        </div>}
      </div>

      {/* Product list */}
      <div ref={listRef} style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
        {sortedProducts.length === 0 && !loading ? (
          <Empty description={mode === "catalog" ? "当前没有已审核产品" : "当前没有待处理项目"} image={Empty.PRESENTED_IMAGE_SIMPLE} style={{ marginTop: 40 }} />
        ) : (
          <List
            size="small"
            loading={loading}
            dataSource={sortedProducts}
            renderItem={(product) => <ProductLibraryItem
              product={product}
              selected={selectedIds.includes(product.id)}
              filenameById={filenameById}
              onToggleSelect={onToggleSelect}
              onEdit={openEdit}
              onConfirm={(id) => void handleConfirm(id)}
              onReviewCurves={(() => {
                // A published single-product series must not be sent back to
                // multi-product binding merely because old parser candidates
                // remain in its source audit trail.
                if (product.nav_count > 0 && product.reviewed_nav_count >= product.nav_count) return undefined;
                const source = files.find((file) => product.source_file_ids?.includes(file.id) && (file.unbound_curve_count ?? 0) > 0);
                return source ? () => openCurveBinding(source) : undefined;
              })()}
              onReject={(id) => void handleReject(id)}
              onDelete={handleDelete}
            />}
          />
        )}
      </div>

      {/* Footer / batch actions */}
      {selectedIds.length > 0 ? (
        <div style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 8, padding: "6px 0", minWidth: 0, flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 180px", minWidth: 0 }}>
            <Text strong style={{ fontSize: 12 }}>研究对象：{selectedIds.length} 个产品</Text>
            <Text type="secondary" style={{ display: "block", fontSize: 11 }}>用于工作台筛选与 Agent 研究，不等同于对比对象。</Text>
          </div>
          <Button size="small" onClick={() => onOpenSelection(selectedIds, "agent")}>进入 Agent 对话</Button>
          <Button size="small" onClick={() => onOpenSelection(selectedIds, "scores")}>查看评分与画像</Button>
          <Button size="small" onClick={() => onOpenSelection(selectedIds, "ranking")}>查看周度排名</Button>
          {selectedIds.length >= 2 && <Button size="small" type="primary" onClick={() => onOpenSelection(selectedIds, "portfolio")}>组合风险分析</Button>}
          <Button type="link" size="small" style={{ paddingInline: 0 }} onClick={() => selectedIds.forEach((id) => onToggleSelect(id))}>清空研究对象</Button>
        </div>
      ) : (
        <Text type="secondary" style={{ flexShrink: 0, fontSize: 11 }}>{mode === "catalog" ? "仅显示已审核、可直接研究的产品。点击产品即可加入研究对象。" : "仅显示待处理产品；已完成产品可在“产品列表”中查看。"}</Text>
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
