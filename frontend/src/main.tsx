import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Alert,
  Button,
  Card,
  Col,
  ConfigProvider,
  Divider,
  Dropdown,
  Drawer,
  Layout,
  Row,
  Space,
  Statistic,
  Typography,
} from "antd";
import {
  analyzeNav,
  extractSingleImage,
  sampleColorAtPixel,
  getVlmConfig,
  kbGetNavSeries,
  kbListNavCandidates,
  kbGetReviewPage,
  kbGetSourcePreview,
  kbReplaceNavSeries,
  updateVlmConfig,
  type DataFrequency,
  type ImageExtractionFrequency,
  type KbProduct,
  type NavAnalysisResponse,
  type NavPoint,
  type ReportGenerateResponse,
} from "./api";
import { formatNavText, parseNavText, sampleWeeklyNav } from "./parser/navParser";
import { parseAxisDate } from "./parser/axisLabelParser";
import { percentage } from "./utils/format";
import { exportNavXlsx } from "./report/exportXlsx";
import { ReportExtractionView } from "./components/display";
import HistoryDrawer from "./components/HistoryDrawer";
import type { HistoryRecord } from "./storage/historyStorage";
import { StepSection } from "./components/StepSection";
import AnalysisPanel from "./AnalysisPanel";
import FactorAttributionPanel from "./FactorAttributionPanel";
import ExternalFactorLibraryPanel from "./ExternalFactorLibraryPanel";
import {
  ACCENT_STORAGE_KEY,
  applyAccentCssVariables,
  buildSerifTheme,
  FONT_DISPLAY,
  getPreset,
  loadAccentKey,
} from "./theme";
import SettingsDrawer, {
  DEFAULTS_STORAGE_KEY,
  ENGINE_STORAGE_KEY,
  loadEngineConfig,
  loadResearchDefaults,
  type AnalysisEngineConfig,
  type ResearchDefaults,
} from "./SettingsDrawer";
import FactorLibraryDrawer from "./FactorLibraryDrawer";
import ChartExtractDrawer from "./ChartExtractDrawer";
import FofWorkbench from "./fof/FofWorkbench";
import { useImageDigitization } from "./hooks/useImageDigitization";
import { useProductIdentity } from "./hooks/useProductIdentity";
import { useHistory } from "./hooks/useHistory";
import type { CandidateCurvePoint } from "./components/nav/CandidateCurveOverlay";
import ImageExtractionActions from "./components/nav/ImageExtractionActions";
import AdvancedCalibrationPanel from "./components/nav/AdvancedCalibrationPanel";
import NavImportReviewPanel from "./components/nav/NavImportReviewPanel";
import ChartPreviewReview from "./components/nav/ChartPreviewReview";
import { ImportWorkspaceOpenHeader } from "./components/nav/ImportWorkspaceIntro";
import PerformanceResultCard from "./components/nav/PerformanceResultCard";
import { autoFitCurveToSelectedLine } from "./utils/curveAutoFit";
import "@fontsource/playfair-display/400.css";
import "@fontsource/playfair-display/600.css";
import "@fontsource/playfair-display/700.css";
import "@fontsource/source-sans-3/400.css";
import "@fontsource/source-sans-3/500.css";
import "@fontsource/source-sans-3/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "antd/dist/reset.css";
import "./serif.css";

const { Content } = Layout;
const { Paragraph, Title, Text } = Typography;

/** Research starts empty: sample NAV must never be mistaken for product data. */
const SAMPLE_NAV_TEXT = "";

function getReviewBlockMessage(
  conflictReasons: string[],
  conflictConfirmed: boolean,
  chartReviewReasons: string[],
  chartReviewConfirmed: boolean,
): string | undefined {
  if (conflictReasons.length > 0 && !conflictConfirmed) {
    return "请先在原图核对冲突点，并勾选人工复核确认后再计算。";
  }
  if (chartReviewReasons.length > 0 && !chartReviewConfirmed) {
    return "图表识别结果仍是候选，请先对照原图完成曲线、坐标和日期校准。";
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Main application
// ---------------------------------------------------------------------------

/** Orchestrates NAV input, image digitization, product identity and results. */
function App(): React.JSX.Element {
  // NAV text and analysis state
  const [navText, setNavText] = useState(SAMPLE_NAV_TEXT);
  const [researchDefaults, setResearchDefaults] = useState<ResearchDefaults>(loadResearchDefaults);
  const [engineConfig, setEngineConfig] = useState<AnalysisEngineConfig>(loadEngineConfig);
  const [frequency, setFrequency] = useState<DataFrequency>(researchDefaults.frequency);
  // A chart trace describes pixels, not daily disclosed NAV observations.
  const [imageExtractionFrequency, setImageExtractionFrequency] = useState<ImageExtractionFrequency>("weekly");
  const [riskFreeRate, setRiskFreeRate] = useState(researchDefaults.riskFreeRate);
  const [analysisResult, setAnalysisResult] = useState<NavAnalysisResponse>();
  const [analysisError, setAnalysisError] = useState<string>();
  const [aiReport, setAiReport] = useState<ReportGenerateResponse>();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [importMessage, setImportMessage] = useState<string>();
  const [workflowStep, setWorkflowStep] = useState(0);
  const [isImportWorkspaceOpen, setIsImportWorkspaceOpen] = useState(true);

  // Image digitization (extracted hook)
  const {
    imageStartDate, setImageStartDate,
    imageEndDate, setImageEndDate,
    imageNavMin, setImageNavMin,
    imageNavMax, setImageNavMax,
    imageValueMode, handleImageValueModeChange,
    pendingImage, imagePreviewUrl,
    isDigitizing, imageError, setImageError,
    activeDateAnchor, setActiveDateAnchor,
    isZoomModalOpen, setIsZoomModalOpen,
    zoomLevel, setZoomLevel,
    startXRatio, endXRatio, topYRatio, bottomYRatio,
    missingFields, ocrHints,
    selectImage, handleImageImport: runImageImport, markDateAnchor, applyLocatedChart, clearFieldIssue, resetImage,
  } = useImageDigitization();

  // Product identity and strategy profile (extracted hook)
  const {
    productName, setProductName,
    sourceText, setSourceText,
    isRecognizingProduct,
    reportExtraction,
    resetIdentity,
  } = useProductIdentity();

  // Benchmark and disclosure comparison state
  const [reportedCumulativeReturn, setReportedCumulativeReturn] = useState<number>();
  const [benchmarkCumulativeReturn, setBenchmarkCumulativeReturn] = useState<number>();
  const [candidateCurve, setCandidateCurve] = useState<CandidateCurvePoint[]>([]);
  const [candidateConfidence, setCandidateConfidence] = useState<number>();
  const [isCurveEditMode, setIsCurveEditMode] = useState(false);
  const [isAutoFittingCurve, setIsAutoFittingCurve] = useState(false);
  const [isSmartDigitizing, setIsSmartDigitizing] = useState(false);
  const [curveLineColor, setCurveLineColor] = useState("#C9983E");
  const [isPickingCurveColor, setIsPickingCurveColor] = useState(false);
  const [autoFitMessage, setAutoFitMessage] = useState<string>();
  const calibrationRequestIdRef = useRef(0);
  const smartExtractionRequestIdRef = useRef(0);
  const calibrationSaveRequestIdRef = useRef(0);

  // History records (extracted hook)
  const {
    historyRecords, isHistoryOpen, setIsHistoryOpen,
    handleSaveHistory: saveToHistory, handleDeleteHistory, handleClearAllHistory,
  } = useHistory();

  // Settings: accent theme and drawer visibility.
  const [accentKey, setAccentKey] = useState(loadAccentKey);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [factorLibOpen, setFactorLibOpen] = useState(false);
  const [chartExtractOpen, setChartExtractOpen] = useState(false);
  const [activeView, setActiveView] = useState<"workbench" | "tools">("workbench");
  const [calibratingProductId, setCalibratingProductId] = useState<string>();
  const [calibrationSourceFileId, setCalibrationSourceFileId] = useState<string>();
  const [calibrationSourcePage, setCalibrationSourcePage] = useState<number>();
  const [calibrationSourceIsPdf, setCalibrationSourceIsPdf] = useState(false);
  const [isSavingCalibration, setIsSavingCalibration] = useState(false);
  const [calibrationConflictReasons, setCalibrationConflictReasons] = useState<string[]>([]);
  const [manualConflictConfirmed, setManualConflictConfirmed] = useState(false);
  const [chartReviewReasons, setChartReviewReasons] = useState<string[]>([]);
  const [chartReviewConfirmed, setChartReviewConfirmed] = useState(false);
  const accentPreset = useMemo(() => getPreset(accentKey), [accentKey]);
  const themeConfig = useMemo(() => buildSerifTheme(accentPreset), [accentPreset]);

  useEffect(() => {
    applyAccentCssVariables(accentPreset);
    try {
      localStorage.setItem(ACCENT_STORAGE_KEY, accentKey);
    } catch {
      // Storage may be unavailable; the theme still applies for this session.
    }
  }, [accentKey, accentPreset]);

  // The backend keeps API settings in its process environment. Restore the
  // browser-local VLM setting at app boot so a restart does not make batch
  // uploads silently fall back to CV before the Settings drawer is opened.
  // The key is never rendered or logged.
  useEffect(() => {
    let saved: { provider?: string; model?: string; apiKey?: string; baseUrl?: string };
    try {
      saved = JSON.parse(localStorage.getItem("cta_research_vlm") ?? "{}") as typeof saved;
    } catch {
      return;
    }
    if (!saved.apiKey?.trim()) return;
    void getVlmConfig()
      .then((config) => {
        if (config.api_key_set) return;
        return updateVlmConfig({
          provider: saved.provider,
          model: saved.model,
          api_key: saved.apiKey,
          base_url: saved.baseUrl || undefined,
        });
      })
      .catch(() => {
        // Backend startup is asynchronous; Settings retries when it opens.
      });
  }, []);

  // Keep a stable ref to the latest image preview URL so it can be revoked on unmount.
  const imagePreviewUrlRef = useRef(imagePreviewUrl);
  imagePreviewUrlRef.current = imagePreviewUrl;

  useEffect(() => {
    return () => {
      if (imagePreviewUrlRef.current) {
        URL.revokeObjectURL(imagePreviewUrlRef.current);
      }
    };
  }, []);

  const navCount = useMemo(() => navText.trim().split(/\r?\n/).filter(Boolean).length, [navText]);
  const candidateCurvePoints = useMemo<CandidateCurvePoint[]>(() => {
    // An overlay is evidence of this image's CV result, never a rendering of
    // whichever NAV text happened to be left from the prior image.  The old
    // fallback made a stale series look like a freshly traced CV candidate.
    return candidateCurve.length >= 2 ? candidateCurve : [];
  }, [candidateCurve]);

  const candidateMaximumDrawdown = useMemo(() => {
    try {
      const values = parseNavText(navText).map((point) => point.net_asset_value);
      if (values.length < 2) return undefined;
      let peak = values[0] ?? 0;
      let drawdown = 0;
      for (const value of values) {
        peak = Math.max(peak, value);
        if (peak > 0) drawdown = Math.min(drawdown, value / peak - 1);
      }
      return drawdown;
    } catch {
      return undefined;
    }
  }, [navText]);

  /** Difference between computed cumulative return and a user-supplied disclosure value. */
  const disclosedDifference = useMemo(() => {
    if (!analysisResult || reportedCumulativeReturn === undefined) return undefined;
    return analysisResult.metrics.cumulative_return - reportedCumulativeReturn / 100;
  }, [analysisResult, reportedCumulativeReturn]);

  /** True when the user has supplied enough text NAV data to attempt an analysis. */
  const canAnalyzeText = navCount >= 2;

  /** Parsed NAV points for the AI analysis panel (safe: returns [] on parse error). */
  const parsedNavPoints = useMemo<NavPoint[]>(() => {
    try {
      return parseNavText(navText);
    } catch {
      return [];
    }
  }, [navText]);

  const handleAnalysis = useCallback(async (): Promise<void> => {
    setAnalysisError(undefined);
    if (!canAnalyzeText) {
      setAnalysisError("至少需要两条有效净值记录才能计算。");
      return;
    }
    const reviewBlockMessage = getReviewBlockMessage(
      calibrationConflictReasons,
      manualConflictConfirmed,
      chartReviewReasons,
      chartReviewConfirmed,
    );
    if (reviewBlockMessage) {
      setAnalysisError(reviewBlockMessage);
      return;
    }
    setIsAnalyzing(true);
    try {
      const result = await analyzeNav(parseNavText(navText), frequency, riskFreeRate);
      setAnalysisResult(result);
      // A new product series invalidates any previously extracted benchmark.
      setBenchmarkCumulativeReturn(undefined);
      // Calculation is the completion gate for data intake.  Move the user
      // straight to the analysis workspace once the verified series is ready.
      setWorkflowStep(1);
    } catch (error) {
      setAnalysisResult(undefined);
      setAnalysisError(error instanceof Error ? error.message : "分析失败，请检查输入。");
    } finally {
      setIsAnalyzing(false);
    }
  }, [calibrationConflictReasons.length, canAnalyzeText, chartReviewConfirmed, chartReviewReasons.length, frequency, manualConflictConfirmed, navText, riskFreeRate]);

  // Keyboard shortcut: Ctrl/Cmd + Enter triggers analysis from the textarea.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent): void {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        void handleAnalysis();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleAnalysis]);

  async function handleChartApply(navTextValue: string, reviewConfirmed = false): Promise<void> {
    setNavText(navTextValue);
    setManualConflictConfirmed(false);
    setChartReviewReasons(reviewConfirmed ? [] : ["review_required"]);
    setChartReviewConfirmed(reviewConfirmed);
    setAnalysisResult(undefined);
    setAnalysisError(undefined);
    setBenchmarkCumulativeReturn(undefined);
    if (!reviewConfirmed) {
      setWorkflowStep(0);
      setImportMessage("图表结果已写入候选区，但尚未完成人工复核；请对照原图确认曲线、坐标和日期后再计算。");
      return;
    }
    const reviewBlockMessage = getReviewBlockMessage(
      calibrationConflictReasons,
      false,
      [],
      true,
    );
    if (reviewBlockMessage) {
      setWorkflowStep(0);
      setAnalysisError(reviewBlockMessage);
      setImportMessage("图表结果已写入候选区；请先完成产品资料冲突复核，再计算。");
      return;
    }
    try {
      const points = parseNavText(navTextValue);
      if (points.length < 2) throw new Error("提取结果不足两条有效净值记录。");
      setIsAnalyzing(true);
      const result = await analyzeNav(points, frequency, riskFreeRate);
      setAnalysisResult(result);
      setWorkflowStep(1);
      setImportMessage(`已写入并计算 ${points.length} 条图表净值，现已进入分析步骤。`);
    } catch (error) {
      setWorkflowStep(0);
      setAnalysisError(error instanceof Error ? error.message : "图表净值写入后计算失败，请核对日期与数值。");
      setImportMessage("已写入图表净值；日期或数值尚需核验后才能计算。");
    } finally {
      setIsAnalyzing(false);
    }
  }

  const handleCalibrateProduct = useCallback(async (product: KbProduct, preferredSourceFileId?: string): Promise<void> => {
    const requestId = ++calibrationRequestIdRef.current;
    ++smartExtractionRequestIdRef.current;
    ++calibrationSaveRequestIdRef.current;
    setIsSmartDigitizing(false);
    setIsSavingCalibration(false);
    resetImage();
    setCandidateCurve([]);
    setCandidateConfidence(undefined);
    setIsPickingCurveColor(false);
    setIsCurveEditMode(false);
    setAutoFitMessage(undefined);
    setActiveView("tools");
    setWorkflowStep(0);
    setIsImportWorkspaceOpen(true);
    setCalibratingProductId(product.id);
    setCalibrationSourcePage(undefined);
    setCalibrationSourceIsPdf(false);
    setProductName(product.standard_name);
    setAnalysisResult(undefined);
    setAnalysisError(undefined);
    setCalibrationConflictReasons(product.nav_quality?.reasons ?? []);
    setManualConflictConfirmed(false);
    setChartReviewReasons([]);
    setChartReviewConfirmed(false);
    try {
      const nav = await kbGetNavSeries(product.id);
      if (requestId !== calibrationRequestIdRef.current) return;
      const candidates = await kbListNavCandidates(product.id);
      if (requestId !== calibrationRequestIdRef.current) return;
      const pendingCandidates = candidates.filter((candidate) => candidate.status === "pending" && candidate.point_count >= 2);
      const preferredCandidate = pendingCandidates.sort((left, right) => {
        const methodRank = (candidate: typeof left): number => candidate.frequency === "monthly" ? 1 : 0;
        return methodRank(right) - methodRank(left)
          || (right.confidence ?? 0) - (left.confidence ?? 0)
          || right.point_count - left.point_count;
      })[0];
      const sourcePoints = nav.length > 0
        ? nav.map((point) => ({ observation_date: point.observation_date, net_asset_value: point.acc_nav ?? point.nav }))
        : (preferredCandidate?.points ?? []).map((point) => ({ observation_date: point.observation_date, net_asset_value: point.nav }));
      const reviewPoints = preferredCandidate?.frequency === "monthly"
        ? sourcePoints
        : sampleWeeklyNav(sourcePoints);
      const candidateFrequency = preferredCandidate?.frequency;
      const usingPendingCandidate = nav.length === 0 && Boolean(preferredCandidate);
      setChartReviewReasons(usingPendingCandidate ? ["candidate_pending_review"] : []);
      setChartReviewConfirmed(false);
      if (preferredCandidate) {
        setImportMessage(
          candidateFrequency === "monthly"
            ? `已载入月收益表复利重建的 ${preferredCandidate.point_count} 个候选净值点，请对照原表复核后确认。`
            : `已载入 ${preferredCandidate.point_count} 个候选净值点，请对照原始资料复核。`,
        );
      }
      setNavText(formatNavText(reviewPoints));
      setFrequency(candidateFrequency === "monthly" ? "monthly" : "weekly");
      // A pending/unmatched curve may not have NAV points yet, but its chart
      // fragment still carries the source file. Use that file so calibration
      // can be opened to create the first reviewed series.
      const sourceFileId = preferredSourceFileId
        ?? nav.find((point) => point.source_file_id)?.source_file_id
        ?? product.source_file_ids?.[0];
      setCalibrationSourceFileId(sourceFileId ?? undefined);
      if (sourceFileId) {
        // Always use the browser-displayable preview, including ordinary
        // images.  Fetching the original PDF first made the old branch depend
        // on File MIME/name parsing; a valid rendered PNG could then be
        // dropped silently and leave the calibration canvas empty.
        const location = await kbGetReviewPage(sourceFileId, product.id);
        if (requestId !== calibrationRequestIdRef.current) return;
        if (location.is_pdf && location.page_number === null) {
          setCalibrationSourcePage(undefined);
          setCalibrationSourceIsPdf(true);
          setImportMessage(`正在复核「${product.standard_name}」：${location.reason}。系统没有回退到 PDF 封面；请先在来源资料中确认业绩页。`);
          setImageError("未定位到可交给 VLM 的产品业绩页，因此尚未调用 VLM。请确认资料中是否包含历史净值曲线。");
          return;
        }
        const pageNumber = location.page_number ?? 1;
        const preview = await kbGetSourcePreview(sourceFileId, Math.max(0, pageNumber - 1));
        if (requestId !== calibrationRequestIdRef.current) return;
        if (!selectImage(preview)) {
          throw new Error("来源资料已读取，但预览图格式不受支持");
        }
        setCalibrationSourcePage(location.is_pdf ? pageNumber : undefined);
        setCalibrationSourceIsPdf(location.is_pdf);
        setImportMessage(
          `正在复核「${product.standard_name}」：已定位并载入${location.is_pdf ? `来源 PDF 第 ${pageNumber} 页` : "来源原图"}${location.reason ? `（${location.reason}）` : ""}。尚未调用 VLM；点击“自动定位并识别产品净值”后才会执行 VLM/CV。`,
        );
      }
      if (!sourceFileId) {
        setImportMessage(`正在复核「${product.standard_name}」：未关联原始资料。请返回 FOF 工作台补充资料后再校准。`);
      }
    } catch (error) {
      if (requestId !== calibrationRequestIdRef.current) return;
      setImportMessage(`已载入「${product.standard_name}」的候选净值；原图读取失败时请返回 FOF 工作台重新上传资料。`);
      setAnalysisError(error instanceof Error ? error.message : "读取待复核产品失败");
    }
  }, [resetImage, selectImage, setImageError, setProductName]);

  const handleSaveCalibratedNav = useCallback(async (): Promise<void> => {
    if (!calibratingProductId) return;
    if (calibrationConflictReasons.length > 0 && !manualConflictConfirmed) {
      setAnalysisError("请先在原图核对冲突点，并勾选人工复核确认后再保存。");
      return;
    }
    const reviewBlockMessage = getReviewBlockMessage(
      [],
      true,
      chartReviewReasons,
      chartReviewConfirmed,
    );
    if (reviewBlockMessage) {
      setAnalysisError(reviewBlockMessage);
      return;
    }
    const productId = calibratingProductId;
    const calibrationRequestId = calibrationRequestIdRef.current;
    const saveRequestId = ++calibrationSaveRequestIdRef.current;
    const isCurrentSave = (): boolean => (
      saveRequestId === calibrationSaveRequestIdRef.current
      && calibrationRequestId === calibrationRequestIdRef.current
    );
    setIsSavingCalibration(true);
    try {
      const points = parseNavText(navText);
      await kbReplaceNavSeries(productId, points, {
        frequency,
        sourceFileId: calibrationSourceFileId,
      });
      if (!isCurrentSave()) return;
      const result = await analyzeNav(points, frequency, riskFreeRate);
      if (!isCurrentSave()) return;
      setAnalysisResult(result);
      setWorkflowStep(1);
      window.dispatchEvent(new Event("kb:product-updated"));
      setImportMessage(`已确认并保存 ${points.length} 条净值；最大回撤 ${percentage(result.metrics.maximum_drawdown)}，现已进入研究。`);
      setAnalysisError(undefined);
    } catch (error) {
      if (!isCurrentSave()) return;
      setAnalysisError(error instanceof Error ? error.message : "保存校准净值失败");
    } finally {
      if (isCurrentSave()) setIsSavingCalibration(false);
    }
  }, [calibratingProductId, calibrationConflictReasons.length, calibrationSourceFileId, chartReviewConfirmed, chartReviewReasons.length, frequency, manualConflictConfirmed, navText, riskFreeRate]);

  async function handleImageImport(
    lineKind: "product" | "benchmark" = "product",
    lineColor?: string,
  ): Promise<void> {
    setImportMessage(undefined);
    const result = await runImageImport(lineKind, imageExtractionFrequency, lineColor);
    if (!result) return;

    if (result.navText !== undefined) {
      setNavText(result.navText);
      if (result.extractionFrequency) setFrequency(result.extractionFrequency);
      setCandidateCurve((result.candidateCurve ?? []).map((point) => ({ x: point.x_ratio, y: point.y_ratio })));
      setCandidateConfidence(result.confidence);
      setManualConflictConfirmed(false);
      setChartReviewReasons(["manual_digitization_candidate"]);
      setChartReviewConfirmed(false);
      setIsCurveEditMode(false);
      setAutoFitMessage(undefined);
      // A freshly digitized product series invalidates any prior benchmark.
      setBenchmarkCumulativeReturn(undefined);
      setAnalysisResult(undefined);
    }
    if (result.benchmarkReturn !== undefined) {
      setBenchmarkCumulativeReturn(result.benchmarkReturn);
    }
    if (result.message) setImportMessage(result.message);
  }

  const handleSmartImageImport = useCallback(async (): Promise<void> => {
    if (!pendingImage || !imagePreviewUrl) {
      setImageError("请先选择一张净值曲线图片。");
      return;
    }
    const requestId = ++smartExtractionRequestIdRef.current;
    setIsSmartDigitizing(true);
    setImageError(undefined);
    setImportMessage(undefined);
    try {
      // VLM finds the plot/axes/colour; CV traces the pixels.  No manual
      // dates or min/max values are required when the chart has usable ticks.
      const response = await extractSingleImage(pendingImage, undefined, { useVlm: true });
      if (requestId !== smartExtractionRequestIdRef.current) return;
      const result = response.result;
      const reviewRequired = result.review_required !== false;
      const reviewReasons = reviewRequired
        ? (result.review_reasons?.length ? result.review_reasons : ["review_required"])
        : [];
      const sourceLabel = calibrationSourceIsPdf && calibrationSourcePage
        ? `当前来源：PDF 第 ${calibrationSourcePage} 页。`
        : "当前来源：原图。";
      const vlmLabel = result.vlm_succeeded
        ? `VLM：已调用并成功（${result.vlm_model ?? result.vlm_provider ?? "已配置模型"}）。`
        : result.vlm_attempted
          ? `VLM：已调用但失败${result.vlm_error ? `（${result.vlm_error}）` : ""}。`
          : "VLM：未调用。";
      const original = new Image();
      original.src = imagePreviewUrl;
      await new Promise<void>((resolve, reject) => { original.onload = () => resolve(); original.onerror = () => reject(new Error("原图无法读取")); });
      if (requestId !== smartExtractionRequestIdRef.current) return;
      const [offsetX = 0, offsetY = 0] = result.crop_offset ?? [];
      const area = result.plot_area;
      if (area) {
        applyLocatedChart({
          startXRatio: (area.left + offsetX) / original.naturalWidth,
          endXRatio: (area.right + offsetX) / original.naturalWidth,
          topYRatio: (area.top + offsetY) / original.naturalHeight,
          bottomYRatio: (area.bottom + offsetY) / original.naturalHeight,
          navMin: result.structure?.y_range?.[0],
          navMax: result.structure?.y_range?.[1],
        });
      }
      const tickLabels = result.structure?.x_ticks ?? [];
      const startDate = tickLabels[0] ? parseAxisDate(tickLabels[0]) : undefined;
      const endDate = tickLabels.length > 1 ? parseAxisDate(tickLabels[tickLabels.length - 1]!) : undefined;
      if (startDate) setImageStartDate(startDate);
      if (endDate) setImageEndDate(endDate);
      const locatedProduct = result.structure?.curves.find((item) => item.is_benchmark !== true && typeof item.color_hex === "string");
      const locatedColor = typeof locatedProduct?.color_hex === "string" ? locatedProduct.color_hex : undefined;
      if (locatedColor) setCurveLineColor(locatedColor.toUpperCase());

      if (result.needs_color_pick) {
        setIsPickingCurveColor(true);
        setIsCurveEditMode(false);
        const frameLabel = result.plot_area_source === "vlm"
          ? "VLM 已定位净值图框"
          : result.plot_area_source === "cv"
            ? "已保留 CV 候选图框"
            : "尚未定位图框";
        const axisNote = result.structure?.y_range?.length === 2 ? "" : " 若纵轴范围也未识别，请在高级校准中填写最小值和最大值。";
        setImportMessage(`${sourceLabel}${vlmLabel}${frameLabel}。请直接点击产品曲线取色；若日期首尾未自动填入，再放大图片分别标记首日期和末日期，无需逐点画线。${axisNote}`);
        return;
      }

      // The backend may intentionally return a recoverable error together
      // with a VLM/CV plot frame.  Apply that frame before surfacing the
      // message so the user can immediately take one colour sample instead
      // of losing the location and having to redraw it.
      if (result.error) {
        setIsCurveEditMode(false);
        throw new Error(`${sourceLabel}${vlmLabel}${result.error}。${area ? "已显示候选图框：" : ""}请从产品线上取色；若日期首尾未自动填入，再放大图片分别标记它们。`);
      }

      const curve = result.curves.find((item) => !item.is_benchmark && item.points.length >= 3);
      if (!curve) {
        setIsCurveEditMode(false);
        const frameLabel = result.plot_area_source === "vlm"
          ? "已显示 VLM 定位的图框"
          : result.plot_area_source === "cv"
            ? "已显示 CV 候选图框"
            : "尚未定位图框";
        throw new Error(`${sourceLabel}${vlmLabel}${frameLabel}。请先在产品线上点一下“从图上取色”；若首尾日期未自动填入，放大原图后分别点“标记首日期”和“标记末日期”，再按所选颜色识别。`);
      }
      const valid = curve.points.filter((point) => /^\d{4}-\d{2}-\d{2}$/.test(point.date) && typeof point.value === "number" && point.value > 0);
      if (valid.length < 3) {
        setIsCurveEditMode(false);
        const yRange = result.structure?.y_range;
        const hasValidYRange = Boolean(yRange && yRange.length === 2 && yRange[1]! > yRange[0]!);
        const frameLabel = result.plot_area_source === "vlm"
          ? `已显示 VLM 定位的图框${hasValidYRange ? "与纵轴范围" : ""}`
          : result.plot_area_source === "cv"
            ? "已显示 CV 候选图框"
            : "尚未定位图框";
        throw new Error(`${sourceLabel}${vlmLabel}${frameLabel}。请在产品线上点一下“从图上取色”；若首尾日期未自动填入，放大原图后分别标记首尾日期，再按所选颜色识别。`);
      }
      const candidate = valid.map((point) => ({
        x: Math.max(0, Math.min(1, (point.x_px + offsetX) / original.naturalWidth)),
        y: Math.max(0, Math.min(1, (point.y_px + offsetY) / original.naturalHeight)),
      }));
      const extractedFrequency: DataFrequency = result.frequency === "daily" || result.frequency === "monthly" || result.frequency === "weekly"
        ? result.frequency : "weekly";
      const text = formatNavText(valid.map((point) => ({ observation_date: point.date, net_asset_value: point.value! })));
      setNavText(text);
      setFrequency(extractedFrequency);
      setCandidateCurve(candidate);
      setCandidateConfidence(result.confidence);
      setManualConflictConfirmed(false);
      setChartReviewReasons(reviewReasons);
      setChartReviewConfirmed(false);
      setIsCurveEditMode(false);
      setAnalysisResult(undefined);
      if (reviewRequired) {
        setImportMessage(`${sourceLabel}${vlmLabel}CV 已在定位图框内追踪 ${valid.length} 个${extractedFrequency === "weekly" ? "周频" : extractedFrequency === "monthly" ? "月频" : "日频"}候选点；置信度 ${(result.confidence * 100).toFixed(0)}%。${reviewReasons.join("、")}。请先对照原图完成人工校准并确认，不能直接进入研究。`);
      } else {
        setImportMessage(`${sourceLabel}${vlmLabel}CV 已在定位图框内追踪 ${valid.length} 个${extractedFrequency === "weekly" ? "周频" : extractedFrequency === "monthly" ? "月频" : "日频"}点；置信度 ${(result.confidence * 100).toFixed(0)}%。请直接核对虚线。`);
        const reviewBlockMessage = getReviewBlockMessage(
          calibrationConflictReasons,
          manualConflictConfirmed,
          [],
          true,
        );
        if (reviewBlockMessage) {
          setAnalysisError(reviewBlockMessage);
          setImportMessage("识别结果已写入候选区；请先完成产品资料冲突复核，再计算。");
          return;
        }
        try {
          const analysis = await analyzeNav(parseNavText(text), extractedFrequency, riskFreeRate);
          if (requestId === smartExtractionRequestIdRef.current) setAnalysisResult(analysis);
        } catch {
          if (requestId === smartExtractionRequestIdRef.current) setAnalysisResult(undefined);
        }
      }
    } catch (error) {
      if (requestId !== smartExtractionRequestIdRef.current) return;
      setImageError(error instanceof Error ? error.message : "智能识别失败，请打开高级校准。");
    } finally {
      if (requestId === smartExtractionRequestIdRef.current) setIsSmartDigitizing(false);
    }
  }, [applyLocatedChart, calibrationConflictReasons, calibrationSourceIsPdf, calibrationSourcePage, imagePreviewUrl, manualConflictConfirmed, pendingImage, riskFreeRate, setImageEndDate, setImageStartDate]);

  const handleCalibrationImageClick = useCallback(async (event: React.MouseEvent<HTMLImageElement>): Promise<void> => {
    if (!isPickingCurveColor || !pendingImage) {
      void markDateAnchor(event);
      return;
    }
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width * event.currentTarget.naturalWidth;
    const y = (event.clientY - bounds.top) / bounds.height * event.currentTarget.naturalHeight;
    try {
      const picked = await sampleColorAtPixel(pendingImage, x, y, 4);
      setCurveLineColor(picked.color_hex.toUpperCase());
      setImportMessage(`已从原图取色 ${picked.color_hex.toUpperCase()}；若自动识别未补齐日期和纵轴范围，请打开高级校准，填写后按“按当前校准识别产品净值”。`);
    } catch (error) {
      setImageError(error instanceof Error ? error.message : "图上取色失败");
    } finally {
      setIsPickingCurveColor(false);
    }
  }, [isPickingCurveColor, markDateAnchor, pendingImage]);

  const applyCurveToNav = useCallback((nextCurve: CandidateCurvePoint[]): void => {
    const hasAxisCalibration = topYRatio !== undefined
      && bottomYRatio !== undefined
      && imageNavMin !== undefined
      && imageNavMax !== undefined
      && bottomYRatio > topYRatio
      && imageNavMax > imageNavMin;
    setNavText((previous) => {
      try {
        const points = parseNavText(previous);
        if (points.length !== nextCurve.length) return previous;
        if (hasAxisCalibration) {
          return formatNavText(points.map((point, index) => {
            const constrainedY = Math.max(topYRatio!, Math.min(bottomYRatio!, nextCurve[index]!.y));
            const normalized = (bottomYRatio! - constrainedY) / (bottomYRatio! - topYRatio!);
            return { ...point, net_asset_value: imageNavMin! + normalized * (imageNavMax! - imageNavMin!) };
          }));
        } else {
          // Before axis calibration, retain the current NAV range and only
          // apply the smooth relative shape correction made on the image.
          const sourceCurve = candidateCurvePoints;
          const yValues = sourceCurve.map((curvePoint) => curvePoint.y);
          const yTop = Math.min(...yValues);
          const yBottom = Math.max(...yValues);
          const navValues = points.map((navPoint) => navPoint.net_asset_value);
          const navMin = Math.min(...navValues);
          const navMax = Math.max(...navValues);
          return formatNavText(points.map((point, index) => {
            const normalized = yBottom > yTop ? (yBottom - nextCurve[index]!.y) / (yBottom - yTop) : 0.5;
            return { ...point, net_asset_value: navMin + Math.max(0, Math.min(1, normalized)) * (navMax - navMin) };
          }));
        }
      } catch {
        return previous;
      }
    });
  }, [bottomYRatio, candidateCurvePoints, imageNavMax, imageNavMin, topYRatio]);

  const handleCandidatePointMove = useCallback((index: number, yRatio: number): void => {
    const source = candidateCurvePoints;
    if (!source[index]) return;
    const hasAxisCalibration = topYRatio !== undefined && bottomYRatio !== undefined;
    const constrainedY = hasAxisCalibration
      ? Math.max(topYRatio!, Math.min(bottomYRatio!, yRatio))
      : Math.max(0, Math.min(1, yRatio));
    const middle = Math.floor((source.length - 1) / 2);
    const delta = constrainedY - source[index].y;
    const nextCurve = source.map((point, pointIndex) => {
      // Start / middle / end anchors control neighbouring sections with a
      // linear falloff, so three edits reshape the whole curve smoothly.
      let weight = 0;
      if (index === 0) weight = pointIndex <= middle ? 1 - pointIndex / Math.max(middle, 1) : 0;
      else if (index === source.length - 1) weight = pointIndex >= middle ? (pointIndex - middle) / Math.max(source.length - 1 - middle, 1) : 0;
      else weight = 1 - Math.abs(pointIndex - middle) / Math.max(middle, source.length - 1 - middle, 1);
      return { ...point, y: Math.max(0, Math.min(1, point.y + delta * weight)) };
    });
    setCandidateCurve(nextCurve);
    applyCurveToNav(nextCurve);
  }, [applyCurveToNav, bottomYRatio, candidateCurvePoints, topYRatio]);

  const handleAutoFitCurve = useCallback(async (): Promise<void> => {
    if (!imagePreviewUrl || candidateCurvePoints.length < 3) return;
    setIsAutoFittingCurve(true);
    setAutoFitMessage(undefined);
    try {
      const result = await autoFitCurveToSelectedLine(imagePreviewUrl, candidateCurvePoints, curveLineColor);
      setCandidateCurve(result.curve);
      applyCurveToNav(result.curve);
      setIsCurveEditMode(false);
      setAutoFitMessage(`已自动贴合所选颜色曲线：匹配 ${result.matched} 个位置，中位偏差 ${(result.medianResidual * 100).toFixed(1)}% 图高。仅在局部不贴合时再用三个锚点微调。`);
    } catch (error) {
      setAutoFitMessage(error instanceof Error ? error.message : "自动贴合失败，请改用三个锚点微调");
    } finally {
      setIsAutoFittingCurve(false);
    }
  }, [applyCurveToNav, candidateCurvePoints, curveLineColor, imagePreviewUrl]);

  function clearAll(): void {
    setNavText("");
    setAnalysisResult(undefined);
    setAnalysisError(undefined);
    setImportMessage(undefined);
    setBenchmarkCumulativeReturn(undefined);
    setReportedCumulativeReturn(undefined);
    setCandidateCurve([]);
    setCandidateConfidence(undefined);
    setChartReviewReasons([]);
    setChartReviewConfirmed(false);
    setIsCurveEditMode(false);
    setAutoFitMessage(undefined);
    resetIdentity();
    resetImage();
  }

  function handleDefaultsChange(next: ResearchDefaults): void {
    setResearchDefaults(next);
    setFrequency(next.frequency);
    setRiskFreeRate(next.riskFreeRate);
    try {
      localStorage.setItem(DEFAULTS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Storage may be unavailable; defaults still apply for this session.
    }
  }

  function handleEngineChange(next: AnalysisEngineConfig): void {
    setEngineConfig(next);
    try {
      localStorage.setItem(ENGINE_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Storage may be unavailable; engine config still applies for this session.
    }
  }

  function handleSaveHistory(): void {
    if (!analysisResult) return;
    saveToHistory({
      productName,
      frequency,
      navCount,
      navText,
      analysisResult,
      sourceText,
      aiReport,
    });
    setImportMessage("已保存到历史记录。");
  }

  async function handleViewHistory(record: HistoryRecord): Promise<void> {
    try {
      const points = parseNavText(record.nav_text);
      if (points.length < 2) throw new Error("历史记录中的净值数据不足两条");
      setIsHistoryOpen(false);
      setActiveView("tools");
      setProductName(record.product_name);
      setFrequency(record.frequency);
      setNavText(record.nav_text);
      setSourceText(record.source_text ?? "");
      setAiReport(record.ai_report);
      setAnalysisError(undefined);
      setIsAnalyzing(true);
      const result = await analyzeNav(points, record.frequency, riskFreeRate);
      setAnalysisResult(result);
      setWorkflowStep(1);
      setImportMessage(`已恢复「${record.product_name}」的历史研究看板；指标已按保存的净值序列重新计算。`);
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "恢复历史研究失败");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <ConfigProvider theme={themeConfig}>
      <Layout style={{ minHeight: "100vh" }}>
        {/* View toggle bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "10px 24px",
            borderBottom: "1px solid var(--serif-border, #f0f0f0)",
            background: "var(--serif-card, #fff)",
            position: "sticky",
            top: 0,
            zIndex: 100,
          }}
        >
          <Space size={16}>
            <Title level={5} style={{ margin: 0, fontFamily: FONT_DISPLAY, color: "var(--serif-foreground)" }}>
              私募 CTA 研究平台
            </Title>
            <Space size={4}>
              <Button
                size="small"
                type={activeView === "workbench" ? "primary" : "text"}
                ghost={activeView === "workbench"}
                onClick={() => setActiveView("workbench")}
              >
                FOF 工作台
              </Button>
              <Button
                size="small"
                type={activeView === "tools" ? "primary" : "text"}
                ghost={activeView === "tools"}
                onClick={() => setActiveView("tools")}
              >
                研究工具
              </Button>
            </Space>
          </Space>
          <Space size={8}>
            <Dropdown menu={{ items: [
              { key: "factor", label: "因子库", onClick: () => setFactorLibOpen(true) },
              { key: "chart", label: "图表提取", onClick: () => setChartExtractOpen(true) },
              { type: "divider" },
              { key: "settings", label: "设置", onClick: () => setSettingsOpen(true) },
            ] }}>
              <Button size="small">工具</Button>
            </Dropdown>
            <Button size="small" onClick={() => setIsHistoryOpen(true)}>历史 {historyRecords.length}</Button>
          </Space>
        </div>

        {activeView === "workbench" ? (
          <div style={{ flex: 1, width: "100%", minWidth: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <FofWorkbench onCalibrateProduct={handleCalibrateProduct} />
          </div>
        ) : (
        <Content style={{ maxWidth: 1240, width: "100%", margin: "32px auto 64px", padding: "0 24px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
            marginBottom: 20,
          }}
        >
          <div>
            <Title
              level={4}
              style={{
                fontFamily: FONT_DISPLAY,
                margin: 0,
                color: "var(--serif-foreground)",
              }}
            >
              单产品研究
            </Title>
            <span className="masthead-subtitle" style={{ color: "var(--serif-muted-foreground)" }}>
              校准净值后，生成因子分析与研究报告
            </span>
          </div>
        </div>
        <Card size="small" style={{ marginBottom: 20 }}>
          <div>
            <Space wrap size={12}>
              <Text strong>{productName || "尚未选择产品"}</Text>
              <Text type="secondary">{navCount} 个净值点</Text>
              <Text type={analysisResult ? "success" : "warning"}>{analysisResult ? "已完成净值计算，可继续归因" : "净值待校验"}</Text>
              {workflowStep > 0 && <Button size="small" onClick={() => setWorkflowStep(0)}>返回数据校准</Button>}
            </Space>
            {sourceText.trim() && (
              <details style={{ marginTop: 8 }}>
                <summary style={{ cursor: "pointer", color: "var(--serif-muted-foreground)", fontSize: 12 }}>资料线索（已提取，点击查看）</summary>
                <Paragraph type="secondary" style={{ whiteSpace: "pre-wrap", margin: "8px 0 0", fontSize: 12 }}>{sourceText}</Paragraph>
              </details>
            )}
          </div>
        </Card>
        <Row gutter={[24, 24]}>
          {/* Left column: data ingestion and review */}
          <Col xs={24} style={{ display: workflowStep === 0 ? "flex" : "none", flexDirection: "column" }}>
            <Card
              title="净值与资料"
              style={{ flex: 1 }}
              extra={
                <Button size="small" onClick={clearAll}>
                  清空
                </Button>
              }
            >
              <div style={{ display: isImportWorkspaceOpen ? "block" : "none" }}>
              <ImportWorkspaceOpenHeader />
              <div style={{ marginBottom: 8 }}>
                <StepSection step={1} title="上传净值资料" visible={workflowStep === 0}>
                  <Alert
                    type="info"
                    showIcon
                    message="资料上传已统一至 FOF 工作台"
                    description="请在 FOF 工作台上传净值表、曲线图片或周报；选择产品后可在此页复核净值并生成单产品研究。"
                    action={<Button size="small" type="primary" onClick={() => setActiveView("workbench")}>前往 FOF 工作台</Button>}
                  />
                </StepSection>

                <NavImportReviewPanel
                  navText={navText}
                  onNavTextChange={(value) => {
                    setNavText(value);
                    if (calibrationConflictReasons.length > 0) setManualConflictConfirmed(false);
                    if (chartReviewReasons.length > 0) setChartReviewConfirmed(false);
                  }}
                  conflictReasons={calibrationConflictReasons}
                  conflictConfirmed={manualConflictConfirmed}
                  onConflictConfirmedChange={setManualConflictConfirmed}
                  chartReviewReasons={chartReviewReasons}
                  chartReviewConfirmed={chartReviewConfirmed}
                  onChartReviewConfirmedChange={setChartReviewConfirmed}
                  isCalibrating={Boolean(calibratingProductId)}
                  isSaving={isSavingCalibration}
                  isAnalyzing={isAnalyzing}
                  canAnalyze={canAnalyzeText}
                  onSaveReviewed={() => void handleSaveCalibratedNav()}
                  onAnalyze={() => void handleAnalysis()}
                  analysisError={analysisError}
                  onDismissAnalysisError={() => setAnalysisError(undefined)}
                  navCount={navCount}
                />

                {pendingImage && <>
                  <Divider titlePlacement="left">图表识别与复核</Divider>
                  <ChartPreviewReview
                    src={imagePreviewUrl}
                    anchors={{ startXRatio, endXRatio, topYRatio, bottomYRatio }}
                    candidateCurve={candidateCurvePoints}
                    editable={isCurveEditMode}
                    isPicking={isPickingCurveColor}
                    onImageClick={(event) => void handleCalibrationImageClick(event)}
                    activeAnchor={activeDateAnchor}
                    onActiveAnchorChange={setActiveDateAnchor}
                    onToggleColorPicker={() => {
                      setIsPickingCurveColor((value) => !value);
                      setActiveDateAnchor(undefined);
                    }}
                    canMark={Boolean(pendingImage)}
                    zoomOpen={isZoomModalOpen}
                    onZoomOpen={() => { setZoomLevel(1); setIsZoomModalOpen(true); }}
                    onZoomClose={() => { setIsZoomModalOpen(false); setActiveDateAnchor(undefined); }}
                    zoomLevel={zoomLevel}
                    onZoomLevelChange={setZoomLevel}
                    onPointMove={handleCandidatePointMove}
                    navCount={navCount}
                    frequency={frequency}
                    confidence={candidateConfidence}
                    maximumDrawdown={candidateMaximumDrawdown === undefined ? undefined : percentage(candidateMaximumDrawdown)}
                    isAutoFitting={isAutoFittingCurve}
                    onAutoFit={() => void handleAutoFitCurve()}
                    onToggleEdit={() => setIsCurveEditMode((value) => !value)}
                    autoFitMessage={autoFitMessage}
                    onDismissAutoFit={() => setAutoFitMessage(undefined)}
                    showModal
                  />

                  <AdvancedCalibrationPanel
                  visible={workflowStep === 0 && isCurveEditMode}
                  valueMode={imageValueMode}
                  onValueModeChange={handleImageValueModeChange}
                  activeAnchor={activeDateAnchor}
                  onActiveAnchorChange={setActiveDateAnchor}
                  canMark={Boolean(pendingImage)}
                  anchors={{ start: startXRatio, end: endXRatio, top: topYRatio, bottom: bottomYRatio }}
                  startDate={imageStartDate}
                  onStartDateChange={(value) => { setImageStartDate(value); clearFieldIssue("首日期", "start"); }}
                  endDate={imageEndDate}
                  onEndDateChange={(value) => { setImageEndDate(value); clearFieldIssue("末日期", "end"); }}
                  navMin={imageNavMin}
                  onNavMinChange={(value) => { setImageNavMin(value); clearFieldIssue("最小值", "bottom"); }}
                   navMax={imageNavMax}
                   onNavMaxChange={(value) => { setImageNavMax(value); clearFieldIssue("最大值", "top"); }}
                   canStartRecognition={Boolean(
                     pendingImage && imageStartDate && imageEndDate
                     && imageNavMin !== undefined && imageNavMax !== undefined
                     && imageNavMax > imageNavMin,
                   )}
                   isDigitizing={isDigitizing}
                   onStartRecognition={() => void handleImageImport("product", curveLineColor)}
                   missingFields={missingFields}
                  ocrHints={ocrHints}
                  />

                  <ImageExtractionActions
                    pendingImage={Boolean(pendingImage)}
                    isDigitizing={isDigitizing}
                    isSmartDigitizing={isSmartDigitizing}
                    imageError={imageError}
                    importMessage={importMessage}
                    curveLineColor={curveLineColor}
                    onCurveLineColorChange={setCurveLineColor}
                    isPickingCurveColor={isPickingCurveColor}
                    onToggleColorPicker={() => {
                      setIsPickingCurveColor((value) => !value);
                      setActiveDateAnchor(undefined);
                    }}
                    reportedCumulativeReturn={reportedCumulativeReturn}
                    onReportedCumulativeReturnChange={setReportedCumulativeReturn}
                    imageExtractionFrequency={imageExtractionFrequency}
                    onImageExtractionFrequencyChange={setImageExtractionFrequency}
                    onSmartExtract={() => void handleSmartImageImport()}
                    onOpenAdvancedCalibration={() => setIsCurveEditMode(true)}
                    productName={productName}
                    onProductNameChange={setProductName}
                    isRecognizingProduct={isRecognizingProduct}
                  />

                  {reportExtraction && <ReportExtractionView report={reportExtraction} />}
                </>}
              </div>
              </div>
            </Card>
          </Col>

          {/* Results */}
          <Col xs={24} lg={24} style={{ display: workflowStep === 1 ? "flex" : "none", flexDirection: "column" }}>
            <PerformanceResultCard
              analysis={analysisResult}
              error={analysisError}
              disclosedDifference={disclosedDifference}
              benchmarkReturn={benchmarkCumulativeReturn}
              onExport={() => { if (analysisResult) void exportNavXlsx(navText, analysisResult.metrics, productName); }}
              onSaveHistory={handleSaveHistory}
            />
          </Col>
        </Row>

        {/* AI Strategy Attribution Analysis */}
        {workflowStep === 1 && <AnalysisPanel navPoints={parsedNavPoints} frequency={frequency} productName={productName} strategyHint={sourceText} qualityOverrideConfirmed={manualConflictConfirmed} engineConfig={engineConfig} initialReport={aiReport} onReport={setAiReport} />}

        {/* Quantitative Factor Attribution (L1) */}
        {workflowStep === 2 && <>
          <FactorAttributionPanel navPoints={parsedNavPoints} frequency={frequency} />
          <ExternalFactorLibraryPanel />
        </>}

      </Content>
        )}

        {/* Drawers accessible from both views */}
        <HistoryDrawer
          open={isHistoryOpen}
          onClose={() => setIsHistoryOpen(false)}
          records={historyRecords}
          onDelete={handleDeleteHistory}
          onView={(record) => { void handleViewHistory(record); }}
        />

        <FactorLibraryDrawer open={factorLibOpen} onClose={() => setFactorLibOpen(false)} />

        <ChartExtractDrawer
          open={chartExtractOpen}
          onClose={() => setChartExtractOpen(false)}
          onApplyToWorkspace={(navTextValue, confirmed) => { void handleChartApply(navTextValue, confirmed); }}
        />

        <SettingsDrawer
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          accentKey={accentKey}
          onAccentChange={setAccentKey}
          defaults={researchDefaults}
          onDefaultsChange={handleDefaultsChange}
          engine={engineConfig}
          onEngineChange={handleEngineChange}
          onClearHistory={handleClearAllHistory}
        />
    </Layout>
    </ConfigProvider>
  );
}

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("无法找到 #root 挂载点");
createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
