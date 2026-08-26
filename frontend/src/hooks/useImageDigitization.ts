/**
 * useImageDigitization — encapsulates all NAV-chart image state and calibration logic.
 *
 * Manages: image selection/preview, axis calibration (date/value bounds),
 * anchor-line marking with OCR label recognition, and the digitization API call.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  digitizeNavImage,
  ocrImageRegion,
  type DataFrequency,
  type ImageExtractionFrequency,
} from "../api";
import { parseAxisDate, parseAxisNumber } from "../parser/axisLabelParser";
import { clampRatio, formatCumulativeReturn, isIsoDate, isValidDateRange } from "../utils/format";

const ACCEPTED_IMAGE_TYPES_SET = new Set(["image/png", "image/jpeg", "image/webp"]);
const MAX_IMAGE_SIZE_MB = 20;

export type AnchorKind = "start" | "end" | "top" | "bottom";

export interface ImageDigitizationState {
  // Calibration fields
  imageStartDate: string;
  setImageStartDate: React.Dispatch<React.SetStateAction<string>>;
  imageEndDate: string;
  setImageEndDate: React.Dispatch<React.SetStateAction<string>>;
  imageNavMin: number | undefined;
  setImageNavMin: React.Dispatch<React.SetStateAction<number | undefined>>;
  imageNavMax: number | undefined;
  setImageNavMax: React.Dispatch<React.SetStateAction<number | undefined>>;
  imageValueMode: "nav" | "cumulative_return";
  handleImageValueModeChange: (value: "nav" | "cumulative_return") => void;

  // Image file state
  pendingImage: File | undefined;
  imageFileName: string | undefined;
  imagePreviewUrl: string | undefined;
  isDigitizing: boolean;
  imageError: string | undefined;
  setImageError: React.Dispatch<React.SetStateAction<string | undefined>>;

  // Anchor / calibration overlay
  activeDateAnchor: AnchorKind | undefined;
  setActiveDateAnchor: React.Dispatch<React.SetStateAction<AnchorKind | undefined>>;
  isZoomModalOpen: boolean;
  setIsZoomModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
  zoomLevel: number;
  setZoomLevel: React.Dispatch<React.SetStateAction<number>>;
  startXRatio: number | undefined;
  endXRatio: number | undefined;
  topYRatio: number | undefined;
  bottomYRatio: number | undefined;
  missingFields: string[];
  ocrHints: Partial<Record<AnchorKind, string>>;

  // Actions
  selectImage: (file: File) => boolean;
  handleImageImport: (
    lineKind: "product" | "benchmark",
    frequency: ImageExtractionFrequency,
    lineColor?: string,
  ) => Promise<{
    navText?: string;
    benchmarkReturn?: number;
    extractionFrequency?: DataFrequency;
    confidence?: number;
    candidateCurve?: Array<{ x_ratio: number; y_ratio: number }>;
    message?: string;
  } | undefined>;
  markDateAnchor: (event: React.MouseEvent<HTMLImageElement>) => Promise<void>;
  applyLocatedChart: (location: {
    startXRatio: number; endXRatio: number; topYRatio: number; bottomYRatio: number;
    navMin?: number; navMax?: number;
  }) => void;
  clearFieldIssue: (field: string, anchor: AnchorKind) => void;
  resetImage: () => void;
}

export function useImageDigitization(): ImageDigitizationState {
  const [imageStartDate, setImageStartDate] = useState("");
  const [imageEndDate, setImageEndDate] = useState("");
  const [imageNavMin, setImageNavMin] = useState<number>();
  const [imageNavMax, setImageNavMax] = useState<number>();
  const [imageValueMode, setImageValueMode] = useState<"nav" | "cumulative_return">("nav");
  const [pendingImage, setPendingImage] = useState<File>();
  const [imageFileName, setImageFileName] = useState<string>();
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string>();
  const [isDigitizing, setIsDigitizing] = useState(false);
  const [imageError, setImageError] = useState<string>();
  const [activeDateAnchor, setActiveDateAnchor] = useState<AnchorKind>();
  const [isZoomModalOpen, setIsZoomModalOpen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [startXRatio, setStartXRatio] = useState<number>();
  const [endXRatio, setEndXRatio] = useState<number>();
  const [topYRatio, setTopYRatio] = useState<number>();
  const [bottomYRatio, setBottomYRatio] = useState<number>();
  const [missingFields, setMissingFields] = useState<string[]>([]);
  const [ocrHints, setOcrHints] = useState<Partial<Record<AnchorKind, string>>>({});

  // Revoke object URL on unmount
  const imagePreviewUrlRef = useRef(imagePreviewUrl);
  imagePreviewUrlRef.current = imagePreviewUrl;
  useEffect(() => {
    return () => {
      if (imagePreviewUrlRef.current) URL.revokeObjectURL(imagePreviewUrlRef.current);
    };
  }, []);

  function handleImageValueModeChange(value: "nav" | "cumulative_return"): void {
    setImageValueMode(value);
    if (value === "cumulative_return") {
      // Keep the range read from the chart (for example -5% to 25%).
      // Defaults are only useful before any axis value has been supplied.
      setImageNavMin((current) => current ?? 0);
      setImageNavMax((current) => current ?? 100);
    } else {
      setImageNavMin(undefined);
      setImageNavMax(undefined);
    }
  }

  /** Validate and load an image file. Returns true if accepted. */
  const selectImage = useCallback((file: File): boolean => {
    if (!ACCEPTED_IMAGE_TYPES_SET.has(file.type)) {
      setImageError("仅支持 PNG、JPG/JPEG 或 WebP 图片。");
      return false;
    }
    if (file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024) {
      setImageError(`图片大小不能超过 ${MAX_IMAGE_SIZE_MB} MB。`);
      return false;
    }

    setPendingImage(file);
    setImageFileName(file.name);
    setImageError(undefined);

    setImagePreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });

    // Reset calibration
    setStartXRatio(undefined);
    setEndXRatio(undefined);
    setTopYRatio(undefined);
    setBottomYRatio(undefined);
    setActiveDateAnchor(undefined);
    setMissingFields([]);
    setOcrHints({});
    setIsZoomModalOpen(false);
    setZoomLevel(1);

    return true;
  }, []);

  async function recognizeAxisLabel(
    anchor: AnchorKind,
    xRatio: number,
    yRatio: number,
  ): Promise<void> {
    let left: number, right: number, top: number, bottom: number;
    if (anchor === "start" || anchor === "end") {
      left = clampRatio(xRatio - 0.1);
      right = clampRatio(xRatio + 0.06);
      top = clampRatio(yRatio + 0.005);
      bottom = clampRatio(yRatio + 0.12);
    } else {
      if (xRatio < 0.5) {
        left = 0;
        right = clampRatio(xRatio - 0.01);
      } else {
        left = clampRatio(xRatio + 0.01);
        right = 1;
      }
      top = clampRatio(yRatio - 0.04);
      bottom = clampRatio(yRatio + 0.04);
    }

    try {
      const { text } = await ocrImageRegion(pendingImage as File, left, top, right, bottom);
      if (!text) {
        setOcrHints((prev) => ({ ...prev, [anchor]: "未识别到标签文字，请手动填写" }));
        return;
      }
      if (anchor === "start" || anchor === "end") {
        const parsed = parseAxisDate(text);
        if (!parsed) {
          setOcrHints((prev) => ({
            ...prev,
            [anchor]: `识别到"${text.trim().slice(0, 18)}"，但需要完整 YYYY-MM-DD 日期，请手动填写`,
          }));
          return;
        }
        if (anchor === "start") setImageStartDate(parsed);
        else setImageEndDate(parsed);
        setOcrHints((prev) => ({ ...prev, [anchor]: undefined }));
      } else {
        const value = parseAxisNumber(text, imageValueMode);
        if (value === undefined) {
          setOcrHints((prev) => ({
            ...prev,
            [anchor]: `识别到"${text.trim().slice(0, 18)}"，但无法解析为数值，请手动填写`,
          }));
          return;
        }
        if (anchor === "top") setImageNavMax(value);
        else setImageNavMin(value);
        setOcrHints((prev) => ({ ...prev, [anchor]: undefined }));
      }
    } catch {
      setOcrHints((prev) => ({ ...prev, [anchor]: "OCR 请求失败，请手动填写" }));
    }
  }

  const markDateAnchor = useCallback(async (event: React.MouseEvent<HTMLImageElement>): Promise<void> => {
    if (!activeDateAnchor || !pendingImage) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const xRatio = clampRatio((event.clientX - bounds.left) / bounds.width);
    const yRatio = clampRatio((event.clientY - bounds.top) / bounds.height);
    if (activeDateAnchor === "start") setStartXRatio(xRatio);
    else if (activeDateAnchor === "end") setEndXRatio(xRatio);
    else if (activeDateAnchor === "top") setTopYRatio(yRatio);
    else setBottomYRatio(yRatio);

    const naturalWidth = event.currentTarget.naturalWidth;
    const naturalHeight = event.currentTarget.naturalHeight;
    if (naturalWidth && naturalHeight) {
      await recognizeAxisLabel(activeDateAnchor, xRatio, yRatio);
    }
    setActiveDateAnchor(undefined);
  }, [activeDateAnchor, pendingImage, imageValueMode]);

  /** Apply a VLM-located plot frame so the same overlay is reusable for review. */
  const applyLocatedChart = useCallback((location: {
    startXRatio: number; endXRatio: number; topYRatio: number; bottomYRatio: number;
    navMin?: number; navMax?: number;
  }): void => {
    setStartXRatio(clampRatio(location.startXRatio));
    setEndXRatio(clampRatio(location.endXRatio));
    setTopYRatio(clampRatio(location.topYRatio));
    setBottomYRatio(clampRatio(location.bottomYRatio));
    if (location.navMin !== undefined) setImageNavMin(location.navMin);
    if (location.navMax !== undefined) setImageNavMax(location.navMax);
  }, []);

  /** Run the digitization API. Returns extracted data or undefined on validation failure. */
  async function handleImageImport(
    lineKind: "product" | "benchmark",
    frequency: ImageExtractionFrequency,
    lineColor?: string,
  ): Promise<{
    navText?: string;
    benchmarkReturn?: number;
    extractionFrequency?: DataFrequency;
    confidence?: number;
    candidateCurve?: Array<{ x_ratio: number; y_ratio: number }>;
    message?: string;
  } | undefined> {
    setImageError(undefined);

    if (!pendingImage) {
      setImageError("请先选择一张净值曲线图片。");
      return undefined;
    }
    const missing: string[] = [];
    if (!imageStartDate) missing.push("首日期");
    if (!imageEndDate) missing.push("末日期");
    if (imageNavMin === undefined) missing.push("最小值");
    if (imageNavMax === undefined) missing.push("最大值");
    if (missing.length > 0) {
      setMissingFields(missing);
      setImageError(`缺少必填项：${missing.join("、")}。请手动填写，或重新点击对应"标记"按钮由 OCR 读取。`);
      return undefined;
    }
    setMissingFields([]);
    setOcrHints({});
    if (imageNavMin === undefined || imageNavMax === undefined) return undefined;
    if (!isIsoDate(imageStartDate) || !isIsoDate(imageEndDate)) {
      setImageError("日期必须使用 YYYY-MM-DD 格式，例如 2025-01-07。");
      return undefined;
    }
    if (!isValidDateRange(imageStartDate, imageEndDate)) {
      setImageError("结束日期不能早于起始日期。");
      return undefined;
    }
    if (imageNavMax <= imageNavMin) {
      setImageError("纵轴最大值必须大于最小值。");
      return undefined;
    }

    setIsDigitizing(true);
    try {
      // A negative axis is a return axis, never a unit-NAV axis.  Convert it
      // before the API validates points so a grey benchmark does not fail on
      // legitimate values such as -5%.
      const effectiveValueMode = imageNavMin < 0 ? "cumulative_return" : imageValueMode;
      const result = await digitizeNavImage(
        pendingImage,
        imageStartDate,
        imageEndDate,
        imageNavMin,
        imageNavMax,
        effectiveValueMode,
        frequency,
        startXRatio,
        endXRatio,
        lineKind,
        topYRatio,
        bottomYRatio,
        lineColor,
      );

      if (result.nav_points.length < 2) {
        setImageError("图片识别返回的净值点不足，请检查坐标范围或更换图片。");
        return undefined;
      }

      const firstPoint = result.nav_points[0];
      const lastPoint = result.nav_points[result.nav_points.length - 1];
      if (!firstPoint || !lastPoint) {
        setImageError("图片识别返回的净值点不足，请检查坐标范围或更换图片。");
        return undefined;
      }
      const firstNav = firstPoint.net_asset_value;
      const lastNav = lastPoint.net_asset_value;
      const cumulativeReturn = firstNav === 0 ? 0 : lastNav / firstNav - 1;

      if (lineKind === "benchmark") {
        return {
          benchmarkReturn: cumulativeReturn,
          extractionFrequency: result.extraction_frequency,
          message: `已提取基准曲线 ${result.nav_points.length} 个候选点，累计收益约 ${formatCumulativeReturn(cumulativeReturn)}。请与图例及披露基准核验。`,
        };
      }

      const navText = result.nav_points
        .map((point) => `${point.observation_date},${point.net_asset_value.toFixed(4)}`)
        .join("\n");
      const ocrNote = result.ocr_text ? ` OCR 读取的标签文字：${result.ocr_text.slice(0, 160)}` : "";
      const warningNote = result.warnings[0] ? ` ${result.warnings[0]}` : "";
      return {
        navText,
        extractionFrequency: result.extraction_frequency,
        confidence: result.confidence,
        candidateCurve: result.candidate_curve,
        message: `已从图片按${result.extraction_frequency === "weekly" ? "周频" : result.extraction_frequency === "monthly" ? "月频" : "日频"}提取 ${result.nav_points.length} 个候选点（置信度 ${(result.confidence * 100).toFixed(0)}%），请逐项核验后计算。${warningNote}${ocrNote}`,
      };
    } catch (error) {
      setImageError(error instanceof Error ? error.message : "图片识别失败。");
      return undefined;
    } finally {
      setIsDigitizing(false);
    }
  }

  function clearFieldIssue(field: string, anchor: AnchorKind): void {
    setMissingFields((prev) => (prev.includes(field) ? prev.filter((item) => item !== field) : prev));
    setOcrHints((prev) => (prev[anchor] ? { ...prev, [anchor]: undefined } : prev));
  }

  function resetImage(): void {
    setImagePreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return undefined;
    });
    setPendingImage(undefined);
    setImageFileName(undefined);
    setImageError(undefined);
    setImageStartDate("");
    setImageEndDate("");
    setImageNavMin(undefined);
    setImageNavMax(undefined);
    setStartXRatio(undefined);
    setEndXRatio(undefined);
    setTopYRatio(undefined);
    setBottomYRatio(undefined);
    setMissingFields([]);
    setOcrHints({});
    setActiveDateAnchor(undefined);
    setIsZoomModalOpen(false);
    setZoomLevel(1);
  }

  return {
    imageStartDate, setImageStartDate,
    imageEndDate, setImageEndDate,
    imageNavMin, setImageNavMin,
    imageNavMax, setImageNavMax,
    imageValueMode, handleImageValueModeChange,
    pendingImage, imageFileName, imagePreviewUrl,
    isDigitizing, imageError, setImageError,
    activeDateAnchor, setActiveDateAnchor,
    isZoomModalOpen, setIsZoomModalOpen,
    zoomLevel, setZoomLevel,
    startXRatio, endXRatio, topYRatio, bottomYRatio,
    missingFields, ocrHints,
    selectImage, handleImageImport, markDateAnchor, applyLocatedChart, clearFieldIssue, resetImage,
  };
}
