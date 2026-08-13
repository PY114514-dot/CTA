/**
 * useProductIdentity — encapsulates product recognition, strategy profile
 * building, multi-product report extraction, and PaddleOCR document parsing.
 */
import { useState } from "react";
import {
  buildProductStrategyProfile,
  extractDocumentWithPaddleOcr,
  extractMultiProductReport,
  recognizeProductImage,
  type MultiProductReportResponse,
  type PaddleOcrDocumentResponse,
  type ProductStrategyProfileResponse,
} from "../api";
import { percentage } from "../utils/format";
import { inferIdentityFromFilename } from "../utils/fileIdentity";

const ACCEPTED_DOCUMENT_TYPES_SET = new Set(["application/pdf", "image/png", "image/jpeg", "image/webp"]);
const MAX_DOCUMENT_SIZE_MB = 50;

export interface ProductIdentityState {
  productName: string;
  setProductName: React.Dispatch<React.SetStateAction<string>>;
  sourceText: string;
  setSourceText: React.Dispatch<React.SetStateAction<string>>;
  strategyProfile: ProductStrategyProfileResponse | undefined;
  profileMessage: string | undefined;
  isRecognizingProduct: boolean;
  isBuildingProfile: boolean;
  reportExtraction: MultiProductReportResponse | undefined;
  paddleOcrResult: PaddleOcrDocumentResponse | undefined;
  paddleOcrError: string | undefined;
  isPaddleOcrParsing: boolean;

  identifyProductFromImage: (file: File) => Promise<void>;
  identifyMultiProductReport: (file: File) => Promise<MultiProductReportResponse | undefined>;
  handleBuildStrategyProfile: () => Promise<void>;
  handlePaddleOcrDocument: (file: File) => Promise<void>;
  resetIdentity: () => void;
}

export function useProductIdentity(): ProductIdentityState {
  const [productName, setProductName] = useState("");
  const [sourceText, setSourceText] = useState("");
  const [strategyProfile, setStrategyProfile] = useState<ProductStrategyProfileResponse>();
  const [profileMessage, setProfileMessage] = useState<string>();
  const [isRecognizingProduct, setIsRecognizingProduct] = useState(false);
  const [isBuildingProfile, setIsBuildingProfile] = useState(false);
  const [reportExtraction, setReportExtraction] = useState<MultiProductReportResponse>();
  const [paddleOcrResult, setPaddleOcrResult] = useState<PaddleOcrDocumentResponse>();
  const [paddleOcrError, setPaddleOcrError] = useState<string>();
  const [isPaddleOcrParsing, setIsPaddleOcrParsing] = useState(false);

  async function identifyProductFromImage(file: File): Promise<void> {
    setIsRecognizingProduct(true);
    setProfileMessage(undefined);
    try {
      const filenameIdentity = inferIdentityFromFilename(file.name);
      setProductName((previous) => previous.trim() || filenameIdentity.productName);
      if (filenameIdentity.managerName) {
        setSourceText((previous) => previous.trim() ? previous : `文件名识别候选：管理人 ${filenameIdentity.managerName}；产品 ${filenameIdentity.productName}。请结合图片内容确认。`);
      }
      const result = await recognizeProductImage(file);
      if (result.product_name_candidate) setProductName((previous) => previous.trim() || result.product_name_candidate!);
      const ocrText = result.ocr_text;
      if (ocrText) {
        setSourceText((previous) => previous.trim() ? `${previous}\n${ocrText}` : ocrText);
      }
      setProfileMessage(result.warnings.join(" ") || undefined);
    } catch (error) {
      setProfileMessage(error instanceof Error ? error.message : "产品名称识别失败，请手动填写。");
    } finally {
      setIsRecognizingProduct(false);
    }
  }

  async function identifyMultiProductReport(file: File): Promise<MultiProductReportResponse | undefined> {
    try {
      const result = await extractMultiProductReport(file);
      setReportExtraction(result);

      if (result.disclosed_metrics.length === 1) {
        const metric = result.disclosed_metrics[0];
        if (!metric) return result;
        const displayName = metric.product_name?.trim() || metric.product_id;

        if (displayName) {
          setProductName((prev) => prev.trim() || displayName);
        }

        const scanSummary = [
          displayName && `产品：${displayName}`,
          metric.strategy && `策略：${metric.strategy}`,
          `区间：${metric.start_date} 至 ${metric.end_date}`,
          `披露累计收益：${percentage(metric.cumulative_return)}`,
          `披露年化收益：${percentage(metric.annualized_return)}`,
          `披露最大回撤：${metric.maximum_drawdown_disclosed === false ? "未披露" : percentage(metric.maximum_drawdown)}`,
        ]
          .filter((line): line is string => Boolean(line))
          .join("；");

        setSourceText((prev) => {
          const trimmed = prev.trim();
          if (!trimmed) return scanSummary;
          if (trimmed.includes(scanSummary)) return trimmed;
          return `${trimmed}\n${scanSummary}`;
        });
      }
      return result;
    } catch (error) {
      setReportExtraction(undefined);
      // eslint-disable-next-line no-console
      console.warn("多产品报告识别未返回结果", error);
      return undefined;
    }
  }

  async function handleBuildStrategyProfile(): Promise<void> {
    if (!productName.trim()) {
      setProfileMessage("请先确认或手动填写产品名称。");
      return;
    }
    setIsBuildingProfile(true);
    setProfileMessage(undefined);
    try {
      const result = await buildProductStrategyProfile(productName.trim(), sourceText);
      setStrategyProfile(result);
    } catch (error) {
      setStrategyProfile(undefined);
      setProfileMessage(error instanceof Error ? error.message : "策略画像生成失败。");
    } finally {
      setIsBuildingProfile(false);
    }
  }

  async function handlePaddleOcrDocument(file: File): Promise<void> {
    if (!ACCEPTED_DOCUMENT_TYPES_SET.has(file.type)) {
      setPaddleOcrError("仅支持 PDF、PNG、JPG/JPEG 或 WebP 文件。 ");
      return;
    }
    if (file.size > MAX_DOCUMENT_SIZE_MB * 1024 * 1024) {
      setPaddleOcrError(`文档大小不能超过 ${MAX_DOCUMENT_SIZE_MB} MB。`);
      return;
    }
    setIsPaddleOcrParsing(true);
    setPaddleOcrError(undefined);
    try {
      const filenameIdentity = inferIdentityFromFilename(file.name);
      setProductName((previous) => previous.trim() || filenameIdentity.productName);
      const result = await extractDocumentWithPaddleOcr(file);
      setPaddleOcrResult(result);
      const extractedText = result.pages.map((page) => page.markdown).join("\n\n").trim();
      if (extractedText) {
        const profileText = extractedText.slice(0, 4800);
        setSourceText((previous) => previous.trim() ? previous : profileText);
        setProfileMessage(
          extractedText.length > profileText.length
            ? "已将文档前 4,800 字写入画像文本；完整 Markdown 保留在下方供核验。"
            : "已将 OCR 文本写入画像输入框；请先人工核验。",
        );
      }
    } catch (error) {
      setPaddleOcrResult(undefined);
      setPaddleOcrError(error instanceof Error ? error.message : "PaddleOCR 文档解析失败。 ");
    } finally {
      setIsPaddleOcrParsing(false);
    }
  }

  function resetIdentity(): void {
    setProductName("");
    setSourceText("");
    setStrategyProfile(undefined);
    setProfileMessage(undefined);
    setReportExtraction(undefined);
  }

  return {
    productName, setProductName,
    sourceText, setSourceText,
    strategyProfile, profileMessage,
    isRecognizingProduct, isBuildingProfile,
    reportExtraction,
    paddleOcrResult, paddleOcrError, isPaddleOcrParsing,
    identifyProductFromImage, identifyMultiProductReport,
    handleBuildStrategyProfile, handlePaddleOcrDocument,
    resetIdentity,
  };
}
