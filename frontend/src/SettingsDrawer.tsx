import React, { useCallback, useEffect, useState } from "react";
import {
  Alert,
  Button,
  Drawer,
  Input,
  InputNumber,
  Popconfirm,
  Select,
  Tag,
  Typography,
} from "antd";
import { getLlmConfig, getVlmConfig, testLlmConfig, testVlmConfig, updateLlmConfig, updateVlmConfig, type DataFrequency, type LlmConfigResponse } from "./api";
import { ACCENT_PRESETS } from "./theme";
import { FieldLabel, SectionTitle } from "./ui/Section";

const { Paragraph, Text } = Typography;

// ---------------------------------------------------------------------------
// Persisted settings helpers (shared with main.tsx)
// ---------------------------------------------------------------------------

export const DEFAULTS_STORAGE_KEY = "cta_research_defaults";
export const ENGINE_STORAGE_KEY = "cta_research_engine";
const LLM_STORAGE_KEY = "cta_research_llm";
const VLM_STORAGE_KEY = "cta_research_vlm";

export interface ResearchDefaults {
  frequency: DataFrequency;
  riskFreeRate: number;
}

export function loadResearchDefaults(): ResearchDefaults {
  try {
    const raw = localStorage.getItem(DEFAULTS_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<ResearchDefaults>;
      const frequency =
        parsed.frequency === "daily" || parsed.frequency === "weekly" || parsed.frequency === "monthly"
          ? parsed.frequency
          : "weekly";
      const riskFreeRate =
        typeof parsed.riskFreeRate === "number" && Number.isFinite(parsed.riskFreeRate)
          ? parsed.riskFreeRate
          : 0.015;
      return { frequency, riskFreeRate };
    }
  } catch {
    // Corrupt storage falls through to defaults.
  }
  return { frequency: "weekly", riskFreeRate: 0.015 };
}

// ---------------------------------------------------------------------------
// Analysis engine configuration (persisted; consumed by AnalysisPanel)
// ---------------------------------------------------------------------------

export interface AnalysisEngineConfig {
  /** Market data source for the attribution pipeline. */
  dataSource: "api" | "upload";
  /** Rolling regression window size (in observations). */
  rollingWindow: number;
  /** Number of top futures varieties to report. */
  topNVarieties: number;
}

export function loadEngineConfig(): AnalysisEngineConfig {
  try {
    const raw = localStorage.getItem(ENGINE_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<AnalysisEngineConfig>;
      return {
        dataSource: parsed.dataSource === "upload" ? "upload" : "api",
        rollingWindow:
          typeof parsed.rollingWindow === "number" && Number.isFinite(parsed.rollingWindow)
            ? Math.min(52, Math.max(4, Math.round(parsed.rollingWindow)))
            : 12,
        topNVarieties:
          typeof parsed.topNVarieties === "number" && Number.isFinite(parsed.topNVarieties)
            ? Math.min(15, Math.max(3, Math.round(parsed.topNVarieties)))
            : 8,
      };
    }
  } catch {
    // Corrupt storage falls through to defaults.
  }
  return { dataSource: "api", rollingWindow: 12, topNVarieties: 8 };
}

// ---------------------------------------------------------------------------
// LLM provider presets (all OpenAI-compatible endpoints)
// ---------------------------------------------------------------------------

interface ProviderPreset {
  value: string;
  label: string;
  apiBase: string;
  model: string;
}

const PROVIDER_PRESETS: ProviderPreset[] = [
  { value: "deepseek", label: "DeepSeek", apiBase: "https://api.deepseek.com/v1", model: "deepseek-v4-flash" },
  { value: "doubao", label: "豆包 Doubao", apiBase: "https://ark.cn-beijing.volces.com/api/v3", model: "" },
  {
    value: "qwen",
    label: "通义千问",
    apiBase: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: "qwen-plus",
  },
  { value: "openai", label: "OpenAI", apiBase: "https://api.openai.com/v1", model: "gpt-4o" },
  { value: "custom", label: "自定义（OpenAI 兼容）", apiBase: "", model: "" },
];

interface LlmFormState {
  provider: string;
  apiBase: string;
  apiKey: string;
  model: string;
}

function loadLlmForm(): LlmFormState {
  try {
    const raw = localStorage.getItem(LLM_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<LlmFormState>;
      return {
        provider: parsed.provider ?? "deepseek",
        apiBase: parsed.apiBase ?? "",
        apiKey: parsed.apiKey ?? "",
        model: parsed.model ?? "",
      };
    }
  } catch {
    // Corrupt storage falls through to defaults.
  }
  return { provider: "deepseek", apiBase: PROVIDER_PRESETS[0]!.apiBase, apiKey: "", model: PROVIDER_PRESETS[0]!.model };
}

// ---------------------------------------------------------------------------
// VLM (vision model) presets for chart extraction
// ---------------------------------------------------------------------------

interface VlmModelPreset {
  value: string;
  label: string;
  description: string;
}

const VLM_MODEL_PRESETS: VlmModelPreset[] = [
  { value: "qwen3-vl-flash", label: "Qwen3-VL-Flash（快速）", description: "速度快、成本低，适合批量处理" },
  { value: "qwen-vl-max", label: "Qwen-VL-Max（高精度）", description: "刻度读取最准，成本较高" },
  { value: "qwen2.5-vl-72b-instruct", label: "Qwen2.5-VL-72B", description: "开源最强，需 DashScope 或自部署" },
  { value: "qwen2.5-vl-7b-instruct", label: "Qwen2.5-VL-7B（本地）", description: "轻量本地部署，配合 Ollama/vLLM" },
  { value: "custom", label: "自定义模型", description: "任意 OpenAI 兼容 VLM 端点" },
];

interface VlmFormState {
  provider: string;
  model: string;
  apiKey: string;
  baseUrl: string;
}

function loadVlmForm(): VlmFormState {
  try {
    const saved = JSON.parse(localStorage.getItem(VLM_STORAGE_KEY) ?? "{}") as Partial<VlmFormState>;
    return { provider: saved.provider ?? "dashscope", model: saved.model ?? "qwen3-vl-flash", apiKey: saved.apiKey ?? "", baseUrl: saved.baseUrl ?? "" };
  } catch {
    return { provider: "dashscope", model: "qwen3-vl-flash", apiKey: "", baseUrl: "" };
  }
}

// ---------------------------------------------------------------------------
// Presentational helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Settings drawer
// ---------------------------------------------------------------------------

interface SettingsDrawerProps {
  open: boolean;
  onClose: () => void;
  accentKey: string;
  onAccentChange: (key: string) => void;
  defaults: ResearchDefaults;
  onDefaultsChange: (defaults: ResearchDefaults) => void;
  engine: AnalysisEngineConfig;
  onEngineChange: (engine: AnalysisEngineConfig) => void;
  onClearHistory: () => void;
}

/** Application settings: theme accent, LLM connection, defaults and data. */
export default function SettingsDrawer({
  open,
  onClose,
  accentKey,
  onAccentChange,
  defaults,
  onDefaultsChange,
  engine,
  onEngineChange,
  onClearHistory,
}: SettingsDrawerProps): React.JSX.Element {
  const [llmForm, setLlmForm] = useState<LlmFormState>(loadLlmForm);
  const [serverConfig, setServerConfig] = useState<LlmConfigResponse>();
  const [backendOnline, setBackendOnline] = useState<boolean>();
  const [isSavingLlm, setIsSavingLlm] = useState(false);
  const [isTestingLlm, setIsTestingLlm] = useState(false);
  const [llmMessage, setLlmMessage] = useState<{ kind: "success" | "error"; text: string }>();
  const [historyCleared, setHistoryCleared] = useState(false);

  // VLM state
  const [vlmForm, setVlmForm] = useState<VlmFormState>(loadVlmForm);
  const [vlmApiKeySet, setVlmApiKeySet] = useState(false);
  const [isSavingVlm, setIsSavingVlm] = useState(false);
  const [isTestingVlm, setIsTestingVlm] = useState(false);
  const [vlmMessage, setVlmMessage] = useState<{ kind: "success" | "error"; text: string }>();
  const llmFormComplete = Boolean(llmForm.apiBase.trim() && llmForm.apiKey.trim());

  const refreshStatus = useCallback(async () => {
    setBackendOnline(undefined);
    try {
      let config = await getLlmConfig();
      // Runtime settings disappear when a local backend is restarted. Restore
      // the user-approved browser-local setting and then verify the response.
      if (!config.enabled && llmForm.apiBase.trim() && llmForm.apiKey.trim()) {
        config = await updateLlmConfig({ api_base: llmForm.apiBase.trim(), api_key: llmForm.apiKey.trim(), model: llmForm.model.trim() || "deepseek-v4-flash", enabled: true });
      }
      setServerConfig(config);
      setBackendOnline(true);
    } catch {
      setServerConfig(undefined);
      setBackendOnline(false);
    }
  }, [llmForm]);

  useEffect(() => {
    if (open) void refreshStatus();
  }, [open, refreshStatus]);

  // Load VLM config when drawer opens
  useEffect(() => {
    if (!open) return;
    void (async () => {
      try {
        let cfg = await getVlmConfig();
        const saved = loadVlmForm();
        if (!cfg.api_key_set && saved.apiKey.trim()) {
          await updateVlmConfig({ provider: saved.provider, model: saved.model, api_key: saved.apiKey, base_url: saved.baseUrl || undefined });
          cfg = await getVlmConfig();
        }
        setVlmForm((prev) => ({
          provider: cfg.provider || "dashscope",
          model: cfg.model || "qwen3-vl-flash",
          apiKey: prev.apiKey || saved.apiKey,
          baseUrl: cfg.base_url || "",
        }));
        setVlmApiKeySet(cfg.api_key_set);
      } catch {
        // Backend offline; keep defaults
      }
    })();
  }, [open]);

  async function handleSaveVlm(): Promise<void> {
    setIsSavingVlm(true);
    setVlmMessage(undefined);
    try {
      const payload: { provider?: string; model?: string; api_key?: string; base_url?: string } = {
        provider: vlmForm.provider,
        model: vlmForm.model,
      };
      if (vlmForm.apiKey.trim()) payload.api_key = vlmForm.apiKey.trim();
      if (vlmForm.baseUrl.trim()) payload.base_url = vlmForm.baseUrl.trim();
      await updateVlmConfig(payload);
      try { localStorage.setItem(VLM_STORAGE_KEY, JSON.stringify(vlmForm)); } catch { /* session still applies */ }
      if (vlmForm.apiKey.trim()) setVlmApiKeySet(true);
      setVlmMessage({ kind: "success", text: `VLM 已切换为 ${vlmForm.model}，图表提取将使用新模型。` });
    } catch (error) {
      setVlmMessage({ kind: "error", text: error instanceof Error ? error.message : "保存 VLM 配置失败" });
    } finally {
      setIsSavingVlm(false);
    }
  }

  async function handleTestVlm(): Promise<void> {
    setIsTestingVlm(true);
    setVlmMessage(undefined);
    try {
      const result = await testVlmConfig();
      setVlmMessage({ kind: "success", text: `连通成功：${result.model}，耗时 ${result.latency_ms}ms。${result.detail}` });
    } catch (error) {
      setVlmMessage({ kind: "error", text: error instanceof Error ? error.message : "VLM 连通性测试失败" });
    } finally {
      setIsTestingVlm(false);
    }
  }

  function handleProviderChange(value: string): void {
    const preset = PROVIDER_PRESETS.find((item) => item.value === value);
    setLlmForm((prev) => ({
      ...prev,
      provider: value,
      apiBase: preset && preset.apiBase ? preset.apiBase : prev.apiBase,
      model: preset && preset.model ? preset.model : prev.model,
    }));
  }

  async function handleSaveLlm(): Promise<void> {
    setIsSavingLlm(true);
    setLlmMessage(undefined);
    try {
      let apiBase = llmForm.apiBase.trim().replace(/\/$/, "");
      // DeepSeek's OpenAI-compatible endpoint requires /v1.  Users often
      // paste the homepage URL, which otherwise saves but fails at request time.
      if (llmForm.provider === "deepseek" && apiBase === "https://api.deepseek.com") apiBase += "/v1";
      if (!apiBase || !llmForm.apiKey.trim()) {
        setLlmMessage({ kind: "error", text: "请填写 API Base 与 API Key 后再保存。" });
        return;
      }
      const result = await updateLlmConfig({
        api_base: apiBase,
        api_key: llmForm.apiKey.trim(),
        model: llmForm.model.trim() || "deepseek-v4-flash",
        enabled: true,
      });
      setServerConfig(result);
      setLlmForm((previous) => ({ ...previous, apiBase }));
      setBackendOnline(true);
      try {
        localStorage.setItem(LLM_STORAGE_KEY, JSON.stringify(llmForm));
      } catch {
        // Storage may be unavailable; the server-side config still applies.
      }
      setLlmMessage(
        result.enabled
          ? { kind: "success", text: `已启用 LLM：${result.model}（报告生成将优先使用大模型，失败时回退模板）。` }
          : { kind: "error", text: "本地后端未确认启用。请确认后端在线后点击“刷新”；DeepSeek 请使用 https://api.deepseek.com/v1。" },
      );
    } catch (error) {
      setLlmMessage({ kind: "error", text: error instanceof Error ? error.message : "保存失败，请检查后端是否在线。" });
    } finally {
      setIsSavingLlm(false);
    }
  }

  async function handleTestLlm(): Promise<void> {
    setIsTestingLlm(true);
    setLlmMessage(undefined);
    try {
      const result = await testLlmConfig();
      setLlmMessage({ kind: "success", text: `连通成功：${result.model}，耗时 ${result.latency_ms}ms。${result.detail}` });
    } catch (error) {
      setLlmMessage({ kind: "error", text: error instanceof Error ? error.message : "LLM 连通性测试失败" });
    } finally {
      setIsTestingLlm(false);
    }
  }

  return (
    <Drawer title="设置" placement="right" width={440} open={open} onClose={onClose}>
      {/* Theme accent ----------------------------------------------------- */}
      <SectionTitle first>主题强调色</SectionTitle>
      <div style={{ display: "flex", gap: 10 }}>
        {ACCENT_PRESETS.map((preset) => {
          const selected = preset.key === accentKey;
          return (
            <button
              key={preset.key}
              type="button"
              onClick={() => onAccentChange(preset.key)}
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 6,
                padding: "10px 4px",
                border: selected ? "1.5px solid var(--serif-foreground)" : "1px solid var(--serif-border)",
                borderRadius: 8,
                background: selected ? "var(--serif-muted)" : "var(--serif-card)",
                cursor: "pointer",
                transition: "border-color 0.2s",
              }}
            >
              <span
                style={{
                  width: 26,
                  height: 26,
                  borderRadius: "50%",
                  background: preset.accent,
                  border: "1px solid rgba(0, 0, 0, 0.08)",
                  display: "block",
                }}
              />
              <span style={{ fontSize: 12, color: "var(--serif-foreground)" }}>{preset.label}</span>
            </button>
          );
        })}
      </div>
      <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 10, marginBottom: 0 }}>
        切换后立即生效并记住选择；「明金」为 CubeMind 式亮金按钮风格。
      </Paragraph>

      {/* LLM connection ---------------------------------------------------- */}
      <SectionTitle>大模型接入</SectionTitle>
      <div style={{ marginBottom: 12 }}>
        {backendOnline === undefined ? (
          <Tag>探测中…</Tag>
        ) : backendOnline ? (
          <Tag color="green">后端在线</Tag>
        ) : (
          <Tag color="red">后端离线</Tag>
        )}
        {serverConfig && backendOnline &&
          (serverConfig.enabled ? (
            <Tag color="gold">LLM 已启用 · {serverConfig.model}</Tag>
          ) : (
            <Tag>LLM 未启用（使用模板报告）</Tag>
          ))}
        <Button size="small" onClick={() => void refreshStatus()} style={{ marginLeft: 4 }}>
          刷新
        </Button>
        {backendOnline === false && llmFormComplete && <Tag color="orange">浏览器已保存配置，后端暂不可确认</Tag>}
      </div>

      <FieldLabel>服务商</FieldLabel>
      <Select
        value={llmForm.provider}
        onChange={handleProviderChange}
        style={{ width: "100%", marginBottom: 12 }}
        options={PROVIDER_PRESETS.map((item) => ({ value: item.value, label: item.label }))}
      />
      <FieldLabel>API Base</FieldLabel>
      <Input
        value={llmForm.apiBase}
        onChange={(event) => setLlmForm((prev) => ({ ...prev, apiBase: event.target.value }))}
        placeholder="https://api.deepseek.com/v1"
        style={{ marginBottom: 12 }}
      />
      <FieldLabel>API Key</FieldLabel>
      <Input.Password
        value={llmForm.apiKey}
        onChange={(event) => setLlmForm((prev) => ({ ...prev, apiKey: event.target.value }))}
        placeholder="sk-…"
        style={{ marginBottom: 12 }}
      />
      <FieldLabel>模型名称</FieldLabel>
      <Input
        value={llmForm.model}
        onChange={(event) => setLlmForm((prev) => ({ ...prev, model: event.target.value }))}
          placeholder="deepseek-v4-flash"
        style={{ marginBottom: 12 }}
      />
      <div style={{ display: "flex", gap: 8 }}>
        <Button style={{ flex: 1 }} loading={isSavingLlm} disabled={isTestingLlm} onClick={() => void handleSaveLlm()}>保存并应用</Button>
        <Button style={{ flex: 1 }} loading={isTestingLlm} disabled={isSavingLlm} onClick={() => void handleTestLlm()}>测试连通性</Button>
      </div>
      <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
        API Key 仅保存在本机浏览器 localStorage，并发送给本地后端用于调用；不会上传到其他服务器。
      </Paragraph>
      {llmMessage && (
        <Alert
          type={llmMessage.kind}
          showIcon
          message={llmMessage.text}
          style={{ marginTop: 10 }}
          closable
          onClose={() => setLlmMessage(undefined)}
        />
      )}

      {/* VLM (chart extraction) ---------------------------------------------- */}
      <SectionTitle>视觉模型（图表识别）</SectionTitle>
      <Alert type="info" showIcon style={{ marginBottom: 10 }} message="图表 VLM 与上方文本 LLM 是两套配置" description="DeepSeek 配置用于生成文字分析；图表自动读刻度/结构需要在此配置支持视觉的模型与 API Key（例如 DashScope Qwen-VL）。" />
      <div style={{ marginBottom: 12 }}>
        {vlmApiKeySet ? (
          <Tag color="green">API Key 已配置</Tag>
        ) : (
          <Tag color="orange">未配置 Key（使用 HSV 兜底）</Tag>
        )}
        <Tag>{vlmForm.model}</Tag>
      </div>
      <FieldLabel>识别模型</FieldLabel>
      <Select
        value={VLM_MODEL_PRESETS.some((p) => p.value === vlmForm.model) ? vlmForm.model : "custom"}
        onChange={(value) => {
          if (value !== "custom") setVlmForm((prev) => ({ ...prev, model: value }));
        }}
        style={{ width: "100%", marginBottom: 4 }}
        options={VLM_MODEL_PRESETS.map((p) => ({ value: p.value, label: p.label }))}
      />
      <Paragraph type="secondary" style={{ fontSize: 11, marginBottom: 12 }}>
        {VLM_MODEL_PRESETS.find((p) => p.value === vlmForm.model)?.description ?? "自定义 OpenAI 兼容 VLM 端点"}
      </Paragraph>
      {vlmForm.model === "custom" || vlmForm.model === "qwen2.5-vl-7b-instruct" ? (
        <>
          <FieldLabel>Base URL</FieldLabel>
          <Input
            value={vlmForm.baseUrl}
            onChange={(e) => setVlmForm((prev) => ({ ...prev, baseUrl: e.target.value }))}
            placeholder="http://localhost:11434/v1"
            style={{ marginBottom: 12 }}
          />
        </>
      ) : null}
      <FieldLabel>DashScope API Key</FieldLabel>
      <Input.Password
        value={vlmForm.apiKey}
        onChange={(e) => setVlmForm((prev) => ({ ...prev, apiKey: e.target.value }))}
        placeholder={vlmApiKeySet ? "已配置（留空保持不变）" : "sk-…"}
        style={{ marginBottom: 12 }}
      />
      <div style={{ display: "flex", gap: 8 }}>
        <Button style={{ flex: 1 }} loading={isSavingVlm} disabled={isTestingVlm} onClick={() => void handleSaveVlm()}>保存 VLM 配置</Button>
        <Button style={{ flex: 1 }} loading={isTestingVlm} disabled={isSavingVlm} onClick={() => void handleTestVlm()}>测试图像连通性</Button>
      </div>
      <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
        高精度模型（Max/72B）对刻度数字识别更准确，适合关键报告；Flash 适合批量日常处理。
      </Paragraph>
      {vlmMessage && (
        <Alert
          type={vlmMessage.kind}
          showIcon
          message={vlmMessage.text}
          style={{ marginTop: 10 }}
          closable
          onClose={() => setVlmMessage(undefined)}
        />
      )}

      {/* Analysis engine ---------------------------------------------------- */}
      <SectionTitle>分析引擎</SectionTitle>
      <FieldLabel>行情数据源</FieldLabel>
      <Select<"api" | "upload">
        value={engine.dataSource}
        onChange={(value) => onEngineChange({ ...engine, dataSource: value })}
        style={{ width: "100%", marginBottom: 12 }}
        options={[
          { value: "api", label: "AKShare 实时行情（推荐）" },
          { value: "upload", label: "仅使用上传 / CSV 兜底数据" },
        ]}
      />
      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
        <div style={{ flex: 1 }}>
          <FieldLabel>滚动窗口（期数）</FieldLabel>
          <InputNumber
            value={engine.rollingWindow}
            onChange={(value) => onEngineChange({ ...engine, rollingWindow: value ?? 12 })}
            min={4}
            max={52}
            style={{ width: "100%" }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <FieldLabel>Top-N 品种数</FieldLabel>
          <InputNumber
            value={engine.topNVarieties}
            onChange={(value) => onEngineChange({ ...engine, topNVarieties: value ?? 8 })}
            min={3}
            max={15}
            style={{ width: "100%" }}
          />
        </div>
      </div>
      <Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 0 }}>
        AI 归因分析的运行参数与 LLM 报告引擎统一在此配置，修改后立即生效。
      </Paragraph>

      {/* Defaults ---------------------------------------------------------- */}
      <SectionTitle>默认参数</SectionTitle>
      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <FieldLabel>无风险利率</FieldLabel>
          <InputNumber
            value={defaults.riskFreeRate}
            onChange={(value) => onDefaultsChange({ ...defaults, riskFreeRate: value ?? 0.015 })}
            min={-1}
            max={1}
            step={0.001}
            style={{ width: "100%" }}
          />
        </div>
      </div>
      <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
        修改后立即应用到当前页面，并作为下次打开时的初始值。
      </Paragraph>

      {/* Data management ---------------------------------------------------- */}
      <SectionTitle>数据管理</SectionTitle>
      <Popconfirm
        title="确定清空全部历史记录？"
        description="此操作不可撤销，将删除所有已保存的净值分析记录。"
        onConfirm={() => {
          onClearHistory();
          setHistoryCleared(true);
        }}
        okText="清空"
        cancelText="取消"
        okButtonProps={{ danger: true }}
      >
        <Button danger>清空全部历史记录</Button>
      </Popconfirm>
      {historyCleared && (
        <Alert
          type="success"
          showIcon
          message="历史记录已清空。"
          style={{ marginTop: 10 }}
          closable
          onClose={() => setHistoryCleared(false)}
        />
      )}

      <div style={{ marginTop: 28, paddingTop: 14, borderTop: "1px solid var(--serif-border)" }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          私募 CTA 研究平台 · 设置项保存在本机浏览器中
        </Text>
      </div>
    </Drawer>
  );
}
