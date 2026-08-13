"""VLM-based chart STRUCTURE extraction.

The VLM's role is to UNDERSTAND the chart, not to measure values:
- Identify curves (name + color)
- Read axis tick labels (X dates, Y values)
- Locate the plot area boundary

Numerical extraction is done deterministically by pixel_tracer.py.
VLM is OPTIONAL — users can provide curve colors manually via frontend.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any

import httpx

from .models import ChartStructure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt: structure only, no value reading
# ---------------------------------------------------------------------------

STRUCTURE_PROMPT = """\
这是一张单产品私募基金/资管产品的净值曲线图。你的首要任务是为后续像素追踪精确定位，
而不是估计曲线上的每一个点。请分析图表结构（不需要读取曲线上的数值）。

请识别以下信息并严格按 JSON 输出：
{
  "curves": [
    {"name": "图例文字", "color_hex": "#FF0000", "color_name": "red", "is_benchmark": false}
  ],
  "x_ticks": [
    {"label": "2024-01", "position": "left"}
  ],
  "date_range": ["2024-01-01", "2024-12-31"],
  "y_ticks": [
    {"label": "1.00", "position": "bottom"}
  ],
  "plot_area": {
    "has_visible_border": true,
    "has_grid_lines": true,
    "bbox_1000": [120, 180, 900, 780]
  },
  "y_axis_label": "单位净值",
  "y_range": [0.8, 1.5],
  "chart_title": "图表标题(如有)",
  "frequency": "weekly"
}

要求：
1. curves: 列出所有曲线。第一条必须是产品净值线（不是基准线）。color_hex 请仔细观察图框内曲线中段的实际像素颜色（不是图例色块），给出最接近的十六进制值。color_name 用英文描述（如 dark_red, navy_blue, forest_green, orange, gray）
2. x_ticks: 按从左到右顺序列出 X 轴所有可见刻度标签原文
3. y_ticks: 按从下到上顺序列出 Y 轴所有可见刻度标签原文（保留小数位）
4. y_range: Y 轴的最小值和最大值 [min, max]
5. is_benchmark: 是否为基准/对比线（通常为灰色虚线或浅色细线）
6. frequency: 根据 X 轴刻度间距判断 daily/weekly/monthly/quarterly
7. date_range 为横轴最左和最右日期；优先直接读取首尾标签，无法可靠判断时输出 []，不要猜测
8. bbox_1000 是实际绘制曲线的坐标框，不包含标题、图例、坐标轴文字；按图像宽高各归一化到 0–1000，格式为 [left, top, right, bottom]。这是后续 CV 唯一允许追踪的区域，无法可靠判断时输出 []，不要猜测
9. 只输出 JSON，不要添加解释文字
"""


# ---------------------------------------------------------------------------
# Prompt: focused Y-axis number reading (zoomed crop)
# ---------------------------------------------------------------------------

Y_AXIS_PROMPT = """\
这是一张图表的 Y 轴刻度区域（已放大）。请读取所有可见的刻度数值。

要求：
1. 按从下到上的顺序读取每个刻度的数字
2. 忽略 % 符号、逗号等，只输出数字本身（如 "10%" 输出 10，"1,000" 输出 1000）
3. 如果有负数请保留负号
4. 严格按 JSON 数组输出，如 [0, 5, 10, 15, 20]
5. 只输出 JSON 数组，不要任何解释
"""


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class VLMProvider:
    """Base class for VLM providers."""

    async def extract_structure(self, image_bytes: bytes, prompt: str = STRUCTURE_PROMPT) -> str:
        raise NotImplementedError


class DashScopeProvider(VLMProvider):
    """Alibaba DashScope API (Qwen-VL series) via OpenAI-compatible endpoint."""

    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str = "qwen3-vl-flash"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model

    async def extract_structure(self, image_bytes: bytes, prompt: str = STRUCTURE_PROMPT) -> str:
        if not self.api_key:
            raise ValueError(
                "DashScope API key not configured. "
                "Set DASHSCOPE_API_KEY environment variable."
            )

        b64_image = base64.b64encode(image_bytes).decode("ascii")
        mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]


class OpenAICompatibleProvider(VLMProvider):
    """Any OpenAI-compatible VLM endpoint (SiliconFlow, local Ollama, etc.)."""

    def __init__(self, base_url: str, api_key: str = "not-needed", model: str = "qwen2.5-vl-7b"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def extract_structure(self, image_bytes: bytes, prompt: str = STRUCTURE_PROMPT) -> str:
        b64_image = base64.b64encode(image_bytes).decode("ascii")
        mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_structure_response(raw_text: str) -> ChartStructure:
    """Parse VLM structure response into ChartStructure.

    Handles markdown fences, partial JSON, etc.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse VLM structure response")
                raise ValueError("VLM response did not contain valid JSON") from None
        else:
            raise ValueError("VLM response did not contain a JSON object")

    if not isinstance(data, dict):
        raise ValueError("VLM response JSON must be an object")

    # Build structured result
    curves = []
    for c in data.get("curves", []):
        curves.append({
            "name": c.get("name", ""),
            "color_hex": c.get("color_hex", ""),
            "color_name": c.get("color_name", ""),
            "is_benchmark": bool(c.get("is_benchmark", False)),
        })

    result = ChartStructure(
        curves=curves,
        x_ticks=_parse_x_ticks(data),
        y_ticks=[str(t) if isinstance(t, str) else t.get("label", "") for t in data.get("y_ticks", [])],
        y_range=[float(v) for v in data.get("y_range", []) if _is_number(v)][:2],
        y_axis_label=data.get("y_axis_label", ""),
        chart_title=data.get("chart_title", ""),
        frequency=data.get("frequency", "unknown"),
        has_grid_lines=bool(data.get("plot_area", {}).get("has_grid_lines", False)),
        plot_bbox_1000=_parse_normalized_bbox(data.get("plot_area", {}).get("bbox_1000", [])),
        raw_response=raw_text,
    )
    if not (result.curves or result.x_ticks or result.y_ticks or result.y_range or result.plot_bbox_1000):
        raise ValueError("VLM response did not contain usable chart structure")
    return result


def _parse_x_ticks(data: dict[str, Any]) -> list[str]:
    """Return VLM date labels, preserving an explicit first/last range.

    Some models reliably read only the two endpoint dates.  Those two values
    are sufficient for the linear X calibration and are preferable to an
    empty result or hallucinated intermediate ticks.
    """
    ticks = [str(t) if isinstance(t, str) else str(t.get("label", "")) for t in data.get("x_ticks", [])]
    ticks = [tick for tick in ticks if tick]
    date_range = data.get("date_range", [])
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        endpoints = [str(value).strip() for value in date_range]
        if all(endpoints) and (len(ticks) < 2 or ticks[0] != endpoints[0] or ticks[-1] != endpoints[1]):
            return endpoints
    return ticks


def _parse_normalized_bbox(value: Any) -> list[int]:
    """Validate a VLM plot rectangle before it is allowed to affect CV."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        left, top, right, bottom = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return []
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        return []
    # Tiny boxes and near-full-page boxes are usually a title/logo or a
    # hallucinated page boundary, not a usable plotting area.
    area_ratio = ((right - left) * (bottom - top)) / 1_000_000
    if not 0.03 <= area_ratio <= 0.90:
        return []
    return [left, top, right, bottom]


def parse_y_axis_response(raw_text: str) -> list[float]:
    """Parse VLM Y-axis response into a list of numbers (bottom→top).

    Handles markdown fences, partial JSON, and stray text.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct JSON array parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [float(x) for x in data if _is_number(x)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find a JSON array in the text
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [float(x) for x in data if _is_number(x)]
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: extract all numbers via regex
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(n) for n in numbers]


def _is_number(x: Any) -> bool:
    """Check if a value is numeric (int/float/numeric string)."""
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(provider_type: str = "dashscope", **kwargs: Any) -> VLMProvider:
    """Create a VLM provider instance."""
    if provider_type == "dashscope":
        return DashScopeProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "qwen3-vl-flash"),
        )
    elif provider_type == "openai_compatible":
        return OpenAICompatibleProvider(
            base_url=kwargs.get("base_url", "http://localhost:11434/v1"),
            api_key=kwargs.get("api_key", "not-needed"),
            model=kwargs.get("model", "qwen2.5-vl-7b"),
        )
    else:
        raise ValueError(f"Unknown VLM provider: {provider_type}")
