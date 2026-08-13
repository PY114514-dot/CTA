"""PaddleOCR cloud document-parsing integration.

This module wraps the asynchronous job API and returns only reviewable
Markdown.  Authentication is always server-side through environment settings.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any

import requests

from app.config import (
    PADDLEOCR_API_TOKEN,
    PADDLEOCR_JOB_URL,
    PADDLEOCR_MODEL,
    PADDLEOCR_MAX_WAIT_SECONDS,
    PADDLEOCR_POLL_INTERVAL_SECONDS,
    PADDLEOCR_TIMEOUT_SECONDS,
)
from app.schemas import PaddleOcrDocumentResponse, PaddleOcrPage


class PaddleOcrServiceError(RuntimeError):
    """A recoverable upstream PaddleOCR configuration or request failure."""


_OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": True,
}


def extract_document(file_bytes: bytes, filename: str) -> PaddleOcrDocumentResponse:
    """Submit a document and wait for its parsed Markdown result."""
    if not PADDLEOCR_API_TOKEN:
        raise PaddleOcrServiceError("未配置 PADDLEOCR_API_TOKEN，无法使用 PaddleOCR 云解析。")
    if not file_bytes:
        raise PaddleOcrServiceError("上传文件为空。")

    headers = {"Authorization": f"bearer {PADDLEOCR_API_TOKEN}"}
    data = {"model": PADDLEOCR_MODEL, "optionalPayload": json.dumps(_OPTIONAL_PAYLOAD)}
    files = {"file": (filename, io.BytesIO(file_bytes))}
    try:
        response = requests.post(
            PADDLEOCR_JOB_URL, headers=headers, data=data, files=files,
            timeout=PADDLEOCR_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        job_id = response.json()["data"]["jobId"]
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        raise PaddleOcrServiceError(f"PaddleOCR 任务提交失败：{_error_message(error)}") from error

    result_url = _wait_for_result(job_id, headers)
    try:
        result_response = requests.get(result_url, timeout=PADDLEOCR_TIMEOUT_SECONDS)
        result_response.raise_for_status()
        return _parse_jsonl(result_response.text)
    except requests.RequestException as error:
        raise PaddleOcrServiceError(f"PaddleOCR 结果下载失败：{_error_message(error)}") from error


def _wait_for_result(job_id: str, headers: dict[str, str]) -> str:
    deadline = time.monotonic() + PADDLEOCR_MAX_WAIT_SECONDS
    while True:
        try:
            response = requests.get(
                f"{PADDLEOCR_JOB_URL}/{job_id}", headers=headers,
                timeout=PADDLEOCR_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()["data"]
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            raise PaddleOcrServiceError(f"PaddleOCR 任务状态查询失败：{_error_message(error)}") from error

        state = data.get("state")
        if state == "done":
            try:
                return str(data["resultUrl"]["jsonUrl"])
            except (KeyError, TypeError) as error:
                raise PaddleOcrServiceError("PaddleOCR 未返回结果文件地址。") from error
        if state == "failed":
            raise PaddleOcrServiceError(f"PaddleOCR 解析失败：{data.get('errorMsg', '未知原因')}")
        if state not in {"pending", "running"}:
            raise PaddleOcrServiceError(f"PaddleOCR 返回未知任务状态：{state!r}")
        if time.monotonic() >= deadline:
            raise PaddleOcrServiceError("PaddleOCR 任务等待超时，请稍后重试。")
        time.sleep(PADDLEOCR_POLL_INTERVAL_SECONDS)


def _parse_jsonl(payload: str) -> PaddleOcrDocumentResponse:
    pages: list[PaddleOcrPage] = []
    warnings: list[str] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            layouts = json.loads(line)["result"]["layoutParsingResults"]
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"第 {line_number} 条 OCR 结果格式无法识别：{_error_message(error)}")
            continue
        for layout in layouts:
            markdown = layout.get("markdown", {}).get("text", "")
            if markdown:
                pages.append(PaddleOcrPage(page_number=len(pages) + 1, markdown=markdown))
    if not pages:
        warnings.append("未从 PaddleOCR 结果中提取到 Markdown；请检查文档质量或模型输出。")
    return PaddleOcrDocumentResponse(pages=pages, model=PADDLEOCR_MODEL, warnings=warnings)


def _error_message(error: Exception) -> str:
    """Avoid returning a verbose response that could include sensitive headers."""
    return str(error)[:300]
