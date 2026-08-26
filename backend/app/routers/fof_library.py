"""P0 API for the versioned FOF product-knowledge foundation."""

from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.fof_library import FofLibraryStore
from app.services.fof_material_research import analyze_material
from app.dependencies import get_report_config

router = APIRouter(tags=["FOF 产品库"])


@lru_cache(maxsize=1)
def _get_store() -> FofLibraryStore:
    """Lazily construct the library store (avoids an import-time side effect
    that would create directories and open the database during module import)."""
    return FofLibraryStore()


def get_fof_library_store() -> FofLibraryStore:
    return _get_store()


class ProductCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    manager_name: str | None = Field(default=None, max_length=120)
    strategy: str | None = Field(default=None, max_length=80)


class MaterialLinkRequest(BaseModel):
    material_id: str


class ProductReviewRequest(BaseModel):
    verification_status: Literal["pending", "confirmed", "rejected"]
    research_status: Literal["needs_review", "researchable", "unusable"]
    reason: str | None = Field(default=None, max_length=500)


class FofChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


@router.post("/api/fof-library/products")
def create_product(request: ProductCreateRequest, store: FofLibraryStore = Depends(get_fof_library_store)) -> dict:
    return store.create_product(request.name, request.manager_name, request.strategy)


@router.get("/api/fof-library/products")
def list_products(store: FofLibraryStore = Depends(get_fof_library_store)) -> list[dict]:
    return store.list_products()


@router.patch("/api/fof-library/products/{product_id}/review")
def review_product(
    product_id: str, request: ProductReviewRequest, store: FofLibraryStore = Depends(get_fof_library_store),
) -> dict:
    try:
        return store.review_product(
            product_id, request.verification_status, request.research_status, request.reason,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.post("/api/fof-library/materials")
async def ingest_materials(
    files: list[UploadFile] = File(...),
    source_label: str | None = Form(default=None),
    report_date: str | None = Form(default=None),
    parsing_method: str = Form(default="manual_upload"),
    store: FofLibraryStore = Depends(get_fof_library_store),
) -> dict[str, list[dict]]:
    """Register immutable source materials; parsing is a separate future task."""
    accepted_types = {"application/pdf", "image/jpeg", "image/png", "image/webp", "text/csv",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    results: list[dict] = []
    for upload in files:
        if upload.content_type not in accepted_types:
            raise HTTPException(415, f"不支持的文件类型：{upload.content_type or '未知'}")
        results.append(store.ingest_material(
            await upload.read(), upload.filename or "material.bin", upload.content_type,
            source_label, report_date, parsing_method,
        ))
    return {"materials": results}


@router.post("/api/fof-library/products/{product_id}/materials")
def link_material(
    product_id: str, request: MaterialLinkRequest, store: FofLibraryStore = Depends(get_fof_library_store),
) -> dict[str, Literal["ok"]]:
    store.link_material(product_id, request.material_id)
    return {"status": "ok"}


@router.get("/api/fof-library/materials")
def list_materials(store: FofLibraryStore = Depends(get_fof_library_store)) -> list[dict]:
    return store.list_materials()


@router.post("/api/fof-library/materials/{material_id}/analyze")
def analyze_single_material(material_id: str, store: FofLibraryStore = Depends(get_fof_library_store)) -> dict:
    """Create evidence-backed FOF Agent research output from one uploaded image."""
    try:
        return analyze_material(store, material_id)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.post("/api/fof-library/chat")
def fof_library_chat(request: FofChatRequest, store: FofLibraryStore = Depends(get_fof_library_store)) -> dict:
    """FOF workbench conversation grounded in its own product/evidence store."""
    query = request.query.strip()
    config = get_report_config()
    if any(token in query.lower() for token in ("什么模型", "哪个模型", "模型", "model")):
        content = (
            f"当前 FOF Agent 的默认文本解读模型是 **{config.model or 'deepseek-v4-flash'}**。\n\n"
            "图片 OCR、净值计算、候选入库和证据校验由本地确定性工具完成；模型只用于基于已审计证据的文字解释，不会替代计算或复核。"
        )
        return {"content": content, "citations": [], "tool_calls": [{"name": "inspect_agent_configuration", "status": "ok", "duration_ms": 0, "summary": f"默认模型：{config.model or 'deepseek-v4-flash'}"}], "products_referenced": [], "data_context": [], "method_provenance": {"summary": {"method": "本地配置读取", "model": config.model or "deepseek-v4-flash"}}}

    products = store.list_researchable_products()
    if not products:
        return {"content": "尚无可研究产品。请先关联原始材料、复核净值并确认产品身份。", "citations": [], "tool_calls": [{"name": "search_fof_library", "status": "ok", "duration_ms": 0, "summary": "可研究产品 0 个"}], "products_referenced": [], "data_context": [], "method_provenance": {"summary": {"method": "本地证据检索"}}}
    lines = ["以下为已完成复核、可用于研究的产品："]
    citations: list[dict] = []
    for product in products[:8]:
        evidence = store.list_product_evidence(product["product_id"])
        facts = "；".join(f"{item['claim_type']} {item['claim_value']}" for item in evidence[:8]) or "未提取到指标"
        lines.append(f"- **{product['name']}**（{product['manager_name'] or '管理人待确认'}）：{facts}。")
        citations.extend({"file_id": item["material_id"], "filename": None, "page_number": None,
                          "fragment_id": item["evidence_id"], "fragment_type": item["claim_type"],
                          "snippet": f"{item['claim_type']} {item['claim_value']}", "confidence": item["confidence"]}
                         for item in evidence[:8])
    lines.extend(["", "以上仅为研究用途，不构成投资建议；仍需人工尽调与投委会复核。"])
    return {"content": "\n".join(lines), "citations": citations, "tool_calls": [{"name": "search_fof_library", "status": "ok", "duration_ms": 0, "summary": f"检索到 {len(products)} 个可研究产品"}], "products_referenced": [{"id": p["product_id"], "name": p["name"], "category": "fof_candidate"} for p in products], "data_context": [{"product_id": p["product_id"], "name": p["name"], "nav_count": p["nav_count"], "frequency": None, "start_date": None, "end_date": None, "review_status": "可研究", "source_files": [], "fact_count": p["evidence_count"]} for p in products], "method_provenance": {"summary": {"method": "FOF 产品库证据检索", "model": None}}}
