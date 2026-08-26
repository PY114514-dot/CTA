"""Product CRUD route handlers for the knowledge-base API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_session
from app.services import product_store as store

from .helpers import _product_payload
from .schemas import (
    AliasCreate,
    MergeRequest,
    ProductConfirm,
    ProductCreate,
    ProductUpdate,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Product endpoints
# ---------------------------------------------------------------------------


@router.post("/products")
def create_product(body: ProductCreate, session: Session = Depends(get_session)):
    product = store.create_product(session, **body.model_dump())
    return {"id": product.id, "standard_name": product.standard_name, "confirmation_status": product.confirmation_status}


@router.get("/products")
def list_products(
    confirmation_status: str | None = None,
    strategy: str | None = None,
    manager_name: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_session),
):
    products = store.list_products(
        session,
        confirmation_status=confirmation_status,
        strategy=strategy,
        manager_name=manager_name,
        search=search,
        limit=limit,
        offset=offset,
    )
    # 列表页一次取全部产品的净值统计与来源信息，避免逐只物化净值点
    #（N+1 懒加载是列表端点的主要耗时来源）。
    product_ids = [product.id for product in products]
    nav_stats = store.get_nav_stats_bulk(session, product_ids)
    nav_source_names, nav_source_file_ids = store.get_nav_source_info_bulk(session, product_ids)
    empty_stats = {
        "nav_count": 0, "reviewed_count": 0,
        "nav_start": None, "nav_end": None,
        "all_have_source": 1, "all_direct": 0,
    }
    return [
        _product_payload(
            product,
            nav_stats=nav_stats.get(product.id, empty_stats),
            nav_source_names=nav_source_names.get(product.id, []),
            nav_source_file_ids=nav_source_file_ids.get(product.id, []),
        )
        for product in products
    ]


@router.get("/products/count")
def count_products(
    search: str | None = None,
    session: Session = Depends(get_session),
) -> dict[str, int]:
    statement = select(func.count()).select_from(store.ProductEntity)
    if search:
        statement = statement.where(store.ProductEntity.standard_name.ilike(f"%{search}%"))
    return {"total": int(session.execute(statement).scalar() or 0)}


@router.get("/products/{product_id}")
def get_product(product_id: str, session: Session = Depends(get_session)):
    product = store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    return _product_payload(product, include_detail=True)


@router.post("/products/{product_id}/confirm")
def confirm_product(product_id: str, body: ProductConfirm, session: Session = Depends(get_session)):
    product = store.confirm_product(session, product_id, body.confirmed_by)
    if product is None:
        raise HTTPException(404, "Product not found")
    return {"id": product.id, "confirmation_status": product.confirmation_status}


@router.post("/products/{product_id}/reject")
def reject_product(product_id: str, session: Session = Depends(get_session)):
    product = store.reject_product(session, product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    return {"id": product.id, "confirmation_status": product.confirmation_status}


@router.patch("/products/{product_id}")
def update_product(product_id: str, body: ProductUpdate, session: Session = Depends(get_session)):
    product = store.update_product(session, product_id, body.model_dump(exclude_unset=True))
    if product is None:
        raise HTTPException(404, "Product not found")
    return _product_payload(product, include_detail=True)


@router.delete("/products/{product_id}")
def delete_product(product_id: str, session: Session = Depends(get_session)):
    deleted = store.delete_product(session, product_id)
    if not deleted:
        raise HTTPException(404, "Product not found")
    return {"id": product_id, "deleted": True}


@router.post("/products/{product_id}/aliases")
def add_alias(product_id: str, body: AliasCreate, session: Session = Depends(get_session)):
    alias = store.add_alias(session, product_id, body.alias, body.alias_type, body.source_file_id)
    return {"id": alias.id, "alias": alias.alias}


@router.post("/products/merge")
def merge_products(body: MergeRequest, session: Session = Depends(get_session)):
    target = store.merge_products(session, body.source_id, body.target_id)
    if target is None:
        raise HTTPException(404, "Source or target product not found")
    return {"id": target.id, "standard_name": target.standard_name, "merged": True}
