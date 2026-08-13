"""Evidence retrieval service for the FOF Agent.

Provides keyword + metadata hybrid search across products, document fragments
and NAV observations. Every result carries a source citation (file, page,
fragment) so the Agent's answers can be traced back to original materials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import (
    ConfirmationStatus,
    DocumentFragment,
    NavObservation,
    ProductEntity,
    RawFile,
    StructuredFact,
)


@dataclass
class Citation:
    """A traceable reference to a source location."""

    file_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    fragment_id: str | None = None
    fragment_type: str | None = None
    snippet: str | None = None
    confidence: float | None = None


@dataclass
class RetrievalResult:
    """One piece of evidence returned by the retrieval service."""

    category: str  # product | fragment | nav | fact
    title: str
    content: str
    product_id: str | None = None
    product_name: str | None = None
    citation: Citation | None = None
    score: float = 1.0


@dataclass
class RetrievalResponse:
    """Aggregated retrieval results with metadata."""

    query: str
    results: list[RetrievalResult] = field(default_factory=list)
    total: int = 0
    product_ids_searched: list[str] = field(default_factory=list)


def search_products(
    session: Session,
    query: str,
    *,
    product_ids: list[str] | None = None,
    confirmed_only: bool = False,
    limit: int = 10,
) -> list[RetrievalResult]:
    """Search products by name, manager, strategy or alias keywords."""
    stmt = select(ProductEntity)
    if product_ids:
        stmt = stmt.where(ProductEntity.id.in_(product_ids))
    if confirmed_only:
        stmt = stmt.where(ProductEntity.confirmation_status == ConfirmationStatus.CONFIRMED)

    products = list(session.execute(stmt).scalars().all())

    # Score by keyword relevance.
    keywords = _tokenize(query)
    results: list[RetrievalResult] = []
    for product in products:
        score = _score_product(product, keywords)
        if score > 0 or not keywords:
            results.append(RetrievalResult(
                category="product",
                title=product.standard_name,
                content=_product_summary(product),
                product_id=product.id,
                product_name=product.standard_name,
                score=score,
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def search_fragments(
    session: Session,
    query: str,
    *,
    product_ids: list[str] | None = None,
    file_ids: list[str] | None = None,
    limit: int = 10,
) -> list[RetrievalResult]:
    """Full-text keyword search across document fragments."""
    stmt = select(DocumentFragment).where(DocumentFragment.content_text.isnot(None))
    if product_ids:
        stmt = stmt.where(DocumentFragment.product_id.in_(product_ids))
    if file_ids:
        stmt = stmt.where(DocumentFragment.file_id.in_(file_ids))

    fragments = list(session.execute(stmt).scalars().all())

    keywords = _tokenize(query)
    results: list[RetrievalResult] = []
    for fragment in fragments:
        text = fragment.content_text or ""
        score = _score_text(text, keywords)
        if score > 0:
            # Resolve filename for citation display.
            source_file = session.get(RawFile, fragment.file_id) if fragment.file_id else None
            results.append(RetrievalResult(
                category="fragment",
                title=f"{source_file.filename if source_file else '未知文件'} · 第 {fragment.page_number or '?'} 页",
                content=text[:300],
                product_id=fragment.product_id,
                citation=Citation(
                    file_id=fragment.file_id,
                    filename=source_file.filename if source_file else None,
                    page_number=fragment.page_number,
                    fragment_id=fragment.id,
                    fragment_type=fragment.fragment_type,
                    snippet=text[:150],
                    confidence=fragment.ocr_confidence,
                ),
                score=score,
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def search_facts(
    session: Session,
    query: str,
    *,
    product_ids: list[str] | None = None,
    limit: int = 10,
) -> list[RetrievalResult]:
    """Search structured facts by field name or value keywords."""
    stmt = select(StructuredFact).where(StructuredFact.superseded_by.is_(None))
    if product_ids:
        stmt = stmt.where(StructuredFact.product_id.in_(product_ids))

    facts = list(session.execute(stmt).scalars().all())

    keywords = _tokenize(query)
    results: list[RetrievalResult] = []
    for fact in facts:
        searchable = f"{fact.field_name} {fact.field_value}"
        score = _score_text(searchable, keywords)
        if score > 0 or not keywords:
            source_file = session.get(RawFile, fact.source_file_id) if fact.source_file_id else None
            product = session.get(ProductEntity, fact.product_id) if fact.product_id else None
            results.append(RetrievalResult(
                category="fact",
                title=f"{product.standard_name if product else '?'} · {fact.field_name}",
                content=fact.field_value,
                product_id=fact.product_id,
                product_name=product.standard_name if product else None,
                citation=Citation(
                    file_id=fact.source_file_id,
                    filename=source_file.filename if source_file else None,
                    fragment_id=fact.source_fragment_id,
                    confidence=fact.confidence,
                ),
                score=score,
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def retrieve(
    session: Session,
    query: str,
    *,
    product_ids: list[str] | None = None,
    limit: int = 15,
) -> RetrievalResponse:
    """Unified retrieval: searches products, fragments and facts, merges by score."""
    all_results: list[RetrievalResult] = []
    all_results.extend(search_products(session, query, product_ids=product_ids, limit=limit))
    all_results.extend(search_fragments(session, query, product_ids=product_ids, limit=limit))
    all_results.extend(search_facts(session, query, product_ids=product_ids, limit=limit))

    all_results.sort(key=lambda r: r.score, reverse=True)
    trimmed = all_results[:limit]

    return RetrievalResponse(
        query=query,
        results=trimmed,
        total=len(trimmed),
        product_ids_searched=product_ids or [],
    )


def get_product_evidence(session: Session, product_id: str) -> dict[str, Any]:
    """Gather all evidence for a single product: facts, fragments, NAV summary."""
    product = session.get(ProductEntity, product_id)
    if product is None:
        return {"error": "Product not found"}

    facts = list(session.execute(
        select(StructuredFact)
        .where(StructuredFact.product_id == product_id, StructuredFact.superseded_by.is_(None))
    ).scalars().all())

    fragments = list(session.execute(
        select(DocumentFragment)
        .where(DocumentFragment.product_id == product_id)
        .order_by(DocumentFragment.created_at.desc())
        .limit(20)
    ).scalars().all())

    nav_count = session.execute(
        select(NavObservation.id).where(NavObservation.product_id == product_id)
    ).all()

    return {
        "product": {
            "id": product.id,
            "standard_name": product.standard_name,
            "manager_name": product.manager_name,
            "strategy": product.strategy,
            "status": product.status,
            "confirmation_status": product.confirmation_status,
        },
        "facts": [
            {"field_name": f.field_name, "field_value": f.field_value, "confidence": f.confidence}
            for f in facts
        ],
        "fragments": [
            {
                "id": frag.id,
                "type": frag.fragment_type,
                "page": frag.page_number,
                "text_preview": (frag.content_text or "")[:100],
                "confidence": frag.ocr_confidence,
            }
            for frag in fragments
        ],
        "nav_count": len(nav_count),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split query into lowercase keywords (CJK-aware: each char is a token)."""
    tokens: list[str] = []
    for word in text.lower().split():
        # Keep CJK characters as individual tokens for substring matching.
        tokens.append(word)
    return [t for t in tokens if len(t) >= 1]


def _score_text(text: str, keywords: list[str]) -> float:
    """Simple keyword frequency scoring."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


def _score_product(product: ProductEntity, keywords: list[str]) -> float:
    """Score a product by keyword matches across name, manager and strategy."""
    if not keywords:
        return 1.0
    searchable = " ".join(filter(None, [
        product.standard_name,
        product.manager_name,
        product.strategy,
        product.notes,
    ])).lower()
    return _score_text(searchable, keywords)


def _product_summary(product: ProductEntity) -> str:
    parts = [product.standard_name]
    if product.manager_name:
        parts.append(f"管理人: {product.manager_name}")
    if product.strategy:
        parts.append(f"策略: {product.strategy}")
    if product.inception_date:
        parts.append(f"成立: {product.inception_date.isoformat()}")
    parts.append(f"状态: {product.confirmation_status}")
    return " | ".join(parts)
