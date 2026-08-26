"""Product knowledge-base service: CRUD, merge, confirmation and NAV management.

This module is the primary data-access layer for the FOF Agent's product and
material domain. It enforces the master design's traceability and confirmation
rules:
- Every NAV point links to a source file/fragment.
- Products require explicit human confirmation before entering recommendations.
- Merging two products preserves aliases and re-links observations.
- Data snapshots freeze inputs for reproducible recommendations.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

from sqlalchemy import case, delete, exists, func, or_, select
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session, selectinload

from app.models import (
    AgentRun,
    ConfirmationStatus,
    DataSnapshot,
    DecisionRecord,
    DecisionStatus,
    DocumentFragment,
    NavCandidateVersion,
    NavObservation,
    ProductAlias,
    ProductEntity,
    ProductStatus,
    RawFile,
    ReviewStatus,
    StructuredFact,
    ToolInvocation,
)


# ---------------------------------------------------------------------------
# Material / File operations
# ---------------------------------------------------------------------------


def register_file(
    session: Session,
    *,
    filename: str,
    file_hash: str,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    source: str | None = None,
    report_period: str | None = None,
    storage_path: str | None = None,
) -> RawFile:
    """Register an uploaded file. Detects duplicates by hash and bumps version."""
    existing = session.execute(
        select(RawFile).where(RawFile.file_hash == file_hash)
    ).scalars().first()
    if existing is not None:
        # Same content already registered; create a new version entry.
        max_version = session.execute(
            select(RawFile.version)
            .where(RawFile.filename == filename)
            .order_by(RawFile.version.desc())
            .limit(1)
        ).scalar() or 0
        version = max_version + 1
    else:
        version = 1

    record = RawFile(
        filename=filename,
        file_hash=file_hash,
        mime_type=mime_type,
        size_bytes=size_bytes,
        source=source,
        report_period=report_period,
        version=version,
        storage_path=storage_path,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def compute_file_hash(content: bytes) -> str:
    """SHA-256 hex digest for deduplication."""
    return hashlib.sha256(content).hexdigest()


def list_files(
    session: Session,
    *,
    parsing_status: str | None = None,
    pending_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> list[RawFile]:
    stmt = select(RawFile).options(
        selectinload(RawFile.fragments).selectinload(DocumentFragment.product),
        selectinload(RawFile.nav_observations).selectinload(NavObservation.product),
    ).order_by(RawFile.uploaded_at.desc())
    if parsing_status:
        stmt = stmt.where(RawFile.parsing_status == parsing_status)
    if pending_only:
        unresolved_fragment = exists(
            select(DocumentFragment.id).where(
                DocumentFragment.file_id == RawFile.id,
                or_(
                    DocumentFragment.product_id.is_(None),
                    DocumentFragment.content_data["binding_status"].as_string() == "unmatched",
                ),
            )
        )
        unreviewed_nav = exists(
            select(NavObservation.id).where(
                NavObservation.source_file_id == RawFile.id,
                NavObservation.review_status != ReviewStatus.REVIEWED,
            )
        )
        stmt = stmt.where(
            or_(RawFile.source.is_(None), RawFile.source != "futures_weekly_sqlite"),
            or_(
                RawFile.parsing_status.in_(("pending", "processing", "failed", "completed_no_nav")),
                unresolved_fragment,
                unreviewed_nav,
            ),
        )
    return list(session.execute(stmt.offset(offset).limit(limit)).scalars().all())


def update_file_status(
    session: Session, file_id: str, status: str, error: str | None = None
) -> RawFile | None:
    record = session.get(RawFile, file_id)
    if record is None:
        return None
    record.parsing_status = status
    record.parsing_error = error
    session.commit()
    session.refresh(record)
    return record


def update_file_extraction_audit(
    session: Session, file_id: str, audit: dict[str, Any]
) -> RawFile | None:
    """Persist the actual extraction path used for one material.

    This is deliberately separate from the human-readable parsing status.  A
    file can be parsed successfully without yielding a NAV series, and a VLM
    can be configured without ever being called.
    """
    record = session.get(RawFile, file_id)
    if record is None:
        return None
    record.extraction_audit = audit
    session.commit()
    session.refresh(record)
    return record


# ---------------------------------------------------------------------------
# Document Fragment operations
# ---------------------------------------------------------------------------


def add_fragment(
    session: Session,
    *,
    file_id: str,
    fragment_type: str,
    page_number: int | None = None,
    bbox: dict[str, Any] | None = None,
    content_text: str | None = None,
    content_data: dict[str, Any] | None = None,
    ocr_confidence: float | None = None,
    product_id: str | None = None,
) -> DocumentFragment:
    fragment = DocumentFragment(
        file_id=file_id,
        fragment_type=fragment_type,
        page_number=page_number,
        bbox=bbox,
        content_text=content_text,
        content_data=content_data,
        ocr_confidence=ocr_confidence,
        product_id=product_id,
    )
    session.add(fragment)
    session.commit()
    session.refresh(fragment)
    return fragment


def get_fragments_for_file(session: Session, file_id: str) -> list[DocumentFragment]:
    return list(
        session.execute(
            select(DocumentFragment)
            .where(DocumentFragment.file_id == file_id)
            .order_by(DocumentFragment.page_number, DocumentFragment.created_at)
        ).scalars().all()
    )


# ---------------------------------------------------------------------------
# Product Entity operations
# ---------------------------------------------------------------------------


def create_product(
    session: Session,
    *,
    standard_name: str,
    manager_name: str | None = None,
    strategy: str | None = None,
    inception_date: date | None = None,
    nav_frequency: str | None = None,
    status: str = ProductStatus.UNKNOWN,
    notes: str | None = None,
) -> ProductEntity:
    product = ProductEntity(
        standard_name=standard_name,
        manager_name=manager_name,
        strategy=strategy,
        inception_date=inception_date,
        nav_frequency=nav_frequency,
        status=status,
        notes=notes,
    )
    session.add(product)
    session.commit()
    session.refresh(product)
    return product


def get_product(session: Session, product_id: str) -> ProductEntity | None:
    return session.execute(
        select(ProductEntity)
        .options(
            selectinload(ProductEntity.aliases),
            selectinload(ProductEntity.nav_observations),
            selectinload(ProductEntity.facts),
        )
        .where(ProductEntity.id == product_id)
    ).scalars().first()


def list_products(
    session: Session,
    *,
    confirmation_status: str | None = None,
    strategy: str | None = None,
    manager_name: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ProductEntity]:
    stmt = (
        select(ProductEntity)
        .options(
            selectinload(ProductEntity.fragments).selectinload(DocumentFragment.file),
            selectinload(ProductEntity.facts),
        )
        .order_by(ProductEntity.created_at.desc())
    )
    if confirmation_status:
        stmt = stmt.where(ProductEntity.confirmation_status == confirmation_status)
    if strategy:
        stmt = stmt.where(ProductEntity.strategy == strategy)
    if manager_name:
        stmt = stmt.where(ProductEntity.manager_name.ilike(f"%{manager_name}%"))
    if search:
        stmt = stmt.where(ProductEntity.standard_name.ilike(f"%{search}%"))
    return list(session.execute(stmt.offset(offset).limit(limit)).scalars().all())


def confirm_product(
    session: Session, product_id: str, confirmed_by: str = "user"
) -> ProductEntity | None:
    product = session.get(ProductEntity, product_id)
    if product is None:
        return None
    product.confirmation_status = ConfirmationStatus.CONFIRMED
    product.confirmed_by = confirmed_by
    product.confirmed_at = datetime.now()
    session.commit()
    session.refresh(product)
    return product


def reject_product(session: Session, product_id: str) -> ProductEntity | None:
    product = session.get(ProductEntity, product_id)
    if product is None:
        return None
    product.confirmation_status = ConfirmationStatus.REJECTED
    session.commit()
    session.refresh(product)
    return product


def add_alias(
    session: Session,
    product_id: str,
    alias: str,
    alias_type: str = "name",
    source_file_id: str | None = None,
) -> ProductAlias:
    record = ProductAlias(
        product_id=product_id,
        alias=alias,
        alias_type=alias_type,
        source_file_id=source_file_id,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def merge_products(
    session: Session, source_id: str, target_id: str
) -> ProductEntity | None:
    """Merge source product into target. Re-links NAV, facts, fragments and aliases."""
    source = session.get(ProductEntity, source_id)
    target = session.get(ProductEntity, target_id)
    if source is None or target is None:
        return None
    if source_id == target_id:
        return target

    # Re-link child records.
    # Preserve the target's existing observation when both identities contain
    # the same date.  A duplicate date is normally the same report uploaded
    # through two filenames; retaining both would silently distort returns.
    for nav in session.execute(
        select(NavObservation).where(NavObservation.product_id == source_id)
    ).scalars().all():
        duplicate = session.execute(
            select(NavObservation.id).where(
                NavObservation.product_id == target_id,
                NavObservation.observation_date == nav.observation_date,
            ).limit(1)
        ).scalar_one_or_none()
        if duplicate is None:
            nav.product_id = target_id
        else:
            session.delete(nav)

    for fact in session.execute(
        select(StructuredFact).where(StructuredFact.product_id == source_id)
    ).scalars().all():
        fact.product_id = target_id

    for frag in session.execute(
        select(DocumentFragment).where(DocumentFragment.product_id == source_id)
    ).scalars().all():
        frag.product_id = target_id

    for alias in session.execute(
        select(ProductAlias).where(ProductAlias.product_id == source_id)
    ).scalars().all():
        alias.product_id = target_id

    # Add source standard_name as alias of target.
    session.add(ProductAlias(
        product_id=target_id,
        alias=source.standard_name,
        alias_type="merged_name",
    ))

    # Mark source as merged.
    source.confirmation_status = ConfirmationStatus.CONFIRMED
    source.merged_into_id = target_id
    source.status = ProductStatus.CLOSED

    session.commit()
    session.refresh(target)
    return target


def update_product(
    session: Session, product_id: str, fields: dict[str, Any]
) -> ProductEntity | None:
    """Partial update of editable product fields. Unknown keys are ignored."""
    product = session.get(ProductEntity, product_id)
    if product is None:
        return None
    editable = {
        "standard_name", "manager_name", "strategy", "inception_date",
        "close_date", "nav_frequency", "status", "notes", "strategy_disclosure",
    }
    for key, value in fields.items():
        if key in editable:
            setattr(product, key, value)
    session.commit()
    session.refresh(product)
    return product


def delete_product(session: Session, product_id: str) -> bool:
    """Delete a product and its owned data. Returns False if not found.

    NAV observations, structured facts and aliases are product-owned and
    removed. Document fragments are file-derived evidence, so they are kept
    but unlinked (product_id set to NULL). Any product that had been merged
    into this one has its merged_into_id cleared.
    """
    product = session.get(ProductEntity, product_id)
    if product is None:
        return False

    session.execute(delete(NavObservation).where(NavObservation.product_id == product_id))
    session.execute(delete(NavCandidateVersion).where(NavCandidateVersion.product_id == product_id))
    session.execute(delete(StructuredFact).where(StructuredFact.product_id == product_id))
    session.execute(delete(ProductAlias).where(ProductAlias.product_id == product_id))

    for frag in session.execute(
        select(DocumentFragment).where(DocumentFragment.product_id == product_id)
    ).scalars().all():
        frag.product_id = None

    for merged in session.execute(
        select(ProductEntity).where(ProductEntity.merged_into_id == product_id)
    ).scalars().all():
        merged.merged_into_id = None

    session.delete(product)
    session.commit()
    return True


def delete_file(session: Session, file_id: str) -> dict[str, Any] | bool:
    """Delete a raw file and the research records that only it supported.

    A source-linked NAV point or fact cannot survive without its source: that
    would create an apparently research-ready, but unauditable, product.
    """
    record = session.get(RawFile, file_id)
    if record is None:
        return False

    storage = record.storage_path
    removed_nav_count = session.execute(
        delete(NavObservation).where(NavObservation.source_file_id == file_id)
    ).rowcount or 0
    removed_fact_count = session.execute(
        delete(StructuredFact).where(StructuredFact.source_file_id == file_id)
    ).rowcount or 0
    removed_alias_count = session.execute(
        delete(ProductAlias).where(ProductAlias.source_file_id == file_id)
    ).rowcount or 0
    session.delete(record)  # cascade deletes fragments
    session.commit()
    return {
        "storage_path": storage,
        "removed_nav_count": removed_nav_count,
        "removed_fact_count": removed_fact_count,
        "removed_alias_count": removed_alias_count,
    }


def bind_file_to_product(
    session: Session,
    file_id: str,
    product_id: str,
    *,
    bound_by: str = "user",
) -> dict[str, Any] | None:
    """Bind orphaned evidence from a file to a confirmed product choice.

    Only fragments that are currently unbound or explicitly marked as an
    unmatched curve are moved. Existing product associations are never
    silently overwritten. If parsing produced no fragment at all, a small
    manual-binding fragment is created so the file remains traceable and can
    be reparsed later.
    """
    record = session.get(RawFile, file_id)
    product = session.get(ProductEntity, product_id)
    if record is None or product is None:
        return None

    fragments = list(session.execute(
        select(DocumentFragment).where(DocumentFragment.file_id == file_id)
    ).scalars().all())
    bound_count = 0
    unresolved_fragment_ids: set[str] = set()
    unresolved_product_ids: set[str] = set()
    for fragment in fragments:
        content_data = dict(fragment.content_data or {})
        is_unmatched = content_data.get("binding_status") == "unmatched"
        if fragment.product_id is not None and not is_unmatched:
            continue
        if fragment.product_id is not None:
            unresolved_product_ids.add(fragment.product_id)
        unresolved_fragment_ids.add(fragment.id)
        fragment.product_id = product.id
        content_data.update({
            "binding_status": "matched",
            "binding_reason": "用户确认绑定",
            "bound_by": bound_by,
        })
        fragment.content_data = content_data
        bound_count += 1

    # Curves are often persisted as NAV observations against the temporary
    # "待绑定曲线" entity.  Move those observations together with their
    # unresolved fragment; keep a target product's existing value when the
    # same date is already present (usually a higher-confidence table value).
    bound_nav_count = 0
    dropped_duplicate_nav_count = 0
    nav_source_filter = []
    if unresolved_fragment_ids:
        nav_source_filter.append(NavObservation.source_fragment_id.in_(unresolved_fragment_ids))
    if unresolved_product_ids:
        nav_source_filter.append(
            (NavObservation.source_file_id == file_id)
            & NavObservation.product_id.in_(unresolved_product_ids)
        )
    if nav_source_filter:
        nav_rows = session.execute(
            select(NavObservation).where(nav_source_filter[0] if len(nav_source_filter) == 1 else nav_source_filter[0] | nav_source_filter[1])
        ).scalars().all()
        for observation in nav_rows:
            if observation.product_id == product.id:
                continue
            duplicate_id = session.execute(
                select(NavObservation.id).where(
                    NavObservation.product_id == product.id,
                    NavObservation.observation_date == observation.observation_date,
                    NavObservation.id != observation.id,
                )
            ).scalar()
            if duplicate_id is not None:
                session.delete(observation)
                dropped_duplicate_nav_count += 1
            else:
                observation.product_id = product.id
                bound_nav_count += 1

    bound_fact_count = 0
    if unresolved_fragment_ids or unresolved_product_ids:
        fact_stmt = select(StructuredFact).where(StructuredFact.source_file_id == file_id)
        if unresolved_fragment_ids:
            fact_stmt = fact_stmt.where(
                StructuredFact.source_fragment_id.in_(unresolved_fragment_ids)
                | StructuredFact.product_id.in_(unresolved_product_ids or {"__none__"})
            )
        elif unresolved_product_ids:
            fact_stmt = fact_stmt.where(StructuredFact.product_id.in_(unresolved_product_ids))
        for fact in session.execute(fact_stmt).scalars().all():
            if fact.product_id != product.id:
                fact.product_id = product.id
                bound_fact_count += 1

    # The image pipeline may create a temporary entity named
    # "文件名 / 待绑定曲线 N" so an unmatched curve remains visible.  Once its
    # evidence has been moved, remove that empty placeholder; otherwise the
    # product list would show a confusing duplicate beside the real target.
    cleaned_placeholder_count = 0
    for old_product_id in unresolved_product_ids:
        if old_product_id == product.id:
            continue
        old_product = session.get(ProductEntity, old_product_id)
        if old_product is None or "待绑定曲线" not in (old_product.standard_name or ""):
            continue
        has_fragment = session.execute(
            select(DocumentFragment.id).where(DocumentFragment.product_id == old_product_id).limit(1)
        ).scalar() is not None
        has_nav = session.execute(
            select(NavObservation.id).where(NavObservation.product_id == old_product_id).limit(1)
        ).scalar() is not None
        has_fact = session.execute(
            select(StructuredFact.id).where(StructuredFact.product_id == old_product_id).limit(1)
        ).scalar() is not None
        has_alias = session.execute(
            select(ProductAlias.id).where(ProductAlias.product_id == old_product_id).limit(1)
        ).scalar() is not None
        if not (has_fragment or has_nav or has_fact or has_alias):
            session.delete(old_product)
            cleaned_placeholder_count += 1

    created_manual_fragment = False
    if not fragments:
        add_fragment(
            session,
            file_id=file_id,
            fragment_type="manual_binding",
            content_text="用户手动绑定文件与产品，等待重新解析或补录净值",
            content_data={
                "method": "用户手动绑定",
                "binding_status": "matched",
                "binding_reason": "解析未生成可关联片段",
                "bound_by": bound_by,
            },
            product_id=product.id,
        )
        created_manual_fragment = True

    # Keep the original filename searchable as an alias, but avoid duplicate
    # aliases when the same binding is repeated.
    alias_text = (record.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    alias_exists = session.execute(
        select(ProductAlias.id).where(
            ProductAlias.product_id == product.id,
            ProductAlias.source_file_id == file_id,
            ProductAlias.alias == alias_text,
        )
    ).scalar()
    if alias_text and alias_exists is None:
        session.add(ProductAlias(
            product_id=product.id,
            alias=alias_text,
            alias_type="source_filename",
            source_file_id=file_id,
        ))

    session.commit()
    return {
        "file_id": file_id,
        "product_id": product.id,
        "product_name": product.standard_name,
        "bound_fragments": bound_count,
        "bound_nav": bound_nav_count,
        "dropped_duplicate_nav": dropped_duplicate_nav_count,
        "bound_facts": bound_fact_count,
        "cleaned_placeholders": cleaned_placeholder_count,
        "created_manual_fragment": created_manual_fragment,
    }


# ---------------------------------------------------------------------------
# NAV Observation operations
# ---------------------------------------------------------------------------


def create_nav_candidate_version(
    session: Session, product_id: str, source_file_id: str, points: list[dict[str, Any]], *,
    source_fragment_id: str | None = None, frequency: str | None = None, confidence: float | None = None,
) -> NavCandidateVersion:
    """Persist extracted points without altering the product's official NAV."""
    candidate = NavCandidateVersion(
        product_id=product_id, source_file_id=source_file_id, source_fragment_id=source_fragment_id,
        points=[{**point, "observation_date": str(point["observation_date"])} for point in points],
        frequency=frequency, confidence=confidence,
    )
    session.add(candidate)
    session.commit()
    session.refresh(candidate)
    return candidate


def list_nav_candidate_versions(session: Session, product_id: str) -> list[NavCandidateVersion]:
    return list(session.execute(
        select(NavCandidateVersion).where(NavCandidateVersion.product_id == product_id)
        .order_by(NavCandidateVersion.created_at.desc())
    ).scalars().all())


def add_nav_observations(
    session: Session,
    product_id: str,
    points: list[dict[str, Any]],
    *,
    source_file_id: str | None = None,
    source_fragment_id: str | None = None,
    frequency: str | None = None,
) -> int:
    """Bulk-insert NAV points. Skips duplicates by (product_id, date). Returns inserted count."""
    inserted = 0
    for point in points:
        obs_date = point["observation_date"]
        if isinstance(obs_date, str):
            obs_date = date.fromisoformat(obs_date)
        exists = session.execute(
            select(NavObservation.id).where(
                NavObservation.product_id == product_id,
                NavObservation.observation_date == obs_date,
            )
        ).scalar()
        if exists is not None:
            continue
        session.add(NavObservation(
            product_id=product_id,
            observation_date=obs_date,
            nav=point["nav"],
            acc_nav=point.get("acc_nav"),
            frequency=frequency,
            source_file_id=source_file_id,
            source_fragment_id=source_fragment_id,
        ))
        inserted += 1
    session.commit()
    return inserted


def replace_nav_observations(
    session: Session,
    product_id: str,
    points: list[dict[str, Any]],
    *,
    source_file_id: str | None = None,
    frequency: str | None = None,
    reviewed_by: str = "user",
) -> int:
    """Replace a product's candidate NAV series with a user-reviewed series."""
    session.query(NavObservation).where(NavObservation.product_id == product_id).delete()
    for point in points:
        obs_date = point["observation_date"]
        if isinstance(obs_date, str):
            obs_date = date.fromisoformat(obs_date)
        session.add(NavObservation(
            product_id=product_id,
            observation_date=obs_date,
            nav=float(point["nav"]),
            acc_nav=point.get("acc_nav"),
            frequency=frequency,
            source_file_id=source_file_id,
            review_status=ReviewStatus.REVIEWED,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.now(),
        ))
    session.commit()
    return len(points)


def finalize_manual_nav_review(
    session: Session,
    *,
    file_id: str,
    product_id: str,
    fragment_id: str | None = None,
    reviewed_by: str = "user",
) -> dict[str, Any]:
    """Persist the one-time human decision for a source material.

    A manual NAV save is a terminal review decision for the selected source
    evidence. It marks that evidence reviewed, retires machine candidates
    replaced by the manual series, and records a terminal workflow state so a
    later product-library refresh never schedules VLM/CV again for it.
    """
    record = session.get(RawFile, file_id)
    if record is None:
        raise ValueError("来源文件不存在")
    if fragment_id is not None:
        fragment = session.get(DocumentFragment, fragment_id)
        if fragment is None or fragment.file_id != file_id:
            raise ValueError("来源片段不属于该文件")
        if fragment.product_id not in {None, product_id}:
            raise ValueError("来源片段不属于当前产品")
        fragments = [fragment]
    else:
        fragments = list(session.execute(
            select(DocumentFragment).where(
                DocumentFragment.file_id == file_id,
                DocumentFragment.product_id == product_id,
            )
        ).scalars().all())

    reviewed_at = datetime.now()
    for fragment in fragments:
        content_data = dict(fragment.content_data or {})
        content_data.update({
            "review_status": "reviewed",
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at.isoformat(),
            "workflow_stage": "human_reviewed",
        })
        fragment.content_data = content_data

    candidate_query = select(NavCandidateVersion).where(
        NavCandidateVersion.source_file_id == file_id,
        NavCandidateVersion.product_id == product_id,
        NavCandidateVersion.status == "pending",
    )
    if fragment_id is not None:
        candidate_query = candidate_query.where(NavCandidateVersion.source_fragment_id == fragment_id)
    retired_candidates = list(session.execute(candidate_query).scalars().all())
    for candidate in retired_candidates:
        candidate.status = "discarded"

    unresolved = session.execute(
        select(DocumentFragment.id).where(
            DocumentFragment.file_id == file_id,
            (DocumentFragment.product_id.is_(None))
            | (DocumentFragment.content_data["binding_status"].as_string() == "unmatched"),
        )
    ).scalars().all()
    audit = dict(record.extraction_audit or {})
    workflow = dict(audit.get("workflow") or {})
    workflow.update({
        "stage": "human_review",
        "identity": "resolved",
        "trace": "human_reviewed",
        "next_action": "confirm_product_binding" if unresolved else "none",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at.isoformat(),
    })
    audit["workflow"] = workflow
    audit["human_review"] = {
        "status": "permanently_saved",
        "product_id": product_id,
        "fragment_id": fragment_id,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at.isoformat(),
    }
    record.extraction_audit = audit
    if unresolved:
        record.parsing_status = "completed_no_nav"
        record.parsing_error = "当前产品净值已人工审核；文件仍有其他曲线或片段待确认绑定"
    else:
        record.parsing_status = "completed"
        record.parsing_error = None
    session.commit()
    return {
        "file_id": file_id,
        "product_id": product_id,
        "fragment_id": fragment_id,
        "reviewed_fragments": len(fragments),
        "retired_candidates": len(retired_candidates),
    }


def get_nav_series(
    session: Session, product_id: str, *, reviewed_only: bool = False
) -> list[NavObservation]:
    stmt = (
        select(NavObservation)
        .where(NavObservation.product_id == product_id)
        .order_by(NavObservation.observation_date)
    )
    if reviewed_only:
        stmt = stmt.where(NavObservation.review_status == ReviewStatus.REVIEWED)
    return list(session.execute(stmt).scalars().all())


def get_nav_series_bulk(
    session: Session, product_ids: list[str], *, reviewed_only: bool = False
) -> dict[str, list[Row]]:
    """单次查询批量读取多只产品的净值序列，按 product_id 分组返回。

    与逐只调用 get_nav_series 返回相同的字段（Row 支持属性访问），但走
    Core 列查询、不构造 ORM 对象：排名/评分宇宙构建一次要取 700+ 只
    产品、18 万+ 净值点，ORM 物化是主要耗时（约 5s），列查询可压到 1s 内。
    """
    grouped: dict[str, list[Row]] = {product_id: [] for product_id in product_ids}
    if not product_ids:
        return grouped
    stmt = (
        select(
            NavObservation.id,
            NavObservation.product_id,
            NavObservation.observation_date,
            NavObservation.nav,
            NavObservation.acc_nav,
            NavObservation.frequency,
            NavObservation.source_file_id,
            NavObservation.review_status,
        )
        .where(NavObservation.product_id.in_(product_ids))
        .order_by(NavObservation.product_id, NavObservation.observation_date)
    )
    if reviewed_only:
        stmt = stmt.where(NavObservation.review_status == ReviewStatus.REVIEWED)
    for row in session.execute(stmt):
        grouped.setdefault(row.product_id, []).append(row)
    return grouped


def get_nav_stats_bulk(session: Session, product_ids: list[str]) -> dict[str, dict[str, Any]]:
    """单查询聚合多只产品的净值统计，供产品列表载荷使用。

    列表页每只产品都要展示点数、复核数、起止日期与来源状态；逐只物化
    全部净值点（18 万+ 行 ORM 对象）是主要耗时。这里全部交给 SQL 聚合。
    """
    if not product_ids:
        return {}
    stmt = (
        select(
            NavObservation.product_id,
            func.count().label("nav_count"),
            func.sum(case((NavObservation.review_status == ReviewStatus.REVIEWED, 1), else_=0)).label("reviewed_count"),
            func.min(NavObservation.observation_date).label("nav_start"),
            func.max(NavObservation.observation_date).label("nav_end"),
            # 任一净值点缺来源文件则为 0，全部有来源为 1
            func.min(case((NavObservation.source_file_id.is_(None), 0), else_=1)).label("all_have_source"),
            # 任一净值点不是 futures_weekly_sqlite 直连来源则为 0，全部是则为 1
            func.min(case((RawFile.source == "futures_weekly_sqlite", 1), else_=0)).label("all_direct"),
        )
        .select_from(NavObservation)
        .outerjoin(RawFile, NavObservation.source_file_id == RawFile.id)
        .where(NavObservation.product_id.in_(product_ids))
        .group_by(NavObservation.product_id)
    )
    result: dict[str, dict[str, Any]] = {}
    for row in session.execute(stmt):
        result[row.product_id] = {
            "nav_count": int(row.nav_count or 0),
            "reviewed_count": int(row.reviewed_count or 0),
            "nav_start": row.nav_start,
            "nav_end": row.nav_end,
            "all_have_source": int(row.all_have_source or 1),
            "all_direct": int(row.all_direct or 0),
        }
    return result


def get_nav_source_info_bulk(
    session: Session, product_ids: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """返回每只产品净值点的去重来源文件名与来源文件 ID。

    按 (product_id, file_id) 分组，只返回每只产品几个去重文件，而不是
    逐行遍历全部净值点。
    """
    names: dict[str, list[str]] = {}
    file_ids: dict[str, list[str]] = {}
    if not product_ids:
        return names, file_ids
    stmt = (
        select(NavObservation.product_id, RawFile.id, RawFile.filename)
        .join(RawFile, NavObservation.source_file_id == RawFile.id)
        .where(NavObservation.product_id.in_(product_ids))
        .group_by(NavObservation.product_id, RawFile.id, RawFile.filename)
        .order_by(NavObservation.product_id, RawFile.id)
    )
    for product_id, file_id, filename in session.execute(stmt):
        names.setdefault(product_id, []).append(filename)
        file_ids.setdefault(product_id, []).append(file_id)
    return names, file_ids


def review_nav_observations(
    session: Session, observation_ids: list[str], status: str, reviewed_by: str = "user"
) -> int:
    """Batch-update review status. Returns count updated."""
    updated = 0
    for obs_id in observation_ids:
        obs = session.get(NavObservation, obs_id)
        if obs is not None:
            obs.review_status = status
            obs.reviewed_by = reviewed_by
            obs.reviewed_at = datetime.now()
            updated += 1
    session.commit()
    return updated


# ---------------------------------------------------------------------------
# Structured Fact operations
# ---------------------------------------------------------------------------


def add_fact(
    session: Session,
    *,
    product_id: str,
    field_name: str,
    field_value: str,
    source_file_id: str | None = None,
    source_fragment_id: str | None = None,
    confidence: float | None = None,
) -> StructuredFact:
    fact = StructuredFact(
        product_id=product_id,
        field_name=field_name,
        field_value=field_value,
        source_file_id=source_file_id,
        source_fragment_id=source_fragment_id,
        confidence=confidence,
    )
    session.add(fact)
    session.commit()
    session.refresh(fact)
    return fact


def supersede_fact(session: Session, old_fact_id: str, new_fact: StructuredFact) -> StructuredFact:
    """Replace an old fact with a new version, preserving history."""
    old = session.get(StructuredFact, old_fact_id)
    if old is not None:
        new_fact.extraction_version = old.extraction_version + 1
        old.superseded_by = new_fact.id
    session.add(new_fact)
    session.commit()
    session.refresh(new_fact)
    return new_fact


def get_facts_for_product(
    session: Session, product_id: str, *, current_only: bool = True
) -> list[StructuredFact]:
    stmt = select(StructuredFact).where(StructuredFact.product_id == product_id)
    if current_only:
        stmt = stmt.where(StructuredFact.superseded_by.is_(None))
    return list(session.execute(stmt.order_by(StructuredFact.field_name)).scalars().all())


# ---------------------------------------------------------------------------
# Agent Run & Audit operations
# ---------------------------------------------------------------------------


def create_agent_run(
    session: Session,
    *,
    user_query: str,
    session_id: str | None = None,
    plan: dict[str, Any] | None = None,
) -> AgentRun:
    run = AgentRun(user_query=user_query, session_id=session_id, plan=plan)
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def complete_agent_run(
    session: Session,
    run_id: str,
    *,
    phase: str,
    tools_used: list[str] | None = None,
    citations: list[dict[str, Any]] | None = None,
    reflection: dict[str, Any] | None = None,
    answer: str | None = None,
    data_snapshot_id: str | None = None,
    duration_ms: float | None = None,
) -> AgentRun | None:
    run = session.get(AgentRun, run_id)
    if run is None:
        return None
    run.phase = phase
    run.tools_used = tools_used
    run.citations = citations
    run.reflection = reflection
    run.answer = answer
    run.data_snapshot_id = data_snapshot_id
    run.duration_ms = duration_ms
    session.commit()
    session.refresh(run)
    return run


def add_tool_invocation(
    session: Session,
    run_id: str,
    *,
    tool_name: str,
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    status: str = "ok",
    error_message: str | None = None,
    duration_ms: float | None = None,
) -> ToolInvocation:
    invocation = ToolInvocation(
        run_id=run_id,
        tool_name=tool_name,
        input_summary=input_summary,
        output_summary=output_summary,
        status=status,
        error_message=error_message,
        duration_ms=duration_ms,
    )
    session.add(invocation)
    session.commit()
    session.refresh(invocation)
    return invocation


def list_agent_runs(
    session: Session, *, session_id: str | None = None, limit: int = 20, offset: int = 0
) -> list[AgentRun]:
    stmt = select(AgentRun).order_by(AgentRun.created_at.desc())
    if session_id:
        stmt = stmt.where(AgentRun.session_id == session_id)
    return list(session.execute(stmt.offset(offset).limit(limit)).scalars().all())


# ---------------------------------------------------------------------------
# Data Snapshot & Decision operations
# ---------------------------------------------------------------------------


def build_research_evidence_snapshot(
    session: Session, product_ids: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Freeze the exact source records used by a product research run.

    The returned snapshot stores enough identifiers and immutable file-version
    metadata to reproduce an allocation later.  The flat citations list is
    shaped for Agent-run audit records and the chat response.
    """
    evidence_by_product: dict[str, list[dict[str, Any]]] = {}
    citations: list[dict[str, Any]] = []
    seen_citations: set[tuple[str, str | None]] = set()

    for product_id in dict.fromkeys(product_ids):
        product = session.get(ProductEntity, product_id)
        if product is None:
            continue
        observations = get_nav_series(session, product_id, reviewed_only=False)
        facts = get_facts_for_product(session, product_id)
        fragment_ids = {
            item.source_fragment_id for item in observations if item.source_fragment_id
        } | {
            item.source_fragment_id for item in facts if item.source_fragment_id
        }
        source_file_ids = {
            item.source_file_id for item in observations if item.source_file_id
        } | {
            item.source_file_id for item in facts if item.source_file_id
        }
        fragments = list(session.execute(
            select(DocumentFragment).where(
                (DocumentFragment.product_id == product_id)
                | DocumentFragment.id.in_(fragment_ids)
            )
        ).scalars().all())
        source_file_ids.update(fragment.file_id for fragment in fragments if fragment.file_id)
        files = {
            item.id: item for item in session.execute(
                select(RawFile).where(RawFile.id.in_(source_file_ids))
            ).scalars().all()
        } if source_file_ids else {}

        evidence_by_product[product_id] = [{
            "product_id": product.id,
            "product_name": product.standard_name,
            "nav_observation_ids": [item.id for item in observations],
            "fact_ids": [item.id for item in facts],
            "fragment_ids": [item.id for item in fragments],
            "files": [
                {
                    "file_id": item.id,
                    "filename": item.filename,
                    "file_hash": item.file_hash,
                    "version": item.version,
                    "report_period": item.report_period,
                }
                for item in files.values()
            ],
        }]
        for fragment in fragments:
            file = files.get(fragment.file_id)
            if file is None:
                continue
            key = (file.id, fragment.id)
            if key in seen_citations:
                continue
            seen_citations.add(key)
            citations.append({
                "file_id": file.id,
                "filename": file.filename,
                "page_number": fragment.page_number,
                "fragment_id": fragment.id,
                "fragment_type": fragment.fragment_type,
                "snippet": (fragment.content_text or "")[:280] or None,
                "confidence": fragment.ocr_confidence,
            })
        for file in files.values():
            key = (file.id, None)
            if key in seen_citations or any(existing_file_id == file.id for existing_file_id, _ in seen_citations):
                continue
            seen_citations.add(key)
            citations.append({
                "file_id": file.id,
                "filename": file.filename,
                "page_number": None,
                "fragment_id": None,
                "fragment_type": "source_file",
                "snippet": "净值来源文件（尚无可定位的材料片段）",
                "confidence": None,
            })
    return evidence_by_product, citations


def create_snapshot(
    session: Session, *, label: str | None = None, content: dict[str, Any]
) -> DataSnapshot:
    snapshot = DataSnapshot(label=label, content=content)
    session.add(snapshot)
    session.commit()
    session.refresh(snapshot)
    return snapshot


def create_decision(
    session: Session,
    *,
    decision_type: str,
    title: str | None = None,
    content: dict[str, Any],
    run_id: str | None = None,
    data_snapshot_id: str | None = None,
    status: str = DecisionStatus.DRAFT,
) -> DecisionRecord:
    decision = DecisionRecord(
        decision_type=decision_type,
        title=title,
        content=content,
        run_id=run_id,
        data_snapshot_id=data_snapshot_id,
        status=status,
    )
    session.add(decision)
    session.commit()
    session.refresh(decision)
    return decision


def review_decision(
    session: Session,
    decision_id: str,
    *,
    status: str,
    reviewer: str | None = None,
    veto_reason: str | None = None,
    adjustment: dict[str, Any] | None = None,
) -> DecisionRecord | None:
    decision = session.get(DecisionRecord, decision_id)
    if decision is None:
        return None
    decision.status = status
    decision.reviewer = reviewer
    decision.reviewed_at = datetime.now()
    decision.veto_reason = veto_reason
    decision.adjustment = adjustment
    session.commit()
    session.refresh(decision)
    return decision


def list_decisions(
    session: Session, *, status: str | None = None, limit: int = 20, offset: int = 0
) -> list[DecisionRecord]:
    stmt = select(DecisionRecord).order_by(DecisionRecord.created_at.desc())
    if status:
        stmt = stmt.where(DecisionRecord.status == status)
    return list(session.execute(stmt.offset(offset).limit(limit)).scalars().all())
