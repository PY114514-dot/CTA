"""Import the reviewed weekly-futures SQLite export into the product library."""
from __future__ import annotations
import random
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models import ConfirmationStatus, DocumentFragment, NavObservation, ProductAlias, ProductEntity, RawFile, ReviewStatus

_CUTOFF = date(2026, 7, 31)


def _read_source_rows(path: Path) -> list[dict[str, object]]:
    """Build a normalized NAV path from the source's validated weekly returns."""
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"product_info", "weekly_return_raw", "nav_weekly_snapshot"} <= tables:
            raise ValueError("不是支持的期货周频 SQLite 文件")
        rows = con.execute(
            """SELECT p.fid,p.name,p.manager,p.strategy_one,p.strategy_two,p.inception_date,n.week,
                      COALESCE(n.nav_cn,n.cum_nav_cnw,n.unit_nav_pn) nav,r.weekly_return_value
               FROM product_info p
               JOIN weekly_return_raw r ON r.fid=p.fid
               JOIN nav_weekly_snapshot n ON n.fid=r.fid AND n.week=r.week
               WHERE r.week<=? AND r.weekly_return_value>-1 AND abs(r.weekly_return_value)<=.30
                 AND n.nav_status IN ('exact','prior') AND n.nav_lag_days<=7
                 AND COALESCE(n.nav_cn,n.cum_nav_cnw,n.unit_nav_pn)>0
               ORDER BY p.fid,n.week""",
            (_CUTOFF.isoformat(),),
        ).fetchall()
    finally:
        con.close()
    normalized: list[dict[str, object]] = []
    nav = 1.0
    for index, row in enumerate(rows):
        if index and row["fid"] == rows[index - 1]["fid"]:
            nav *= 1.0 + float(row["weekly_return_value"])
        else:
            nav = 1.0
        normalized.append({**dict(row), "nav": nav})
    return normalized

def import_futures_weekly_sqlite(session: Session, *, path: Path, raw_file: RawFile, limit: int = 100, seed: int = 20260820, confirm_products: bool = False) -> dict:
    if not 1 <= limit <= 1000: raise ValueError("产品数量必须在 1 到 1000 之间")
    rows = _read_source_rows(path)
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows: grouped[row["fid"]].append(row)
    eligible = [fid for fid, values in grouped.items() if len(values) >= 104 and values[-1]["week"] >= _CUTOFF.isoformat()]
    def group(fid: str) -> str:
        text = f"{grouped[fid][0]['strategy_one'] or ''} {grouped[fid][0]['strategy_two'] or ''}"
        return "量化期货" if "量化期货" in text else "主观期货" if "主观期货" in text else "未细分"
    rng = random.Random(seed)
    quotas = (("量化期货", 67), ("主观期货", 30), ("未细分", 3))
    selected = [fid for label, count in quotas for fid in rng.sample(sorted(fid for fid in eligible if group(fid) == label), count)] if limit == 100 and all(sum(group(fid) == label for fid in eligible) >= count for label, count in quotas) else rng.sample(sorted(eligible), min(limit, len(eligible)))
    fragment = DocumentFragment(file_id=raw_file.id, fragment_type="table", content_text="期货周频 SQLite 结构化净值导入", content_data={"method":"futures_weekly_sqlite_v1", "cutoff":_CUTOFF.isoformat(), "seed":seed}, ocr_confidence=1.0)
    session.add(fragment); session.flush(); created = existing = observations = updated = 0
    for fid in selected:
        source = grouped[fid]
        product = session.execute(select(ProductEntity).join(ProductAlias).where(ProductAlias.alias == fid, ProductAlias.alias_type == "source_fid")).scalars().first()
        if product is None:
            item = source[0]
            product = ProductEntity(standard_name=item["name"] or fid, manager_name=item["manager"], strategy=group(fid), inception_date=date.fromisoformat(item["inception_date"]) if item["inception_date"] else None, nav_frequency="weekly", confirmation_status=ConfirmationStatus.CONFIRMED if confirm_products else ConfirmationStatus.PENDING, confirmed_by="user:sqlite_import" if confirm_products else None)
            session.add(product); session.flush(); session.add(ProductAlias(product_id=product.id, alias=fid, alias_type="source_fid", source_file_id=raw_file.id)); created += 1
        else: existing += 1
        known = {
            item.observation_date: item
            for item in session.execute(select(NavObservation).where(NavObservation.product_id == product.id)).scalars()
        }
        for row in source:
            day = date.fromisoformat(row["week"])
            nav = float(row["nav"])
            observation = known.get(day)
            if observation is None:
                session.add(NavObservation(product_id=product.id, observation_date=day, nav=float(row["nav"]), frequency="weekly", source_file_id=raw_file.id, source_fragment_id=fragment.id, review_status=ReviewStatus.REVIEWED, reviewed_by="sqlite_import")); observations += 1
            elif observation.nav != nav:
                observation.nav = nav
                observation.frequency = "weekly"
                observation.review_status = ReviewStatus.REVIEWED
                observation.reviewed_by = "sqlite_import"
                updated += 1
    raw_file.parsing_status = "completed"; raw_file.extraction_audit = {"method":"futures_weekly_sqlite_v1", "products":len(selected), "observations":observations}; session.commit()
    from app.services.cta_ranking_refresh import refresh_weekly_ranking
    ranking = refresh_weekly_ranking(session)
    # 导入完成后在后台补全 Phase-D 归因证据（幂等：已冻结且净值指纹未变的
    # 产品直接复用）。批量任务不阻塞导入响应，进度见 /phase-d/batch-status。
    from app.services.phase_d_batch import get_batch_runner, list_phase_d_batch_candidates
    runner = get_batch_runner()
    if runner.status()["status"] != "running":
        runner.start(list_phase_d_batch_candidates(session))
    return {"products_selected":len(selected), "products_created":created, "products_existing":existing, "nav_observations_added":observations, "nav_observations_updated":updated, "ranking":ranking, "cutoff":_CUTOFF.isoformat(), "seed":seed}
