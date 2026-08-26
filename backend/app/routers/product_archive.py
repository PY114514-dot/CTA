"""Lightweight, deterministic product research archive for screening."""
from __future__ import annotations
import math
from collections import defaultdict
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends
from app.database import get_session
from app.models import DataSnapshot, NavObservation, ProductEntity, ReviewStatus

router = APIRouter(prefix="/api/product-archive", tags=["产品研究库"])
_archive_cache: tuple[tuple[str | None, str | None], dict] | None = None
# 批次 13（体验）：peers 端点按产品缓存。键 = (已审核净值行数, 净值总和)
# 两数聚合签名，任何净值新增/删除/改值都会改变签名；比绑定排名快照更
# 及时（手工改净值不必等周度排名刷新）。聚合一次约 100ms，命中即省下
# 1.2s 的相关矩阵重算。
_peers_cache: dict[str, tuple[tuple[int, float], dict]] = {}
_PEERS_CACHE_MAX = 1024


def _universe_size_expr(session: Session):
    """按后端方言解析快照内容的 universe_size 标量，避免整块 JSON 反序列化。

    SQLite 的 content 存为 TEXT，用 json_extract；PostgreSQL 是 JSONB，
    用下标访问。评分快照内容可达 16.9MB，逐块反序列化只为取一个数字
    是产品研究库列表端点的主要耗时之一。
    """
    if session.get_bind().dialect.name == "postgresql":
        return DataSnapshot.content["universe_size"].as_integer().label("universe_size")
    return func.json_extract(DataSnapshot.content, "$.universe_size").label("universe_size")


def _current_ranking_snapshot(session: Session) -> DataSnapshot | None:
    """Use the broadest ranking run, not a later single-product inspection."""
    rows = session.execute(
        select(DataSnapshot.id, DataSnapshot.created_at, _universe_size_expr(session))
        .where(DataSnapshot.label.like("codex-cta:%"))
    ).all()
    if not rows:
        return None
    best_id = max(
        rows,
        key=lambda row: (int(row.universe_size or 0), row.created_at.timestamp() if row.created_at else 0),
    ).id
    return session.get(DataSnapshot, best_id)


def _current_score_snapshot(session: Session) -> DataSnapshot | None:
    rows = session.execute(
        select(DataSnapshot.id, DataSnapshot.created_at, _universe_size_expr(session))
        .where(DataSnapshot.label.like("cta-product-score:%"))
    ).all()
    if not rows:
        return None
    best_id = max(
        rows,
        key=lambda row: (int(row.universe_size or 0), row.created_at.timestamp() if row.created_at else 0),
    ).id
    return session.get(DataSnapshot, best_id)


@router.get("/products")
def list_product_archive(session: Session = Depends(get_session)) -> dict:
    global _archive_cache
    snapshot = _current_ranking_snapshot(session)
    score_snapshot = _current_score_snapshot(session)
    cache_key = (snapshot.id if snapshot else None, score_snapshot.id if score_snapshot else None)
    if _archive_cache is not None and _archive_cache[0] == cache_key:
        return _archive_cache[1]
    ranking = {item["product_id"]: item for item in (snapshot.content or {}).get("rankings", [])} if snapshot else {}
    score_summaries = {
        item["product_id"]: item.get("summary", {})
        for item in (score_snapshot.content or {}).get("products", [])
    } if score_snapshot else {}
    products = session.execute(select(ProductEntity).where(ProductEntity.confirmation_status == "confirmed")).scalars().all()
    # 只取 (product_id, nav) 两列，避免 18 万行 ORM 对象物化。
    rows = session.execute(select(NavObservation.product_id, NavObservation.nav).where(NavObservation.review_status == ReviewStatus.REVIEWED, NavObservation.product_id.in_([p.id for p in products])).order_by(NavObservation.product_id, NavObservation.observation_date)).all() if products else []
    navs: dict[str, list[float]] = defaultdict(list)
    for product_id, nav in rows:
        if nav > 0: navs[product_id].append(float(nav))
    items=[]
    for product in products:
        values=navs[product.id]; returns=[right/left-1 for left,right in zip(values,values[1:])]
        periods={"daily":252,"monthly":12}.get(product.nav_frequency or "weekly",52)
        total=values[-1]/values[0]-1 if len(values)>=2 else None
        annual=(1+total)**(periods/len(returns))-1 if total is not None and total>-1 and returns else None
        mean=sum(returns)/len(returns) if returns else 0; variance=sum((x-mean)**2 for x in returns)/(len(returns)-1) if len(returns)>1 else 0
        vol=math.sqrt(variance*periods) if variance else 0
        peak=values[0] if values else 0; drawdown=0.0
        for value in values:
            peak=max(peak,value); drawdown=min(drawdown,value/peak-1)
        score=ranking.get(product.id,{})
        summary = score_summaries.get(product.id, {})
        headline = summary.get("headline", {})
        data_status = "高回撤" if drawdown <= -0.65 else "正常"
        items.append({"product_id":product.id,"name":product.standard_name,"manager":product.manager_name,"strategy":product.strategy,"nav_count":len(values),"nav_end":product.nav_end.isoformat() if hasattr(product,'nav_end') else None,"annual_return":annual,"annual_volatility":vol,"sharpe":(annual-.015)/vol if annual is not None and vol>1e-12 else None,"maximum_drawdown":drawdown,"data_status":data_status,"rank":score.get("rank"),"quality_score":summary.get("quality_score", score.get("score")),"confidence_score":summary.get("confidence_score"),"attribution_quality_score":summary.get("attribution_quality_score"),"r_squared":headline.get("r_squared"),"oos_r_squared":headline.get("oos_r_squared")})
    result = {"items":items,"ranking_as_of":(snapshot.content or {}).get("as_of_date") if snapshot else None,"score_as_of":(score_snapshot.content or {}).get("as_of_date") if score_snapshot else None}
    _archive_cache = (cache_key, result)
    return result


@router.get("/products/{product_id}/peers")
def product_peers(product_id: str, session: Session = Depends(get_session)) -> dict:
    # 净值数据签名：COUNT 捕捉增删，SUM 捕捉改值；UUID 主键无序，MAX(id)
    # 无意义。命中缓存直接返回，跳过 ~1.2s 的相关矩阵重算。
    nav_count, nav_sum = session.execute(
        select(
            func.count(),
            func.coalesce(func.sum(NavObservation.nav), 0.0),
        ).where(NavObservation.review_status == ReviewStatus.REVIEWED)
    ).one()
    signature = (int(nav_count), round(float(nav_sum), 4))
    cached = _peers_cache.get(product_id)
    if cached is not None and cached[0] == signature:
        return cached[1]
    target = session.get(ProductEntity, product_id)
    if target is None:
        result: dict = {"similar": [], "diversifiers": []}
        _peers_cache[product_id] = (signature, result)
        return result
    peers = session.execute(select(ProductEntity).where(
        ProductEntity.confirmation_status == "confirmed",
        ProductEntity.nav_frequency == target.nav_frequency,
        ProductEntity.strategy == target.strategy,
        ProductEntity.id != target.id,
    )).scalars().all()
    ids = [target.id, *(item.id for item in peers)]
    # 只取三列，避免同策略产品全量净值点的 ORM 对象物化。
    rows = session.execute(select(NavObservation.product_id, NavObservation.observation_date, NavObservation.nav).where(NavObservation.review_status == ReviewStatus.REVIEWED, NavObservation.product_id.in_(ids)).order_by(NavObservation.product_id, NavObservation.observation_date)).all()
    navs: dict[str, list[tuple[object, float]]] = defaultdict(list)
    for row_product_id, observation_date, nav in rows:
        if nav > 0: navs[row_product_id].append((observation_date, float(nav)))
    def returns(product: str) -> dict:
        values = navs[product]
        return {right_day: right / left - 1 for (left_day, left), (right_day, right) in zip(values, values[1:])}
    target_returns = returns(target.id)
    output=[]
    for peer in peers:
        peer_returns=returns(peer.id); dates=sorted(set(target_returns)&set(peer_returns))
        if len(dates)<12: continue
        left=[target_returns[day] for day in dates]; right=[peer_returns[day] for day in dates]
        left_mean=sum(left)/len(left); right_mean=sum(right)/len(right)
        denominator=math.sqrt(sum((x-left_mean)**2 for x in left)*sum((x-right_mean)**2 for x in right))
        if denominator <= 1e-12: continue
        correlation=sum((x-left_mean)*(y-right_mean) for x,y in zip(left,right))/denominator
        output.append({"product_id":peer.id,"name":peer.standard_name,"manager":peer.manager_name,"correlation":round(correlation,4),"overlap":len(dates)})
    result = {"similar":sorted(output,key=lambda item:item["correlation"],reverse=True)[:5],"diversifiers":sorted(output,key=lambda item:abs(item["correlation"]))[:5]}
    if len(_peers_cache) >= _PEERS_CACHE_MAX:
        _peers_cache.clear()
    _peers_cache[product_id] = (signature, result)
    return result
