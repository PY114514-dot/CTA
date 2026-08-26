"""批量净值端点与 peers 缓存的回归测试（批次 13 体验）。

覆盖两类回归：
1. /api/kb/nav/bulk 的响应必须是 JSON 对象而不是字符串——批次 13 曾用
   JSONResponse 发送预序列化字符串，导致双重编码（响应变成 JSON 字符串）。
2. /api/product-archive/products/{id}/peers 的进程级缓存：冷热两次返回
   内容一致，且条目按请求产品 id 存储（曾有循环变量遮蔽参数的 bug）。
"""

from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_session
from app.main import app
from app.models import ConfirmationStatus, NavObservation, ProductEntity, ReviewStatus
from app.routers import product_archive


def _session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _seed(session, product_id: str, name: str, strategy: str, n_points: int = 20) -> None:
    product = ProductEntity(
        id=product_id,
        standard_name=name,
        manager_name="测试管理人",
        strategy=strategy,
        nav_frequency="weekly",
        confirmation_status=ConfirmationStatus.CONFIRMED,
    )
    session.add(product)
    nav = 1.0
    start = date(2024, 1, 5)
    for index in range(n_points):
        nav *= 1.0 + (0.002 if index % 2 == 0 else -0.001)
        session.add(NavObservation(
            product_id=product_id,
            observation_date=start + timedelta(days=index * 7),
            nav=round(nav, 6),
            acc_nav=None,
            frequency="weekly",
            source_file_id=None,
            review_status=ReviewStatus.REVIEWED,
        ))
    session.commit()


def test_nav_bulk_response_is_object_not_string() -> None:
    """回归：预序列化发送时必须用 Response，双重编码会把对象变成字符串。"""
    factory = _session_factory()
    session = factory()
    _seed(session, "bulk-a", "产品 A", "量化期货", 20)
    _seed(session, "bulk-b", "产品 B", "量化期货", 24)
    session.close()

    app.dependency_overrides[get_session] = lambda: factory()
    try:
        with TestClient(app) as client:
            response = client.post("/api/kb/nav/bulk", json={
                "product_ids": ["bulk-a", "bulk-b"],
                "reviewed_only": True,
            })
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 200
    body = response.json()
    # 双重编码的响应 json() 会得到 str；正确实现应是 product_id → 点列表。
    assert isinstance(body, dict), "批量响应必须是对象，不能是字符串"
    assert set(body) == {"bulk-a", "bulk-b"}
    assert len(body["bulk-a"]) == 20
    assert len(body["bulk-b"]) == 24
    point = body["bulk-a"][0]
    assert set(point) == {"id", "observation_date", "nav", "acc_nav", "frequency", "source_file_id", "review_status"}


def test_nav_bulk_matches_single_product_endpoint() -> None:
    factory = _session_factory()
    session = factory()
    _seed(session, "same-a", "产品 A", "量化期货", 15)
    session.close()

    app.dependency_overrides[get_session] = lambda: factory()
    try:
        with TestClient(app) as client:
            bulk = client.post("/api/kb/nav/bulk", json={"product_ids": ["same-a"], "reviewed_only": True})
            single = client.get("/api/kb/nav/same-a?reviewed_only=true")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert bulk.status_code == 200
    assert single.status_code == 200
    assert bulk.json()["same-a"] == single.json()


def test_nav_bulk_gzip_middleware_compresses_large_body() -> None:
    """GZip 中间件只压超过 minimum_size=1024 的响应。

    注意：TestClient 底层 httpx 会自动解压 gzip 响应，因此 content 已是
    解压后的字节；httpx 能成功解码本身就证明压缩流有效。这里断言
    Content-Encoding 头，以及解压后内容与不压缩时逐字节一致。
    """
    factory = _session_factory()
    session = factory()
    _seed(session, "gzip-a", "产品 A", "量化期货", 200)
    session.close()

    app.dependency_overrides[get_session] = lambda: factory()
    try:
        with TestClient(app) as client:
            plain = client.post("/api/kb/nav/bulk", json={"product_ids": ["gzip-a"]})
            compressed = client.post(
                "/api/kb/nav/bulk",
                json={"product_ids": ["gzip-a"]},
                headers={"Accept-Encoding": "gzip"},
            )
            # httpx 与浏览器一样默认发 Accept-Encoding: gzip，所以上面两个
            # 请求传输层其实都走了压缩；用 /api/health（16 字节）验证
            # minimum_size 阈值：低于 1024 的响应不压。
            small = client.get("/api/health", headers={"Accept-Encoding": "gzip"})
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert compressed.status_code == 200
    assert compressed.headers.get("Content-Encoding") == "gzip"
    assert compressed.content == plain.content
    assert small.headers.get("Content-Encoding") is None


def test_peers_cache_hit_returns_identical_content() -> None:
    factory = _session_factory()
    session = factory()
    _seed(session, "peer-target", "目标产品", "量化期货", 30)
    _seed(session, "peer-1", "同类产品 1", "量化期货", 30)
    _seed(session, "peer-2", "同类产品 2", "量化期货", 30)
    session.close()
    product_archive._peers_cache.clear()

    app.dependency_overrides[get_session] = lambda: factory()
    try:
        with TestClient(app) as client:
            cold = client.get("/api/product-archive/products/peer-target/peers")
            warm = client.get("/api/product-archive/products/peer-target/peers")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert cold.status_code == 200
    assert warm.status_code == 200
    assert cold.json() == warm.json()
    # 缓存条目必须挂在请求的产品 id 下（回归：循环变量曾遮蔽参数导致存错键）。
    assert "peer-target" in product_archive._peers_cache
    product_archive._peers_cache.clear()
