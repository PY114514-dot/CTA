"""后台批量生成 Phase-D 归因证据快照。

批次 13（准确度）：周度排名的稳健性维度需要每只产品的 Phase-D 证据，
但证据此前从未批量生成过（快照数为 0，排名里的归因证据维度一直空转）。
本模块按批次 7 资料解析队列的同一套可见性约定实现后台批量：当前处理
对象、并行上限、排队数量全部可查询；每个结果都冻结为幂等快照，已存在
且净值指纹未变的快照直接复用。

执行器是进程级单例，由 HTTP 端点触发；进程重启后状态归零，已冻结的
快照持久化在 DataSnapshot 中不会丢失。
"""

from __future__ import annotations

import queue
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionFactory
from app.models import ConfirmationStatus, DataSnapshot, ProductEntity
from app.services import product_store
from app.services.cta_attribution_snapshot import nav_fingerprint

PHASE_D_CONCURRENCY = 4
PHASE_D_MAX_CONCURRENCY = 16


def freeze_attribution_snapshot(session: Session, product_id: str, phase: str) -> dict[str, Any]:
    """Compute one phase, freeze it as an idempotent snapshot, return its header.

    与 POST /api/cta-attribution/snapshots 共用同一实现：HTTP 端点直接调用
    本函数，批量任务也在后台线程里调用本函数，两边不会漂移。评估失败
    （数据不足、合约不适用等）以 HTTPException 形式抛出，由调用方决定
    记入 failed 还是 skipped。
    """
    from fastapi import HTTPException

    from app.routers.cta_attribution import _phase_function
    from app.services.cta_attribution_evidence import build_attribution_evidence_package
    from app.services.cta_attribution_snapshot import build_snapshot_content

    product = product_store.get_product(session, product_id)
    if product is None:
        raise HTTPException(404, "产品不存在")
    phase_function = _phase_function(phase)
    if phase_function is None:  # Defensive guard for future route changes.
        raise HTTPException(422, "不支持的 CTA 归因阶段")
    try:
        result = phase_function(product_id, session)
    except HTTPException:
        raise
    observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
    if not observations:
        raise HTTPException(422, "没有可用于快照的已审核净值")
    evidence_package = build_attribution_evidence_package(
        session, phase=phase, observations=observations, result=result,
    )
    label, model_version, content = build_snapshot_content(
        phase=phase, product=product, observations=observations, result=result,
        evidence_package=evidence_package,
    )
    existing = session.execute(
        select(DataSnapshot).where(DataSnapshot.label == label).limit(1)
    ).scalars().first()
    snapshot = existing or product_store.create_snapshot(session, label=label, content=content)
    frozen = snapshot.content
    return {
        "snapshot_id": snapshot.id,
        "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
        "snapshot_type": "cta_dynamic_attribution",
        "phase": phase,
        "model_version": model_version,
        "product_id": frozen.get("product_id"),
        "as_of_date": frozen.get("as_of_date"),
        "nav_fingerprint": frozen.get("nav_fingerprint"),
        "factor_data_version": frozen.get("factor_data_version"),
        "evidence_package": frozen.get("evidence_package", {}),
        "result": frozen.get("results", {}),
        "idempotent": existing is not None,
    }


class PhaseDBatchRunner:
    """进程级单例：有界并发 + 活跃集合 + 队列快照 + 结果账本。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._running = False
        self._concurrency = PHASE_D_CONCURRENCY
        self._queue: list[str] = []
        self._active: set[str] = set()
        self._finished: dict[str, dict[str, Any]] = {}
        self._started_at: str | None = None
        self._total = 0

    # ------------------------------------------------------------------
    # 对外状态
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            # 排队数按「总量 − 已完成 − 处理中」实时折算：_queue 只在任务
            # 结束时才清空，直接用 len(_queue) 会让排队数恒等于总数。
            queued = self._total - len(self._finished) - len(self._active)
            return {
                "status": "running" if self._running else "idle",
                "concurrency": self._concurrency,
                "total": self._total,
                "queued": max(queued, 0),
                "active": sorted(self._active),
                "finished": len(self._finished),
                "ok": sum(1 for item in self._finished.values() if item.get("status") == "ok"),
                "reused": sum(1 for item in self._finished.values() if item.get("idempotent")),
                "skipped": sum(1 for item in self._finished.values() if item.get("status") == "skipped"),
                "failed": sum(1 for item in self._finished.values() if item.get("status") == "error"),
                "started_at": self._started_at,
            }

    def start(self, product_ids: list[str], concurrency: int = PHASE_D_CONCURRENCY) -> dict[str, Any]:
        with self._lock:
            if self._running:
                return {**self.status(), "note": "批量任务已在运行，忽略本次触发。"}
            self._running = True
            self._stop.clear()
            self._concurrency = min(max(int(concurrency), 1), PHASE_D_MAX_CONCURRENCY)
            self._queue = list(product_ids)
            self._active = set()
            self._finished = {}
            self._total = len(self._queue)
            self._started_at = datetime.now(timezone.utc).isoformat()
        threading.Thread(target=self._work, daemon=True, name="phase-d-batch").start()
        return self.status()

    # ------------------------------------------------------------------
    # 后台执行
    # ------------------------------------------------------------------

    def _work(self) -> None:
        tasks: queue.Queue[str] = queue.Queue()
        for product_id in self._queue:
            tasks.put(product_id)

        def worker() -> None:
            while not self._stop.is_set():
                try:
                    product_id = tasks.get_nowait()
                except queue.Empty:
                    return
                with self._lock:
                    self._active.add(product_id)
                try:
                    outcome = self._freeze_one(product_id)
                except Exception as error:  # noqa: BLE001 - 单只失败不拖垮整批
                    outcome = {"status": "error", "error": str(error)}
                finally:
                    with self._lock:
                        self._active.discard(product_id)
                        self._finished[product_id] = outcome
                    tasks.task_done()

        threads = [
            threading.Thread(target=worker, daemon=True, name=f"phase-d-worker-{index}")
            for index in range(self._concurrency)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        with self._lock:
            self._running = False
            self._queue = []

    def _freeze_one(self, product_id: str) -> dict[str, Any]:
        from fastapi import HTTPException

        session: Session = SessionFactory()
        try:
            # 幂等预检：净值指纹未变且已冻结过的产品直接复用，不做重复评估。
            # 评估（含非线性增量）单只约 0.5s，749 只全量重算没有意义。
            observations = product_store.get_nav_series(session, product_id, reviewed_only=True)
            if len(observations) < 20:
                return {"status": "skipped", "reason": "已审核净值不足 20 条"}
            current_fingerprint = nav_fingerprint(product_id, observations)
            existing = session.execute(
                select(DataSnapshot.id).where(
                    DataSnapshot.label.like(f"cta-attribution:phase-d:{product_id}:{current_fingerprint}:%")
                ).limit(1)
            ).scalars().first()
            if existing is not None:
                return {
                    "status": "ok",
                    "snapshot_id": existing,
                    "idempotent": True,
                    "reused": True,
                }
            frozen = freeze_attribution_snapshot(session, product_id, "phase-d")
        except HTTPException as error:
            return {"status": "skipped", "reason": str(error.detail)}
        finally:
            session.close()
        return {
            "status": "ok",
            "snapshot_id": frozen["snapshot_id"],
            "as_of_date": frozen["as_of_date"],
            "idempotent": bool(frozen.get("idempotent")),
            "reused": False,
        }


_RUNNER: PhaseDBatchRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_batch_runner() -> PhaseDBatchRunner:
    """返回进程级单例（懒初始化，避免导入期创建线程）。"""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = PhaseDBatchRunner()
        return _RUNNER


def list_phase_d_batch_candidates(session: Session) -> list[str]:
    """列出需要生成 Phase-D 证据的周频确认产品（有 ≥20 条已审核净值）。

    已冻结快照且净值指纹未变的产品不会重复计算——但「指纹比对」依赖
    批量任务的幂等标签，这里只做粗筛（confirmed + weekly），细筛由
    freeze 的幂等去重完成。
    """
    products = session.execute(
        select(ProductEntity).where(
            ProductEntity.confirmation_status == ConfirmationStatus.CONFIRMED,
            ProductEntity.nav_frequency == "weekly",
        )
    ).scalars().all()
    return [product.id for product in products]
