"""The upload workflow must expose its async parse capacity and bound it."""

import threading
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import RawFile
from app.routers.knowledge_base.files import get_ingestion_queue
from app.services.material_ingestion import entry, ingestion_queue_snapshot


def _memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_snapshot_reports_capacity_and_active_files() -> None:
    snapshot = ingestion_queue_snapshot()
    assert snapshot["max_concurrency"] >= 1
    assert snapshot["active_count"] == len(snapshot["active_file_ids"])


def test_semaphore_bounds_concurrent_ingestion() -> None:
    """A batch of uploads must not fan out unbounded VLM/OCR work at once."""
    original_semaphore = entry._INGEST_SEMAPHORE
    original_impl = entry._ingest_uploaded_material_impl
    entry._INGEST_SEMAPHORE = threading.BoundedSemaphore(2)
    entry._INGEST_ACTIVE_FILE_IDS.clear()
    observed: dict[str, int] = {"max_active": 0, "calls": 0}
    lock = threading.Lock()

    def slow_impl(file_id: str, *_args, **_kwargs) -> None:
        with lock:
            observed["max_active"] = max(observed["max_active"], len(entry._INGEST_ACTIVE_FILE_IDS))
            observed["calls"] += 1
        time.sleep(0.1)

    entry._ingest_uploaded_material_impl = slow_impl
    try:
        threads = [
            threading.Thread(
                target=entry.ingest_uploaded_material,
                args=(f"f{index}", b"", f"f{index}.pdf", "application/pdf"),
            )
            for index in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        entry._ingest_uploaded_material_impl = original_impl
        entry._INGEST_SEMAPHORE = original_semaphore
        entry._INGEST_ACTIVE_FILE_IDS.clear()

    assert observed["calls"] == 6
    assert observed["max_active"] <= 2


def test_queue_endpoint_counts_queued_versus_active() -> None:
    session = _memory_session()
    for index in range(3):
        session.add(RawFile(file_hash=f"{index:064d}", filename=f"p{index}.png", parsing_status="processing"))
    session.commit()

    response = get_ingestion_queue(session)

    assert response["processing_count"] == 3
    assert response["queued_count"] == 3 - response["active_count"]
    assert response["max_concurrency"] >= 1
