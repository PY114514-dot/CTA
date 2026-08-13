"""Reparse completed CTA image/PDF materials sequentially after a VLM upgrade."""

from __future__ import annotations

import json
import time
from collections import Counter
from urllib import request


API = "http://127.0.0.1:8103/api/kb"
VISUAL_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png", ".webp")


def list_files() -> list[dict]:
    result: list[dict] = []
    offset = 0
    while True:
        with request.urlopen(f"{API}/files?limit=100&offset={offset}", timeout=120) as response:
            page = json.loads(response.read().decode("utf-8"))
        result.extend(page)
        if len(page) < 100:
            return result
        offset += len(page)


def reparse(file_id: str) -> None:
    payload = request.Request(f"{API}/files/{file_id}/reparse", data=b"", method="POST")
    with request.urlopen(payload, timeout=120) as response:
        response.read()


def wait_until_idle() -> list[dict]:
    while True:
        records = list_files()
        if not any(item["parsing_status"] in {"pending", "processing"} for item in records):
            return records
        time.sleep(10)


def main() -> None:
    initial = list_files()
    targets = [
        item["id"] for item in initial
        if (
            item["parsing_status"] == "completed_no_nav"
            and item["filename"].lower().endswith(VISUAL_SUFFIXES)
            and item.get("reviewed_nav_count", 0) == 0
        )
    ]
    print(json.dumps({"vlm_reparse_targets": len(targets)}, ensure_ascii=False), flush=True)
    for index, file_id in enumerate(targets, 1):
        wait_until_idle()
        reparse(file_id)
        print(json.dumps({"submitted": index, "total": len(targets), "file_id": file_id}, ensure_ascii=False), flush=True)
    records = wait_until_idle()
    print(json.dumps({"completed": len(targets), "statuses": dict(Counter(item["parsing_status"] for item in records))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
