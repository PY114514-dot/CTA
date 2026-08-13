"""Safe access to files persisted under the upload directory."""

from pathlib import Path

from app.config import UPLOAD_DIRECTORY


def resolve_upload_path(storage_path: str | None) -> Path | None:
    """Resolve a database storage name without allowing escape from uploads."""
    if not storage_path:
        return None
    root = UPLOAD_DIRECTORY.resolve()
    candidate = (root / storage_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("stored upload path escapes the upload directory")
    return candidate
