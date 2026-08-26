"""Upload persistence must never resolve outside its configured root."""

from pathlib import Path

import pytest

from app.config import UPLOAD_DIRECTORY
from app.services.upload_storage import resolve_upload_path


def test_resolve_upload_path_accepts_child_name() -> None:
    assert resolve_upload_path("source.pdf") == (UPLOAD_DIRECTORY.resolve() / "source.pdf")


@pytest.mark.parametrize("value", ["../secret.txt", "..\\secret.txt", str(Path(UPLOAD_DIRECTORY).anchor + "secret.txt")])
def test_resolve_upload_path_rejects_escape(value: str) -> None:
    with pytest.raises(ValueError):
        resolve_upload_path(value)
