"""Application-wide configuration with safe development defaults."""

import os
from collections.abc import Mapping
from pathlib import Path


# Storage is outside source modules so S3/MinIO can later replace local files.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
UPLOAD_DIRECTORY = DATA_DIRECTORY / "uploads"
EXTERNAL_FACTOR_LIBRARY_DIRECTORY = DATA_DIRECTORY / "external_factor_library"
FOF_LIBRARY_DIRECTORY = DATA_DIRECTORY / "fof_library"


def load_local_environment() -> None:
    """Load the ignored backend .env file without overriding real environment variables."""
    env_file = PROJECT_ROOT / "backend" / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def persist_local_environment(values: Mapping[str, str]) -> None:
    """Persist approved local settings without exposing them to version control."""
    if not values:
        return
    if any(not key or "=" in key or any(char in value for char in "\r\n") for key, value in values.items()):
        raise ValueError("本地配置包含无效的键或换行符")

    env_file = PROJECT_ROOT / "backend" / ".env"
    existing = env_file.read_text(encoding="utf-8").splitlines() if env_file.is_file() else []
    remaining = dict(values)
    updated: list[str] = []
    for line in existing:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in remaining:
            updated.append(f"{key}={remaining.pop(key)}")
        else:
            updated.append(line)
    updated.extend(f"{key}={value}" for key, value in remaining.items())
    env_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = env_file.with_suffix(".tmp")
    temporary_file.write_text("\n".join(updated) + "\n", encoding="utf-8")
    temporary_file.replace(env_file)


load_local_environment()

# Tesseract language data.  A project-local tessdata directory is preferred so
# the Simplified Chinese pack (chi_sim) is available without an admin install.
# However, Tesseract on Windows cannot read a TESSDATA_PREFIX whose path
# contains non-ASCII characters (this project's root path is Chinese), so the
# project copy is only used when its path is pure ASCII; otherwise we leave the
# variable unset and Tesseract falls back to its default install directory,
# which has chi_sim installed.  An explicitly set, non-empty TESSDATA_PREFIX is
# always respected; an empty value (common on Windows) is treated as unset.
_TESSDATA_DIRECTORY = PROJECT_ROOT / "backend" / "tessdata"


def _is_ascii_path(path: Path) -> bool:
    """Return True when *path* contains only ASCII characters."""
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


if not os.environ.get("TESSDATA_PREFIX"):
    if (
        (_TESSDATA_DIRECTORY / "chi_sim.traineddata").is_file()
        and _is_ascii_path(_TESSDATA_DIRECTORY)
    ):
        os.environ["TESSDATA_PREFIX"] = str(_TESSDATA_DIRECTORY)

# Production must replace this development list with approved web origins.
DEVELOPMENT_CORS_ORIGINS = [
    "http://localhost:5274", "http://127.0.0.1:5274",
    # Fallback development launcher when the default port is held by a stale
    # local process.  This is still loopback-only and is not a production
    # cross-origin allowance.
    "http://localhost:5275", "http://127.0.0.1:5275",
]

# PaddleOCR cloud credentials are deliberately environment-only.  Do not put
# access tokens in source files, request payloads, or client-visible settings.
PADDLEOCR_JOB_URL = os.getenv(
    "PADDLEOCR_JOB_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
)
PADDLEOCR_API_TOKEN = os.getenv("PADDLEOCR_API_TOKEN", "")
PADDLEOCR_MODEL = os.getenv("PADDLEOCR_MODEL", "PaddleOCR-VL-1.6")
PADDLEOCR_POLL_INTERVAL_SECONDS = float(os.getenv("PADDLEOCR_POLL_INTERVAL_SECONDS", "2"))
PADDLEOCR_TIMEOUT_SECONDS = float(os.getenv("PADDLEOCR_TIMEOUT_SECONDS", "60"))
PADDLEOCR_MAX_WAIT_SECONDS = float(os.getenv("PADDLEOCR_MAX_WAIT_SECONDS", "900"))
