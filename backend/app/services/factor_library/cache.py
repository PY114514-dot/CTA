"""Local CSV cache for computed factor return series.

Factor computation is expensive (fetching data for 18 varieties, computing
signals, constructing portfolios).  Once built, results are cached as CSV
files under `data/factors/` so subsequent requests can load instantly.

Cache key: factor_name + date_range.  A manifest JSON tracks what has been
built and when.
"""

import json
import logging
import hashlib
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from app.config import DATA_DIRECTORY

logger = logging.getLogger(__name__)

FACTORS_DIRECTORY = DATA_DIRECTORY / "factors"
MANIFEST_PATH = FACTORS_DIRECTORY / "_manifest.json"


def _ensure_dir() -> None:
    FACTORS_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _return_version_id(
    factor_name: str,
    returns: pd.Series,
    params: dict | None,
    data_version: dict | None,
) -> str:
    """Fingerprint the exact active-return input, not merely its date range."""
    canonical = pd.DataFrame({"date": returns.index, "return": returns.values}).to_csv(index=False)
    provenance = json.dumps(
        {"params": params or {}, "data_version": data_version or {}},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256((factor_name + canonical + provenance).encode("utf-8")).hexdigest()[:12]


def _cache_filename(factor_name: str, start: date, end: date, version_id: str) -> str:
    return f"{factor_name}_return_{version_id}_{start.isoformat()}_{end.isoformat()}.csv"


def _load_manifest() -> dict:
    if MANIFEST_PATH.is_file():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_manifest(manifest: dict) -> None:
    _ensure_dir()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_factor_returns(
    factor_name: str,
    returns: pd.Series,
    start: date,
    end: date,
    params: dict | None = None,
    data_version: dict | None = None,
) -> Path:
    """Persist a factor return series to CSV and update the manifest.

    Returns the path to the written CSV file.
    """
    _ensure_dir()
    version_id = _return_version_id(factor_name, returns, params, data_version)
    filename = _cache_filename(factor_name, start, end, version_id)
    filepath = FACTORS_DIRECTORY / filename

    # Write CSV: date, return
    out = pd.DataFrame({"date": returns.index, "return": returns.values})
    out.to_csv(filepath, index=False, encoding="utf-8")

    # Update manifest
    manifest = _load_manifest()
    # Keep immutable artifacts and validations collected by earlier builds.
    # The active CSV is intentionally replaceable, but it must never erase the
    # research evidence that explains an older result.
    existing = manifest.get(factor_name, {})
    manifest[factor_name] = {
        "file": filename,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": len(returns),
        "params": params or {},
        "data_version": data_version or {},
        "active_return_version": version_id,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    for preserved_key in ("artifacts", "validations", "contrib_file", "return_versions"):
        if preserved_key in existing:
            manifest[factor_name][preserved_key] = existing[preserved_key]
    version_record = {
        "id": version_id,
        "file": filename,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": len(returns),
        "params": params or {},
        "data_version": data_version or {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    versions = manifest[factor_name].setdefault("return_versions", [])
    if not any(item.get("id") == version_id for item in versions):
        versions.append(version_record)
    _save_manifest(manifest)

    logger.info("Cached factor '%s': %d rows -> %s", factor_name, len(returns), filepath)
    return filepath


def save_factor_artifact(
    factor_name: str,
    artifact_kind: str,
    values: pd.Series,
    start: date,
    end: date,
    *,
    metadata: dict | None = None,
) -> dict:
    """Append an immutable, dated factor artifact to the manifest.

    The normal ``return`` cache remains the active convenience view.  This
    audit store prevents a later rebuild from silently erasing the exact
    baseline / overlay series used in a research result.
    """
    _ensure_dir()
    canonical = values.dropna().astype(float)
    digest = hashlib.sha256(
        (factor_name + artifact_kind + canonical.to_csv()).encode("utf-8")
    ).hexdigest()[:12]
    artifact_id = f"{artifact_kind}_{digest}"
    filename = f"{factor_name}_{artifact_id}_{start.isoformat()}_{end.isoformat()}.csv"
    filepath = FACTORS_DIRECTORY / filename
    pd.DataFrame({"date": canonical.index, "value": canonical.values}).to_csv(filepath, index=False, encoding="utf-8")
    manifest = _load_manifest()
    entry = manifest.setdefault(factor_name, {})
    artifacts = entry.setdefault("artifacts", [])
    record = {
        "id": artifact_id, "kind": artifact_kind, "file": filename,
        "start": start.isoformat(), "end": end.isoformat(), "rows": len(canonical),
        "created_at": datetime.now().isoformat(timespec="seconds"), "metadata": metadata or {},
    }
    if not any(item.get("id") == artifact_id for item in artifacts):
        artifacts.append(record)
        _save_manifest(manifest)
    return record


def load_factor_returns(
    factor_name: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.Series | None:
    """Load cached factor returns.  Returns None if no cache exists.

    If start/end are provided and the cache covers a superset, the series
    is sliced to the requested range.
    """
    manifest = _load_manifest()
    entry = manifest.get(factor_name)
    if entry is None:
        return None

    filepath = FACTORS_DIRECTORY / entry["file"]
    if not filepath.is_file():
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        series = df.set_index("date")["return"]
        series.name = factor_name

        if start:
            series = series[series.index >= pd.Timestamp(start)]
        if end:
            series = series[series.index <= pd.Timestamp(end)]

        return series
    except Exception as exc:
        logger.warning("Failed to load cached factor '%s': %s", factor_name, exc)
        return None


def save_factor_contributions(
    factor_name: str,
    contributions: pd.DataFrame,
    start: date,
    end: date,
) -> Path:
    """Persist the per-variety contribution matrix to CSV and update manifest.

    The DataFrame has DatetimeIndex and one column per variety.  The manifest
    entry for *factor_name* gains a ``contrib_file`` key.
    """
    _ensure_dir()
    filename = f"{factor_name}_contrib_{start.isoformat()}_{end.isoformat()}.csv"
    filepath = FACTORS_DIRECTORY / filename

    out = contributions.copy()
    out.index.name = "date"
    out.to_csv(filepath, encoding="utf-8")

    # Patch manifest entry (must already exist from save_factor_returns)
    manifest = _load_manifest()
    if factor_name in manifest:
        manifest[factor_name]["contrib_file"] = filename
        _save_manifest(manifest)

    logger.info("Cached contributions '%s': %s -> %s", factor_name, contributions.shape, filepath)
    return filepath


def load_factor_contributions(
    factor_name: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame | None:
    """Load cached contribution matrix.  Returns None if unavailable."""
    manifest = _load_manifest()
    entry = manifest.get(factor_name)
    if entry is None or "contrib_file" not in entry:
        return None

    filepath = FACTORS_DIRECTORY / entry["contrib_file"]
    if not filepath.is_file():
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df
    except Exception as exc:
        logger.warning("Failed to load contributions '%s': %s", factor_name, exc)
        return None


def get_cache_info(factor_name: str) -> dict | None:
    """Return manifest entry for a factor, or None if not cached."""
    return _load_manifest().get(factor_name)


def list_cached_factors() -> dict:
    """Return the full manifest (all cached factors)."""
    return _load_manifest()


def clear_cache(factor_name: str | None = None) -> None:
    """Remove cached factor data.  If factor_name is None, clear all."""
    manifest = _load_manifest()

    def _unlink_entry(entry: dict) -> None:
        filenames = [entry.get(key) for key in ("file", "contrib_file")]
        filenames.extend(item.get("file") for item in entry.get("return_versions", []))
        filenames.extend(item.get("file") for item in entry.get("artifacts", []))
        for filename in set(filename for filename in filenames if filename):
            fp = FACTORS_DIRECTORY / filename
            if fp.is_file():
                fp.unlink()

    if factor_name:
        entry = manifest.pop(factor_name, None)
        if entry:
            _unlink_entry(entry)
    else:
        for entry in manifest.values():
            _unlink_entry(entry)
        manifest = {}

    _save_manifest(manifest)
