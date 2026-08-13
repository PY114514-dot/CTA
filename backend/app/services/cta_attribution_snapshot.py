"""Reproducibility metadata for immutable CTA attribution snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

import numpy as np

from app.services.factor_library import cache


MODEL_VERSIONS = {
    "phase-a": "cta-attribution-phase-a-v1.1",
    "phase-b": "cta-attribution-phase-b-v1.1",
    "phase-c": "cta-attribution-phase-c-v1.1",
    "phase-d": "cta-attribution-phase-d-v1.0",
}


def json_safe(value: Any) -> Any:
    """Convert numerical/date values to the JSON types used by SQLite JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _digest(value: Any, length: int = 16) -> str:
    canonical = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def nav_fingerprint(product_id: str, observations: list[Any]) -> str:
    """Fingerprint the exact reviewed observations used by an attribution run."""
    payload = {
        "product_id": product_id,
        "observations": [
            {
                "id": item.id,
                "date": item.observation_date.isoformat(),
                "nav": float(item.nav),
                "frequency": item.frequency,
                "review_status": str(item.review_status),
            }
            for item in observations
        ],
    }
    return _digest(payload)


def factor_data_provenance(factor_names: list[str] | None = None) -> dict[str, Any]:
    """Return stable baseline-cache metadata, excluding mutable build timestamps."""
    manifest = cache.list_cached_factors()
    names = sorted(set(factor_names or manifest.keys()))
    factors: list[dict[str, Any]] = []
    for name in names:
        entry = manifest.get(name)
        if entry is None:
            factors.append({"name": name, "status": "missing"})
            continue
        factors.append({
            "name": name,
            "status": "cached",
            "active_return_version": entry.get("active_return_version"),
            "rows": entry.get("rows"),
            "start": entry.get("start"),
            "end": entry.get("end"),
            "params": entry.get("params", {}),
            "data_version": entry.get("data_version", {}),
        })
    version = _digest({"profile": "baseline", "factors": factors})
    return {"profile": "baseline", "version": version, "factors": factors}


def factor_names_from_result(phase: str, result: dict[str, Any]) -> list[str]:
    if phase == "phase-a":
        return [str(item["factor_name"]) for item in result.get("attribution", {}).get("factor_exposure", [])]
    names = result.get("alignment", {}).get("factor_names")
    if isinstance(names, list):
        return [str(name) for name in names]
    names = result.get("nonlinear_increment", {}).get("factor_names")
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def build_snapshot_content(
    *,
    phase: str,
    product: Any,
    observations: list[Any],
    result: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Build deterministic label fields and the frozen attribution payload."""
    if phase not in MODEL_VERSIONS:
        raise ValueError(f"不支持的 CTA 归因阶段：{phase}")
    nav_version = nav_fingerprint(product.id, observations)
    factor_provenance = factor_data_provenance(factor_names_from_result(phase, result))
    model_version = MODEL_VERSIONS[phase]
    as_of_date = observations[-1].observation_date.isoformat()
    label = f"cta-attribution:{phase}:{product.id}:{nav_version}:{factor_provenance['version']}"
    content = {
        "snapshot_type": "cta_dynamic_attribution",
        "product_id": product.id,
        "product_name": product.standard_name,
        "phase": phase,
        "as_of_date": as_of_date,
        "nav_fingerprint": nav_version,
        "factor_data_version": factor_provenance["version"],
        "factor_data_provenance": factor_provenance,
        "model_version": model_version,
        "frequency": product.nav_frequency or observations[0].frequency or "weekly",
        "parameters": {
            "reviewed_nav_count": len(observations),
            "factor_names": factor_names_from_result(phase, result),
            "phase_parameters": result.get("regime_rules", {}).get("parameters", {}) if phase == "phase-c" else result.get("kalman_beta", {}).get("parameters", {}) if phase == "phase-b" else result.get("nonlinear_increment", {}).get("parameters", {}) if phase == "phase-d" else {},
        },
        "results": json_safe(result),
        "warnings": json_safe(result.get("warnings", [])),
    }
    return label, model_version, content
