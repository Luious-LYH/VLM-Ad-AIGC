"""Read the lightweight v0.1 model registry without importing ML packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path).resolve(strict=True)
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "model_registry/v1":
        raise ValueError(f"invalid model registry: {registry_path}")
    return value


def get_model(registry: dict[str, Any], role: str, model_key: str) -> dict[str, Any]:
    models = registry.get("models", {}).get(role, {})
    if model_key not in models:
        raise KeyError(f"unknown {role} model: {model_key}")
    return models[model_key]
