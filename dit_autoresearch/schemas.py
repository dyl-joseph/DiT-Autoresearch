from __future__ import annotations

from typing import Any


def get_metric(obj: dict[str, Any], path: str) -> float:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        raise TypeError(f"Metric {path!r} must be numeric; got {type(cur).__name__}")
    return float(cur)


def flatten_numeric(obj: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[path] = float(value)
        elif isinstance(value, dict):
            out.update(flatten_numeric(value, path))
    return out
