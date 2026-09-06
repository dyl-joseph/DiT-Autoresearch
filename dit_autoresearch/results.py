from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEADER = [
    "timestamp_utc",
    "commit",
    "parent",
    "status",
    "class",
    "objective_metric",
    "best_before",
    "candidate",
    "improvement_pct",
    "peak_vram_mb",
    "quality",
    "description",
    "run_dir",
]


def ensure_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(HEADER)


def append_result(path: Path, row: dict[str, Any]) -> None:
    ensure_results(path)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t", extrasaction="ignore")
        payload = {k: row.get(k, "") for k in HEADER}
        payload["timestamp_utc"] = row.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
        writer.writerow(payload)


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
