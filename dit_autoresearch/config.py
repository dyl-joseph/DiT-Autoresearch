from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    root: Path
    raw: dict[str, Any]
    source_path: Path

    @property
    def project_name(self) -> str:
        return str(self.raw.get("project_name", "dit-project"))

    @property
    def state_dir(self) -> Path:
        return self.root / self.raw.get("state_dir", ".autoresearch")

    @property
    def results_tsv(self) -> Path:
        return self.root / self.raw.get("results_tsv", ".autoresearch/results.tsv")

    @property
    def mutable_paths(self) -> list[str]:
        return list(self.raw.get("mutable_paths", []))

    @property
    def fixed_paths(self) -> list[str]:
        return list(self.raw.get("fixed_paths", []))


    @property
    def integrity_paths(self) -> list[str]:
        paths = list(self.fixed_paths)
        for p in (self.source_path, self.quality_contract_path):
            try:
                rel = str(p.resolve().relative_to(self.root)).replace("\\", "/")
            except ValueError:
                rel = str(p.resolve())
            if rel not in paths:
                paths.append(rel)
        return paths

    @property
    def commands(self) -> dict[str, str]:
        return dict(self.raw.get("commands", {}))

    @property
    def timeouts(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.raw.get("timeouts_seconds", {}).items()}

    @property
    def objective(self) -> dict[str, Any]:
        return dict(self.raw.get("objective", {}))

    @property
    def guardrails(self) -> dict[str, Any]:
        return dict(self.raw.get("performance_guardrails", {}))

    @property
    def target(self) -> dict[str, Any]:
        return dict(self.raw.get("target", {}))

    @property
    def quantization(self) -> dict[str, Any]:
        return dict(self.raw.get("quantization", {}))

    @property
    def quality_contract_path(self) -> Path:
        p = self.raw.get("quality_contract", "config/quality_contract.json")
        return self.root / p

    @property
    def knowledge_dir(self) -> Path:
        p = self.raw.get("knowledge_dir", "knowledge/handbook")
        return self.root / p


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"Config does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a JSON object")
    root = path.parent.parent if path.parent.name == "config" else path.parent
    root = Path(raw.get("project_root", root)).resolve()
    cfg = Config(root=root, raw=raw, source_path=path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    errors: list[str] = []
    if not cfg.mutable_paths:
        errors.append("mutable_paths must contain at least one path")
    for required in ("perf", "quality"):
        if required not in cfg.commands or not str(cfg.commands[required]).strip():
            errors.append(f"commands.{required} is required")
    metric = cfg.objective.get("metric")
    direction = cfg.objective.get("direction")
    if not metric:
        errors.append("objective.metric is required")
    if direction not in ("lower", "higher"):
        errors.append("objective.direction must be 'lower' or 'higher'")
    if not cfg.quality_contract_path.exists():
        errors.append(f"quality contract not found: {cfg.quality_contract_path}")
    if errors:
        raise ConfigError("Invalid config:\n- " + "\n- ".join(errors))


def load_quality_contract(cfg: Config) -> dict[str, Any]:
    raw = json.loads(cfg.quality_contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("quality contract root must be a JSON object")
    return raw
