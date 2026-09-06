from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .results import read_json


@dataclass
class CommandResult:
    command: str
    returncode: int
    elapsed_s: float
    log_path: Path
    timed_out: bool = False


def format_command(template: str, *, root: Path, run_dir: Path) -> str:
    return template.format(root=str(root), run_dir=str(run_dir))


def run_command(
    command: str,
    *,
    cwd: Path,
    log_path: Path,
    timeout_s: int,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    t0 = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            p = subprocess.run(
                command,
                cwd=cwd,
                shell=True,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                env=env,
            )
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = 124
            timed_out = True
            log.write(f"\n[AUTORESEARCH] TIMEOUT after {timeout_s}s\n")
    return CommandResult(command, code, time.perf_counter() - t0, log_path, timed_out)


def require_metrics(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{label} command did not create required file: {path}")
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise RuntimeError(f"{label} metrics must be a JSON object: {path}")
    return obj
