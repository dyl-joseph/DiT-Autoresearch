from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _cmd(args: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": repr(exc)}


def capture_environment() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi_present": bool(shutil.which("nvidia-smi")),
    }
    if shutil.which("nvidia-smi"):
        out["nvidia_smi_query"] = _cmd([
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem",
            "--format=csv",
        ])
        out["nvidia_topology"] = _cmd(["nvidia-smi", "topo", "-m"])
    try:
        import torch  # type: ignore

        gpu: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            gpu["device_count"] = torch.cuda.device_count()
            gpu["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                })
        out["torch"] = gpu
    except Exception as exc:
        out["torch"] = {"error": repr(exc)}
    return out


def write_environment(path: Path) -> dict[str, Any]:
    obj = capture_environment()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return obj


def detect_a5500(env: dict[str, Any]) -> bool:
    text = json.dumps(env).lower()
    return "a5500" in text
