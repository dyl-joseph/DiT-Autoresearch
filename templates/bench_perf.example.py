"""Project-specific performance harness template.

Contract: write <output> as JSON matching templates/perf.example.json.
Keep this harness fixed during a research session.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    idx = min(len(ys) - 1, round((len(ys) - 1) * p))
    return ys[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Replace run_once with a fixed production-representative benchmark case.
    # Synchronize CUDA around measured regions in the real implementation.
    def run_once() -> None:
        raise SystemExit("Implement project-specific performance benchmark")

    warmup, iters = 5, 20
    for _ in range(warmup):
        run_once()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        samples.append(time.perf_counter() - t0)

    result = {
        "latency": {
            "median_s": statistics.median(samples),
            "p95_s": percentile(samples, 0.95),
            "mean_s": statistics.mean(samples),
        },
        "throughput_items_s": 1.0 / statistics.mean(samples),
        "peak_vram_mb": 0.0,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
