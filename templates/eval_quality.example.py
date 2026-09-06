"""Project-specific benchmark quality evaluator template.

Contract: write <output> as JSON matching templates/quality.example.json.
This is the ground-truth acceptance gate. Keep it fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Replace with the real benchmark suite. Use fixed prompts/seeds/schedules.
    # Quantized candidates must not reduce any required accuracy metric.
    result = {
        "metrics": {
            "benchmark_accuracy": 0.0
        },
        "cases": {}
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit("Implement project-specific quality benchmark before starting autoresearch")


if __name__ == "__main__":
    main()
