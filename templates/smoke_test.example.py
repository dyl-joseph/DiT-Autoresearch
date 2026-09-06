"""Copy into the target repo and replace imports with your pipeline.

This file belongs in fixed_paths after the baseline is established.
"""
from __future__ import annotations


def main() -> None:
    # Load one tiny/representative case and verify the optimized path returns
    # the expected shape/type without NaN/Inf. Do not turn this into the quality
    # benchmark; it is only an early crash detector.
    raise SystemExit("Implement project-specific smoke test before starting autoresearch")


if __name__ == "__main__":
    main()
