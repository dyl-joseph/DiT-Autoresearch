#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q dit_autoresearch
python -m unittest discover -s tests -v
