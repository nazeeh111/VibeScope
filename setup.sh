#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
echo 'Ready. Run ./launch.sh, then open http://127.0.0.1:8765'
