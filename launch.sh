#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x .venv/bin/python ]; then
  echo "Run ./setup.sh once to install the free local dependencies."
  exit 1
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPL_IGNORE_SYSTEM_FONTS=1
exec .venv/bin/python -m vibescope.cli serve "$@"
