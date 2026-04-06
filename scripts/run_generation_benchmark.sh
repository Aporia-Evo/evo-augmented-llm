#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/neat-xor-spacetimedb-v1}"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR/src"

python -m main benchmark-suite \
  --store memory \
  --tasks delayed_xor,bit_memory \
  --seeds 7,11,13,17,19 \
  --output-dir "$ROOT_DIR/results" \
  "$@"
