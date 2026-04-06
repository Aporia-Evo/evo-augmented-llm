#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/neat-xor-spacetimedb-v1}"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR/src"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run_id> [extra run-online args...]"
  exit 1
fi

RUN_ID="$1"
shift

python -m main run-online \
  --config "$ROOT_DIR/configs/base.yaml" \
  --config "$ROOT_DIR/configs/online.yaml" \
  --config "$ROOT_DIR/configs/event_memory.yaml" \
  --config "$ROOT_DIR/configs/local.yaml" \
  --resume-run-id "$RUN_ID" \
  "$@"
