#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
DATABASE_NAME="${1:-neat-xor-v1}"
SPACETIME_PID=""

cleanup() {
  if [[ -n "$SPACETIME_PID" ]]; then
    kill "$SPACETIME_PID" >/dev/null 2>&1 || true
    wait "$SPACETIME_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

if ! command -v spacetime >/dev/null 2>&1; then
  echo "spacetime CLI not found in PATH"
  exit 1
fi

spacetime start >/tmp/neat-xor-spacetimedb-reset.log 2>&1 &
SPACETIME_PID=$!
sleep 5

cd "$ROOT_DIR/spacetimedb"
spacetime publish --server local --delete-data "$DATABASE_NAME"
