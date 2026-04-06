#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/neat-xor-spacetimedb-v1}"
SPACETIME_PID=""

cleanup() {
  if [[ -n "$SPACETIME_PID" ]]; then
    kill "$SPACETIME_PID" >/dev/null 2>&1 || true
    wait "$SPACETIME_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

STORE="memory"
DATABASE_NAME="neat-xor-v1"
USE_GPU="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --store)
      STORE="$2"
      shift 2
      ;;
    --database-name)
      DATABASE_NAME="$2"
      shift 2
      ;;
    --gpu)
      USE_GPU="true"
      shift
      ;;
    *)
      break
      ;;
  esac
done

if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "python3 -m pip is required. Install pip first."
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m pip install --user --break-system-packages virtualenv
  mkdir -p "$(dirname "$VENV_DIR")"
  "$HOME/.local/bin/virtualenv" "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -r "$ROOT_DIR/requirements.txt"
if [[ "$USE_GPU" == "true" ]]; then
  python -m pip install -r "$ROOT_DIR/requirements-gpu.txt"
fi

if [[ "$STORE" == "spacetimedb" ]]; then
  if ! command -v npm >/dev/null 2>&1 && [[ -s "$HOME/.nvm/nvm.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.nvm/nvm.sh"
    nvm use --lts >/dev/null
  fi
  if ! command -v spacetime >/dev/null 2>&1; then
    echo "spacetime CLI not found in PATH"
    exit 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm not found in PATH. Install Node.js in WSL first."
    exit 1
  fi
  if [[ ! -d "$ROOT_DIR/spacetimedb/node_modules" ]]; then
    (
      cd "$ROOT_DIR/spacetimedb"
      npm install --no-bin-links
    )
  fi
  spacetime start >/tmp/neat-xor-spacetimedb.log 2>&1 &
  SPACETIME_PID=$!
  sleep 5
  if [[ ! -f "$HOME/.config/spacetime/cli.toml" ]]; then
    spacetime login --server-issued-login http://127.0.0.1:3000 --no-browser >/dev/null
  fi
  (
    cd "$ROOT_DIR/spacetimedb"
    spacetime publish --server local "$DATABASE_NAME"
  )
fi

export PYTHONPATH="$ROOT_DIR/src"
python -m main run --config "$ROOT_DIR/configs/base.yaml" --config "$ROOT_DIR/configs/xor.yaml" --config "$ROOT_DIR/configs/local.yaml" --store "$STORE" --database-name "$DATABASE_NAME" "$@"
