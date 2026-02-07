#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
  echo "[ERROR] UI is supported only on arm64 (Apple Silicon)."
  exit 1
fi

LOCK_FILE="$ROOT_DIR/requirements-ui.${ARCH}.lock"
WHEELS_DIR="$ROOT_DIR/vendor/wheels/${ARCH}"
APP_SCRIPT="$ROOT_DIR/ui/app.py"
RUN_SCRIPT="$ROOT_DIR/run_app.command"

fail=0

if [ ! -f "$LOCK_FILE" ]; then
  echo "[ERROR] Missing $LOCK_FILE"
  fail=1
fi

if [ ! -d "$WHEELS_DIR" ]; then
  echo "[ERROR] Missing wheels directory: $WHEELS_DIR"
  fail=1
fi

if [ ! -f "$APP_SCRIPT" ]; then
  echo "[ERROR] Missing UI app: $APP_SCRIPT"
  fail=1
fi

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "[ERROR] Missing run script: $RUN_SCRIPT"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "[OK] UI prerequisites look good."
echo "Run: bash run_app.command"
