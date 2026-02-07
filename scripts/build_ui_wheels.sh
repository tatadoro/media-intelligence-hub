#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

IN_FILE="$ROOT_DIR/requirements-ui.txt"
ARCH="$(uname -m)"

if [ "$ARCH" != "arm64" ]; then
  echo "[ERROR] This UI bundle is only supported on arm64 (Apple Silicon) for now."
  exit 1
fi

WHEELS_DIR="$ROOT_DIR/vendor/wheels/${ARCH}"
LOCK_FILE="$ROOT_DIR/requirements-ui.${ARCH}.lock"

mkdir -p "$WHEELS_DIR"

# Clean old artifacts to avoid version conflicts in lockfile
rm -f "$WHEELS_DIR"/*.whl "$WHEELS_DIR"/*.tar.gz "$WHEELS_DIR"/*.zip 2>/dev/null || true

python3 -m pip download -r "$IN_FILE" \
  --dest "$WHEELS_DIR" \
  --prefer-binary

python3 "$ROOT_DIR/scripts/build_ui_lock.py" \
  --wheels-dir "$WHEELS_DIR" \
  --out "$LOCK_FILE"

echo "[OK] Wheels: $WHEELS_DIR"
echo "[OK] Lock:   $LOCK_FILE"
