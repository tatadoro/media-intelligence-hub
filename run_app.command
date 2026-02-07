#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
  echo "[ERROR] This UI bundle is only supported on arm64 (Apple Silicon) for now."
  exit 1
fi
LOCK_FILE="$ROOT_DIR/requirements-ui.${ARCH}.lock"
WHEELS_DIR="$ROOT_DIR/vendor/wheels/${ARCH}"

if [ ! -f "$LOCK_FILE" ]; then
  echo "[ERROR] Missing $LOCK_FILE"
  echo "Run: scripts/build_ui_wheels.sh (requires internet once)"
  exit 1
fi

if [ ! -d "$WHEELS_DIR" ]; then
  echo "[ERROR] Missing wheels directory: $WHEELS_DIR"
  echo "Run: scripts/build_ui_wheels.sh (requires internet once)"
  exit 1
fi

# Ensure build backends are present before installing sdists (e.g., sgmllib3k)
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT

grep -E '^(setuptools|wheel)==' "$LOCK_FILE" > "$TMP_REQ" || true
if [ -s "$TMP_REQ" ]; then
  python -m pip install --no-index --find-links "$WHEELS_DIR" --require-hashes --no-build-isolation -r "$TMP_REQ"
fi

python -m pip install --no-index --find-links "$WHEELS_DIR" --require-hashes --no-build-isolation -r "$LOCK_FILE"

# Streamlit uses ~/.streamlit for credentials; force HOME to a writable dir
export HOME="$ROOT_DIR/.home"
mkdir -p "$HOME/.streamlit"

CONFIG_FILE="$HOME/.streamlit/config.toml"
cat <<'EOF' > "$CONFIG_FILE"
[browser]
gatherUsageStats = false

[server]
showEmailPrompt = false
EOF

CRED_FILE="$HOME/.streamlit/credentials.toml"
if [ ! -f "$CRED_FILE" ]; then
  cat <<'EOF' > "$CRED_FILE"
[general]
email = ""
EOF
fi

python -m streamlit run ui/app.py
