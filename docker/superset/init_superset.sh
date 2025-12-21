#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Checking superset binary..."
which superset || { echo "[ERROR] superset not found in PATH"; exit 1; }
superset --version || true

echo "[INFO] Upgrading metadata DB..."
superset db upgrade

echo "[INFO] Creating admin (best-effort)..."
superset fab create-admin \
  --username admin \
  --firstname Superset \
  --lastname Admin \
  --email admin@example.com \
  --password admin || true

echo "[INFO] Initializing Superset..."
superset init

echo "[INFO] Starting Superset webserver..."
superset run -h 0.0.0.0 -p 8088
