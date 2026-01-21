#!/usr/bin/env bash
set -euo pipefail

CH_HOST="${CH_HOST:-localhost}"
CH_PORT="${CH_PORT:-18123}"
URL="http://${CH_HOST}:${CH_PORT}/ping"

echo "[INFO] Waiting for ClickHouse: $URL"
for i in {1..60}; do
  if curl -fsS "$URL" | grep -qi "ok"; then
    echo "[OK] ClickHouse is ready"
    exit 0
  fi
  sleep 1
done

echo "[ERROR] ClickHouse not ready after 60s: $URL"
exit 1
