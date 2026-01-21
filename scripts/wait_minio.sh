#!/usr/bin/env bash
set -euo pipefail

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
URL="${MINIO_ENDPOINT%/}/minio/health/ready"

echo "[INFO] Waiting for MinIO: $URL"
for i in {1..60}; do
  if curl -fsS "$URL" >/dev/null; then
    echo "[OK] MinIO is ready"
    exit 0
  fi
  sleep 1
done

echo "[ERROR] MinIO not ready after 60s: $URL"
exit 1
