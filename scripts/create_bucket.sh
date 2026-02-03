#!/usr/bin/env bash
set -euo pipefail

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-admin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-}"

if [ -z "${MINIO_SECRET_KEY}" ]; then
  echo "[ERROR] MINIO_SECRET_KEY is not set. Export MINIO_SECRET_KEY (or fill .env) and retry." >&2
  exit 1
fi
MINIO_BUCKET="${MINIO_BUCKET:-mih}"

echo "[INFO] Ensuring MinIO bucket: ${MINIO_BUCKET} at ${MINIO_ENDPOINT}"

# 1) local mc
if command -v mc >/dev/null 2>&1; then
  mc alias set mih "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" >/dev/null
  mc mb -p "mih/${MINIO_BUCKET}" >/dev/null 2>&1 || true
  echo "[OK] Bucket ensured via local mc"
  exit 0
fi

# 2) mc inside a running minio container (best-effort)
MINIO_CONTAINER="$(docker ps --format '{{.Names}}' | grep -E 'minio' | head -n 1 || true)"
if [[ -n "${MINIO_CONTAINER}" ]]; then
  docker exec "${MINIO_CONTAINER}" mc alias set mih "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" >/dev/null
  docker exec "${MINIO_CONTAINER}" mc mb -p "mih/${MINIO_BUCKET}" >/dev/null 2>&1 || true
  echo "[OK] Bucket ensured via docker exec (${MINIO_CONTAINER}) mc"
  exit 0
fi

echo "[WARN] 'mc' not found locally and MinIO container with name like '*minio*' not detected."
echo "[WARN] Install MinIO client 'mc' or adjust MINIO_CONTAINER detection."
exit 1
