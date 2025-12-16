#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/upload_report_to_minio.sh <report_md_path> [last_hours]
#
# Requires:
#   mc configured (default alias: local)
#   MINIO_BUCKET env (or uses media-intel)
#   optional: GIT_COMMIT env

REPORT_PATH="${1:?report md path is required}"
LAST_HOURS="${2:-}"
MINIO_ALIAS="${MINIO_ALIAS:-local}"
MINIO_BUCKET="${MINIO_BUCKET:-media-intel}"

if [[ ! -f "$REPORT_PATH" ]]; then
  echo "[ERROR] Report file not found: $REPORT_PATH"
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
DT="$(date +%Y-%m-%d)"
BASE="reports/dt=${DT}/run_id=${RUN_ID}"

REPORT_KEY="${BASE}/daily_report.md"
META_LOCAL="$(mktemp -t mih_report_meta.XXXXXX.json)"

GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo "")}"

cat > "$META_LOCAL" <<JSON
{
  "run_id": "${RUN_ID}",
  "dt": "${DT}",
  "report_local_path": "${REPORT_PATH}",
  "report_object_key": "${REPORT_KEY}",
  "last_hours": "${LAST_HOURS}",
  "git_commit": "${GIT_COMMIT}",
  "uploaded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "[INFO] Uploading report to MinIO: ${MINIO_ALIAS}/${MINIO_BUCKET}/${REPORT_KEY}"
mc cp "$REPORT_PATH" "${MINIO_ALIAS}/${MINIO_BUCKET}/${REPORT_KEY}"

echo "[INFO] Uploading metadata to MinIO: ${MINIO_ALIAS}/${MINIO_BUCKET}/${BASE}/metadata.json"
mc cp "$META_LOCAL" "${MINIO_ALIAS}/${MINIO_BUCKET}/${BASE}/metadata.json"

echo "[OK] Uploaded:"
echo "  - ${MINIO_ALIAS}/${MINIO_BUCKET}/${REPORT_KEY}"
echo "  - ${MINIO_ALIAS}/${MINIO_BUCKET}/${BASE}/metadata.json"
