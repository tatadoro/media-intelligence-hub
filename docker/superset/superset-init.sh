#!/usr/bin/env bash
set -euo pipefail

echo "[superset-init] starting..."

superset db upgrade

superset fab create-admin \
  --username "${SUPERSET_ADMIN_USERNAME:-admin}" \
  --firstname "${SUPERSET_ADMIN_FIRSTNAME:-Superset}" \
  --lastname "${SUPERSET_ADMIN_LASTNAME:-Admin}" \
  --email "${SUPERSET_ADMIN_EMAIL:-admin@example.com}" \
  --password "${SUPERSET_ADMIN_PASSWORD:-admin}" \
  || true

superset init

if [[ "${SUPERSET_LOAD_EXAMPLES:-false}" == "true" ]]; then
  superset load_examples
fi

echo "[superset-init] done"
exit 0