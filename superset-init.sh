#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Running Superset init..."

# миграции
superset db upgrade

# создаём админа (если уже есть — не падаем)
superset fab create-admin \
  --username "${SUPERSET_ADMIN_USERNAME:-admin}" \
  --firstname "${SUPERSET_ADMIN_FIRSTNAME:-Superset}" \
  --lastname "${SUPERSET_ADMIN_LASTNAME:-Admin}" \
  --email "${SUPERSET_ADMIN_EMAIL:-admin@example.com}" \
  --password "${SUPERSET_ADMIN_PASSWORD:-admin}" || true

# инициализация
superset init

echo "[INFO] Starting Superset..."
exec superset run -h 0.0.0.0 -p 8088
