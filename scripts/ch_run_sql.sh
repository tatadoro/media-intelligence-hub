#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CH_CONTAINER:-clickhouse}"

# Prefer CH_* (used by scripts), fallback to CLICKHOUSE_* (used by Python), then defaults
USER="${CH_USER:-${CLICKHOUSE_USER:-admin}}"
PASSWORD="${CH_PASSWORD:-${CLICKHOUSE_PASSWORD:-__REDACTED__}}"
DATABASE="${CH_DATABASE:-${CLICKHOUSE_DB:-${CLICKHOUSE_DATABASE:-media_intel}}}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sql_file>"
  exit 1
fi

SQL_FILE="$1"
if [[ ! -f "$SQL_FILE" ]]; then
  echo "SQL file not found: $SQL_FILE"
  exit 1
fi

# Retry because ClickHouse can be "running" but not yet accepting connections.
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker exec -i "$CONTAINER" clickhouse-client \
      --database "$DATABASE" \
      -u "$USER" --password "$PASSWORD" \
      --multiquery < "$SQL_FILE" >/dev/null 2>&1; then
    # re-run with output (so report queries print results)
    docker exec -i "$CONTAINER" clickhouse-client \
      --database "$DATABASE" \
      -u "$USER" --password "$PASSWORD" \
      --multiquery < "$SQL_FILE"
    exit 0
  fi
  echo "[WARN] ClickHouse not ready for SQL ($i/10). Retrying..."
  sleep 1
done

echo "[ERROR] Failed to run SQL after retries: $SQL_FILE"
exit 1
