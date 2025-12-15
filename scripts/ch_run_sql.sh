#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CH_CONTAINER:-clickhouse}"
USER="${CH_USER:-admin}"
PASSWORD="${CH_PASSWORD:-__REDACTED__}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sql_file>"
  echo "Example: $0 sql/03_top_keywords.sql"
  exit 1
fi

SQL_FILE="$1"

if [[ ! -f "$SQL_FILE" ]]; then
  echo "SQL file not found: $SQL_FILE"
  exit 1
fi

docker exec -i "$CONTAINER" clickhouse-client \
  -u "$USER" --password "$PASSWORD" \
  --multiquery < "$SQL_FILE"