#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CH_CONTAINER:-clickhouse}"

# Prefer CH_* (used by scripts), fallback to CLICKHOUSE_* (used by Python), then defaults
USER="${CH_USER:-${CLICKHOUSE_USER:-admin}}"
PASSWORD="${CH_PASSWORD:-${CLICKHOUSE_PASSWORD:-}}"
if [ -z "${PASSWORD}" ]; then
  echo "[ERROR] ClickHouse password is not set. Export CH_PASSWORD (or CLICKHOUSE_PASSWORD)." >&2
  exit 2
fi

DATABASE="${CH_DATABASE:-${CH_DB:-${CLICKHOUSE_DB:-${CLICKHOUSE_DATABASE:-media_intel}}}}"

# Output format for clickhouse-client (default: TSVRaw)
FORMAT="${CH_FORMAT:-TSVRaw}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sql_file>" >&2
  exit 1
fi

SQL_FILE="$1"
if [[ ! -f "$SQL_FILE" ]]; then
  echo "[ERROR] SQL file not found: $SQL_FILE" >&2
  exit 1
fi

ch() {
  docker exec -i "$CONTAINER" clickhouse-client \
    --database "$DATABASE" \
    --user "$USER" --password "$PASSWORD" \
    --format "$FORMAT" \
    "$@"
}

is_auth_error() {
  grep -qiE "Authentication failed|REQUIRED_PASSWORD|There is no user|password is incorrect" <<<"$1"
}

is_connectivity_error() {
  # typical errors when CH is starting / not reachable / container DNS not ready
  grep -qiE "Connection refused|Cannot connect|Network is unreachable|Timeout|timed out|Could not resolve host|DB::NetException" <<<"$1"
}

# 1) Readiness check: retry only for connectivity/startup issues; fail fast on auth.
LAST_ERR=""
for i in 1 2 3 4 5 6 7 8 9 10; do
  if ch --query "SELECT 1" >/dev/null 2>&1; then
    LAST_ERR=""
    break
  fi

  set +e
  LAST_ERR="$(ch --query "SELECT 1" 2>&1)"
  RC=$?
  set -e

  if is_auth_error "$LAST_ERR"; then
    echo "[ERROR] ClickHouse auth failed during readiness check (check CH_USER/CH_PASSWORD)." >&2
    echo "$LAST_ERR" >&2
    exit 2
  fi

  if is_connectivity_error "$LAST_ERR"; then
    echo "[WARN] ClickHouse not ready for queries ($i/10). Retrying... (${LAST_ERR})" >&2
    sleep 1
    continue
  fi

  # Unknown error -> don't retry forever, show it
  echo "[ERROR] ClickHouse readiness check failed with non-retryable error (rc=$RC)." >&2
  echo "$LAST_ERR" >&2
  exit 1
done

if ! ch --query "SELECT 1" >/dev/null 2>&1; then
  echo "[ERROR] ClickHouse is not ready after retries." >&2
  if [ -n "$LAST_ERR" ]; then
    echo "[ERROR] Last error: ${LAST_ERR}" >&2
  fi
  exit 1
fi

# 2) Run SQL file. If it fails — print the real error and stop (no fake "not ready").
set +e
OUT="$(ch --multiquery < "$SQL_FILE" 2>&1)"
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  if is_auth_error "$OUT"; then
    echo "[ERROR] ClickHouse auth failed while running SQL (check CH_USER/CH_PASSWORD)." >&2
    echo "$OUT" >&2
    exit 2
  fi
  echo "[ERROR] Failed to run SQL: $SQL_FILE" >&2
  echo "$OUT" >&2
  exit $RC
fi

# Print results (report queries rely on stdout)
printf "%s\n" "$OUT"