FROM apache/superset:latest

USER root

# Ensure pip exists in Superset venv and upgrade tooling
RUN /app/.venv/bin/python -m ensurepip --upgrade \
 && /app/.venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ClickHouse driver (install into Superset venv)
RUN /app/.venv/bin/python -m pip install --no-cache-dir "clickhouse-connect==0.7.16"

# Postgres driver for Superset metadata DB
RUN /app/.venv/bin/python -m pip install --no-cache-dir "psycopg2-binary>=2.9"

# Superset config (forces using your settings, e.g. Postgres metadata DB)
RUN mkdir -p /app/pythonpath
COPY pythonpath/ /app/pythonpath/
ENV SUPERSET_CONFIG_PATH=/app/pythonpath/superset_config.py

# Optional: small smoke test for clickhousedb dialect (doesn't connect, just loads dialect)
RUN /app/.venv/bin/python - <<'PY'
from sqlalchemy import create_engine
e = create_engine("clickhousedb://user:pass@localhost:8123/default")
print("OK: dialect =", e.dialect.name)
PY

# Init script (should be run by a separate one-off service in docker-compose)
COPY --chmod=755 docker/superset/superset-init.sh /app/superset-init.sh

USER superset

EXPOSE 8088

# IMPORTANT:
# Default container command must run the Superset webserver (long-running).
# The init script is invoked via docker-compose service override (superset-init).
CMD ["superset", "run", "-h", "0.0.0.0", "-p", "8088"]