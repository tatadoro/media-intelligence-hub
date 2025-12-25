FROM apache/superset:latest

USER root

# Ensure pip exists in Superset venv and upgrade tooling
RUN /app/.venv/bin/python -m ensurepip --upgrade \
 && /app/.venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ClickHouse SQLAlchemy dialect/driver (install into Superset venv)
RUN /app/.venv/bin/python -m pip install --no-cache-dir "clickhouse-connect==0.7.16"

RUN /app/.venv/bin/python -m pip install --no-cache-dir "psycopg2-binary>=2.9"

# Superset config (forces using your settings, e.g. Postgres metadata DB)
RUN mkdir -p /app/pythonpath
COPY pythonpath/ /app/pythonpath/
ENV SUPERSET_CONFIG_PATH=/app/pythonpath/superset_config.py

# Optional: small smoke test for clickhousedb dialect
RUN /app/.venv/bin/python - <<'PY'
from sqlalchemy import create_engine
e = create_engine("clickhousedb://user:pass@localhost:8123/default")
print("OK: dialect =", e.dialect.name)
PY

USER superset

# Container entry script
COPY --chmod=755 superset-init.sh /app/superset-init.sh