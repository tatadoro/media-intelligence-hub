FROM apache/superset:latest

USER root
RUN pip install --no-cache-dir clickhouse-connect
USER superset

# В базовом образе /app/.venv может быть без pip — добавим его и обновим
RUN /app/.venv/bin/python -m ensurepip --upgrade \
 && /app/.venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Ставим ClickHouse Connect (включает интеграцию для Superset / SQLAlchemy)
RUN /app/.venv/bin/python -m pip install --no-cache-dir "clickhouse-connect==0.7.16"

# Smoke-test: проверяем, что SQLAlchemy URI вида clickhousedb://... парсится
# (сетевой коннект не выполняем)
RUN /app/.venv/bin/python - <<'PY'
from sqlalchemy import create_engine
e = create_engine("clickhousedb://user:pass@localhost:8123/default")
print("OK: dialect =", e.dialect.name)
PY

USER superset

COPY --chmod=755 superset-init.sh /app/superset-init.sh