from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
import pyarrow.parquet as pq
import requests

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator

# EmailOperator импортируем только если реально включаем отправку (чтобы локально не тащить SMTP-настройки)
# from airflow.operators.email import EmailOperator  # не импортируем глобально

PROJECT_DIR = Path("/opt/mih")
SILVER_DIR = PROJECT_DIR / "data" / "silver"
TZ = pendulum.timezone("Europe/Vilnius")


# ---------- утилиты (почти 1-в-1 из твоего mih_etl_latest_strict.py) ----------

def precheck_latest_silver(**context) -> None:
    ti = context["ti"]
    silver_rel = ti.xcom_pull(task_ids=context["params"]["pick_task_id"])
    if not silver_rel:
        raise ValueError("XCom empty: pick_latest_silver returned nothing")

    silver_path = PROJECT_DIR / Path(str(silver_rel))
    logging.info("Selected silver file: %s", silver_path)

    if not silver_path.exists():
        raise FileNotFoundError(f"Selected silver file does not exist: {silver_path}")

    required = {"published_at", "title"}
    checked = 0
    ok = False

    with silver_path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            for row in data[:20]:
                checked += 1
                if isinstance(row, dict) and required.issubset(row.keys()):
                    ok = True
                    break
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                checked += 1
                if isinstance(row, dict) and required.issubset(row.keys()):
                    ok = True
                    break
                if checked >= 20:
                    break

    if not ok:
        raise ValueError(
            f"Silver precheck failed: required keys {sorted(required)} not found "
            f"in first {checked} records of {silver_rel}"
        )

    logging.info("Silver precheck OK (%s records scanned).", checked)


def pick_latest_silver_file(silver_glob: str, exclude_contains: str | None = None) -> str:
    if not SILVER_DIR.exists():
        raise FileNotFoundError(f"Silver dir not found: {SILVER_DIR}")

    candidates = []
    for p in SILVER_DIR.glob(silver_glob):
        if exclude_contains and (exclude_contains in p.name):
            continue
        candidates.append(p)

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No silver files found in: {SILVER_DIR}. Expected pattern: {silver_glob}"
        )

    latest = candidates[-1]
    return str(latest.relative_to(PROJECT_DIR))


def compute_gold_path(pick_task_id: str, **context) -> str:
    """
    Возвращает относительный путь к gold parquet, который действительно создаёт silver_to_gold_local.py.

    В проекте сейчас создаётся файл вида:
      *_clean_processed.parquet
    а не просто:
      *_processed.parquet

    Поэтому:
    - если silver заканчивается на *_clean.json -> gold = *_clean_processed.parquet
    - иначе (на всякий случай) пробуем два варианта и выбираем существующий
    """
    ti = context["ti"]
    silver_rel = ti.xcom_pull(task_ids=pick_task_id)
    if not silver_rel:
        raise ValueError(f"XCom is empty: {pick_task_id} did not return a path")

    silver_rel = str(silver_rel)
    silver_name = Path(silver_rel).name

    # Кандидат №1 — то, что реально пишется в /opt/mih/data/gold
    if silver_name.endswith("_clean.json"):
        gold_name_1 = silver_name.replace("_clean.json", "_clean_processed.parquet")
    elif silver_name.endswith(".json"):
        gold_name_1 = silver_name.replace(".json", "_clean_processed.parquet")
    else:
        raise ValueError(f"Unexpected silver filename: {silver_name}")

    # Кандидат №2 — старый/запасной формат (если где-то ещё используется)
    if silver_name.endswith("_clean.json"):
        gold_name_2 = silver_name.replace("_clean.json", "_processed.parquet")
    elif silver_name.endswith(".json"):
        gold_name_2 = silver_name.replace(".json", "_processed.parquet")
    else:
        gold_name_2 = None

    cand_1 = PROJECT_DIR / "data" / "gold" / gold_name_1
    if cand_1.exists():
        return str(cand_1.relative_to(PROJECT_DIR))

    if gold_name_2:
        cand_2 = PROJECT_DIR / "data" / "gold" / gold_name_2
        if cand_2.exists():
            return str(cand_2.relative_to(PROJECT_DIR))

    # Если не нашли — возвращаем основной ожидаемый путь (и лог/ошибка всплывёт в gold_non_empty)
    return str(cand_1.relative_to(PROJECT_DIR))


def gold_non_empty(gold_task_id: str, **context) -> bool:
    ti = context["ti"]
    gold_rel = ti.xcom_pull(task_ids=gold_task_id)
    if not gold_rel:
        raise ValueError(f"XCom is empty: {gold_task_id} did not return a path")

    gold_path = PROJECT_DIR / Path(str(gold_rel))
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold parquet not found: {gold_path}")

    pf = pq.ParquetFile(str(gold_path))
    rows = int(pf.metadata.num_rows)

    logging.info("Gold rows=%s path=%s", rows, gold_rel)
    return rows > 0


def compute_report_window(last_hours: int = 6, **context) -> dict:
    host = os.getenv("CH_HOST", os.getenv("CLICKHOUSE_HOST", "clickhouse"))
    port = int(os.getenv("CH_PORT", os.getenv("CLICKHOUSE_PORT", "8123")))
    db = os.getenv(
        "CH_DATABASE",
        os.getenv("CLICKHOUSE_DATABASE", os.getenv("CLICKHOUSE_DB", "media_intel")),
    )
    table = os.getenv("REPORT_TABLE", "articles")

    user = os.getenv("CH_USER", os.getenv("CLICKHOUSE_USER", ""))
    password = os.getenv("CH_PASSWORD", os.getenv("CLICKHOUSE_PASSWORD", ""))

    url = f"http://{host}:{port}/"
    query = (
        f"SELECT formatDateTime(max(published_at), '%F %T') "
        f"FROM {db}.{table} FORMAT TabSeparated"
    )

    auth = (user, password) if (user or password) else None
    r = requests.get(url, params={"query": query}, auth=auth, timeout=30)
    r.raise_for_status()

    max_dt_str = r.text.strip()
    if not max_dt_str:
        raise ValueError("ClickHouse returned empty max(published_at). Table is empty?")

    max_dt = datetime.fromisoformat(max_dt_str)
    dt_from = max_dt - timedelta(hours=int(last_hours))

    return {
        "dt_from": dt_from.strftime("%Y-%m-%d %H:%M:%S"),
        "dt_to": max_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }


default_args = {
    "owner": "mih",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def build_pipeline(
    dag: DAG,
    prefix: str,
    silver_glob: str,
    exclude_contains: str | None,
) -> tuple:
    """
    Возвращает (last_task, task_group)
    Мы строим цепочку до load_to_clickhouse включительно.
    """
    with TaskGroup(group_id=f"{prefix}_pipeline", dag=dag) as tg:
        pick = PythonOperator(
            task_id=f"{prefix}_pick_latest_silver",
            python_callable=pick_latest_silver_file,
            op_kwargs={"silver_glob": silver_glob, "exclude_contains": exclude_contains},
        )

        precheck = PythonOperator(
            task_id=f"{prefix}_precheck_silver",
            python_callable=precheck_latest_silver,
            params={"pick_task_id": f"{prefix}_pipeline.{prefix}_pick_latest_silver"},
        )

        validate_silver = BashOperator(
            task_id=f"{prefix}_validate_silver",
            bash_command=(
                "cd /opt/mih && "
                f"make validate-silver IN=\"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_pick_latest_silver') }}}}\""
            ),
        )

        silver_to_gold = BashOperator(
            task_id=f"{prefix}_silver_to_gold",
            bash_command=(
                "cd /opt/mih && "
                "python -m src.pipeline.silver_to_gold_local "
                f"--input \"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_pick_latest_silver') }}}}\" "
                "--with-actions"
            ),
        )

        gold_path = PythonOperator(
            task_id=f"{prefix}_compute_gold_path",
            python_callable=compute_gold_path,
            op_kwargs={"pick_task_id": f"{prefix}_pipeline.{prefix}_pick_latest_silver"},
        )

        non_empty = ShortCircuitOperator(
            task_id=f"{prefix}_gold_non_empty",
            python_callable=gold_non_empty,
            op_kwargs={"gold_task_id": f"{prefix}_pipeline.{prefix}_compute_gold_path"},
        )

        validate_gold = BashOperator(
            task_id=f"{prefix}_validate_gold",
            bash_command=(
                "cd /opt/mih && "
                f"make validate-gold IN=\"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_compute_gold_path') }}}}\""
            ),
        )

        quality_gate = BashOperator(
            task_id=f"{prefix}_quality_gate",
            bash_command=(
                "cd /opt/mih && "
                f"make gate IN=\"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_compute_gold_path') }}}}\" STRICT=1"
            ),
        )

        # batch_id = имя gold файла (как у тебя)
        compute_batch_id = BashOperator(
            task_id=f"{prefix}_compute_batch_id",
            do_xcom_push=True,
            bash_command=(
                "set -euo pipefail; "
                f"GOLD=\"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_compute_gold_path') }}}}\"; "
                "BATCH_ID=$(basename \"$GOLD\"); "
                "echo \"$BATCH_ID\""
            ),
        )

        delete_batch_in_ch = BashOperator(
            task_id=f"{prefix}_delete_batch_in_ch",
            bash_command=(
                "set -euo pipefail; cd /opt/mih; "
                f"BATCH_ID=\"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_compute_batch_id') }}}}\"; "
                "TMP_SQL=$(mktemp); "
                "printf \"ALTER TABLE articles DELETE WHERE batch_id='%s';\\n\" \"$BATCH_ID\" > \"$TMP_SQL\"; "
                "./scripts/ch_run_sql.sh \"$TMP_SQL\"; rm -f \"$TMP_SQL\""
            ),
        )

        load_to_clickhouse = BashOperator(
            task_id=f"{prefix}_load_to_clickhouse",
            bash_command=(
                "cd /opt/mih && "
                "python -m src.pipeline.gold_to_clickhouse_local "
                f"--input \"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_compute_gold_path') }}}}\" "
                f"--batch-id \"{{{{ ti.xcom_pull(task_ids='{prefix}_pipeline.{prefix}_compute_batch_id') }}}}\""
            ),
        )

        pick >> precheck >> validate_silver >> silver_to_gold >> gold_path
        gold_path >> non_empty >> validate_gold >> quality_gate >> compute_batch_id >> delete_batch_in_ch >> load_to_clickhouse

        return load_to_clickhouse, tg


with DAG(
    dag_id="mih_etl_daily",
    default_args=default_args,
    description="MIH daily: tg + rss latest silver -> gold -> CH -> single report -> email",
    start_date=pendulum.datetime(2026, 2, 1, tz=TZ),
    schedule="0 9 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mih", "daily"],
) as dag:
    wait_clickhouse = BashOperator(
        task_id="wait_clickhouse",
        bash_command=(
            "set -euo pipefail; "
            "HOST=${CH_HOST:-${CLICKHOUSE_HOST:-clickhouse}}; "
            "PORT=${CH_PORT:-${CLICKHOUSE_PORT:-8123}}; "
            "USER=${CH_USER:-${CLICKHOUSE_USER:-}}; "
            "PASS=${CH_PASSWORD:-${CLICKHOUSE_PASSWORD:-}}; "
            "echo \"[INFO] Waiting for ClickHouse at http://${HOST}:${PORT}/ping\"; "
            "for i in $(seq 1 30); do "
            "  if [ -n \"$USER$PASS\" ]; then "
            "    curl -sfS -u \"$USER:$PASS\" \"http://${HOST}:${PORT}/ping\" >/dev/null && exit 0; "
            "  else "
            "    curl -sfS \"http://${HOST}:${PORT}/ping\" >/dev/null && exit 0; "
            "  fi; "
            "  sleep 1; "
            "done; "
            "echo \"[ERROR] ClickHouse not ready (timeout)\"; exit 1"
        ),
    )

    wait_minio = BashOperator(
        task_id="wait_minio",
        bash_command=(
            "set -euo pipefail; "
            "for i in $(seq 1 30); do "
            "  curl -sfS http://minio:9000/minio/health/ready >/dev/null && exit 0; "
            "  sleep 1; "
            "done; "
            "echo \"[ERROR] MinIO not ready (timeout)\"; exit 1"
        ),
    )

    # барьер: после того как оба сервиса готовы — начинаем ETL
    ready = EmptyOperator(task_id="ready")

    tg_last, _ = build_pipeline(
        dag=dag,
        prefix="tg",
        silver_glob="articles_*_telegram_*_clean.json",
        exclude_contains=None,
    )

    rss_last, _ = build_pipeline(
        dag=dag,
        prefix="rss",
        silver_glob="articles_[0-9]*_clean.json",
        exclude_contains="telegram",
    )

    sql_reports = BashOperator(
        task_id="sql_reports",
        bash_command="cd /opt/mih && make report",
    )

    compute_window = PythonOperator(
        task_id="compute_report_window",
        python_callable=compute_report_window,
        op_kwargs={"last_hours": int(os.getenv("REPORT_LAST_HOURS", "24"))},
    )

    md_report = BashOperator(
        task_id="md_report",
        do_xcom_push=True,
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/mih; "
            "python -m src.reporting.generate_report "
            "--from \"{{ (ti.xcom_pull(task_ids='compute_report_window') or {})['dt_from'] }}\" "
            "--to \"{{ (ti.xcom_pull(task_ids='compute_report_window') or {})['dt_to'] }}\" "
            "--table articles 1>&2; "
            "REPORT_FILE=$(ls -1t reports/daily_report_*.md | head -n 1); "
            "test -n \"$REPORT_FILE\"; "
            "ABS=\"/opt/mih/${REPORT_FILE}\"; "
            "cp \"$ABS\" /opt/mih/reports/latest.md; "
            "echo \"/opt/mih/reports/latest.md\""
        ),
    )

    # ============================================================
    # Email (мягкий режим для local/dev)
    # - По умолчанию отправка выключена: send_email = EmptyOperator (DAG не падает).
    # - Чтобы включить отправку:
    #     MIH_EMAIL_ENABLED=1
    #     MIH_EMAIL_TO=you@example.com (или "a@b,c@d")
    #     (опционально) MIH_EMAIL_FROM=...
    #
    # Важно по безопасности:
    # - адреса/пароли не хардкодим в DAG
    # - SMTP-учётку заводим как Airflow Connection (или env на контейнере),
    #   а не в репозитории
    # ============================================================
    EMAIL_ENABLED = os.getenv("MIH_EMAIL_ENABLED", "0").strip().lower() in {"1", "true", "yes", "y"}
    EMAIL_TO_RAW = os.getenv("MIH_EMAIL_TO", "").strip()
    EMAIL_FROM = os.getenv("MIH_EMAIL_FROM", "mih@local").strip()

    if EMAIL_TO_RAW:
        email_to = [x.strip() for x in EMAIL_TO_RAW.replace(";", ",").split(",") if x.strip()]
    else:
        email_to = []

    if EMAIL_ENABLED and email_to:
        from airflow.operators.email import EmailOperator  # локальный импорт

        send_email = EmailOperator(
            task_id="send_email",
            to=email_to,
            subject="MIH Daily Report — {{ ds }}",
            html_content="<p>Отчёт во вложении.</p>",
            files=["{{ ti.xcom_pull(task_ids='md_report') }}"],
            from_email=EMAIL_FROM,
        )
    else:
        send_email = EmptyOperator(task_id="send_email")

    # фикс зависимостей: никаких list >> list
    [wait_clickhouse, wait_minio] >> ready
    ready >> [tg_last, rss_last]
    [tg_last, rss_last] >> sql_reports >> compute_window >> md_report >> send_email