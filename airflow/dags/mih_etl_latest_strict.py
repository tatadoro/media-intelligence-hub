from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import pendulum
import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

PROJECT_DIR = Path("/opt/mih")
SILVER_DIR = PROJECT_DIR / "data" / "silver"

def compute_batch_id(ti, **_):
    # ВАЖНО: здесь должен быть task_id реального таска compute_gold_path
    v = ti.xcom_pull(task_ids="compute_gold_path")

    # compute_gold_path может вернуть dict {"gold_path": "..."} или просто строку "..."
    gold_path = v["gold_path"] if isinstance(v, dict) else v
    return {"batch_id": Path(str(gold_path)).name}


compute_batch_id_task = PythonOperator(
    task_id="compute_batch_id",
    python_callable=compute_batch_id,
)

def precheck_latest_silver(**context) -> None:
    """
    Логируем выбранный silver и быстро проверяем, что файл не legacy:
    - файл читается
    - в первых N записях есть обязательные ключи (published_at и title)
    Поддерживаем JSON-массив и JSONL.

    Зачем: чтобы DAG не падал на validate_silver из-за случайно подобранного legacy-файла.
    """
    ti = context["ti"]
    silver_rel = ti.xcom_pull(task_ids="pick_latest_silver")
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

        # JSON-массив
        if first == "[":
            data = json.load(f)
            for row in data[:20]:
                checked += 1
                if isinstance(row, dict) and required.issubset(row.keys()):
                    ok = True
                    break
        # JSONL
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


def pick_latest_silver_file() -> str:
    """
    Возвращает относительный путь (от /opt/mih) к самому свежему silver JSON-файлу.

    ВАЖНО: чтобы не подхватить legacy/тестовые файлы, берём только articles_*.json.
    """
    if not SILVER_DIR.exists():
        raise FileNotFoundError(f"Silver dir not found: {SILVER_DIR}")

    candidates = sorted(
        SILVER_DIR.glob("articles_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No silver files found in: {SILVER_DIR}. Expected pattern: articles_*.json"
        )

    latest = candidates[-1]
    return str(latest.relative_to(PROJECT_DIR))


def compute_report_window(last_hours: int = 6, **context) -> dict:
    """
    Берём последнее время публикации из ClickHouse и строим окно отчёта
    [max_dt - last_hours, max_dt]. Так отчёт всегда по реальным данным.

    Возвращаем dict для XCom: {"dt_from": "...", "dt_to": "..."}
    """
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


def compute_gold_path(**context) -> str:
    """
    Строит относительный путь gold parquet из пути silver, который лежит в XCom.
    Ожидаем, что silver файл заканчивается на *_clean.json,
    а gold — на *_processed.parquet.
    """
    ti = context["ti"]
    silver_rel = ti.xcom_pull(task_ids="pick_latest_silver")
    if not silver_rel:
        raise ValueError("XCom is empty: pick_latest_silver did not return a path")

    silver_name = Path(str(silver_rel)).name

    if silver_name.endswith("_clean.json"):
        gold_name = silver_name.replace("_clean.json", "_processed.parquet")
    elif silver_name.endswith(".json"):
        gold_name = silver_name.replace(".json", "_processed.parquet")
    else:
        raise ValueError(f"Unexpected silver filename: {silver_name}")

    return f"data/gold/{gold_name}"


default_args = {
    "owner": "mih",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}
REPORT_LAST_HOURS = int(os.getenv("REPORT_LAST_HOURS", "1"))
with DAG(
    dag_id="mih_etl_latest_strict",
    default_args=default_args,
    description="MIH ETL: latest silver -> gold -> clickhouse -> reports (strict)",
    start_date=pendulum.datetime(2025, 12, 16, 0, 0, 0, tz="Europe/Belgrade"),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    tags=["mih"],
    params={"report_last_hours": REPORT_LAST_HOURS},
) as dag:
    # --- waiters (infra readiness) ---
    # ВАЖНО: внутри контейнера нельзя полагаться на make wait-*, потому что:
    # - wait-minio в Makefile делает curl на localhost:9000 (а надо на minio:9000)
    # - wait-clickhouse в Makefile использует docker exec (в контейнере docker обычно нет)
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
            "    curl -sfS -u \"$USER:$PASS\" \"http://${HOST}:${PORT}/ping\" >/dev/null && echo \"[OK] ClickHouse ready\" && exit 0; "
            "  else "
            "    curl -sfS \"http://${HOST}:${PORT}/ping\" >/dev/null && echo \"[OK] ClickHouse ready\" && exit 0; "
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
            "echo \"[INFO] Waiting for MinIO at http://minio:9000/minio/health/ready\"; "
            "for i in $(seq 1 30); do "
            "  curl -sfS http://minio:9000/minio/health/ready >/dev/null && echo \"[OK] MinIO ready\" && exit 0; "
            "  sleep 1; "
            "done; "
            "echo \"[ERROR] MinIO not ready (timeout)\"; exit 1"
        ),
    )

    # --- pick/precheck/validate silver ---
    pick_latest_silver = PythonOperator(
        task_id="pick_latest_silver",
        python_callable=pick_latest_silver_file,
    )

    precheck_silver = PythonOperator(
        task_id="precheck_silver",
        python_callable=precheck_latest_silver,
    )

    validate_silver = BashOperator(
        task_id="validate_silver",
        bash_command=(
            "cd /opt/mih && "
            "make validate-silver IN=\"{{ ti.xcom_pull(task_ids='pick_latest_silver') }}\""
        ),
    )

    # --- silver -> gold ---
    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=(
            "cd /opt/mih && "
            "python -m src.pipeline.silver_to_gold_local "
            "--input \"{{ ti.xcom_pull(task_ids='pick_latest_silver') }}\""
        ),
    )

    compute_gold_path_task = PythonOperator(
        task_id="compute_gold_path",
        python_callable=compute_gold_path,
    )

    validate_gold = BashOperator(
        task_id="validate_gold",
        bash_command=(
            "cd /opt/mih && "
            "make validate-gold IN=\"{{ ti.xcom_pull(task_ids='compute_gold_path') }}\""
        ),
    )

    quality_gate = BashOperator(
        task_id="quality_gate",
        bash_command=(
            "cd /opt/mih && "
            "make gate IN=\"{{ ti.xcom_pull(task_ids='compute_gold_path') }}\" STRICT=1"
        ),
    )

    delete_batch_in_ch = BashOperator(
        task_id="delete_batch_in_ch",
        bash_command=(
            "set -euo pipefail; cd /opt/mih; "
            "BATCH_ID=\"{{ ti.xcom_pull(task_ids='compute_batch_id')['batch_id'] }}\"; "
            "TMP_SQL=$(mktemp); "
            "printf \"ALTER TABLE articles DELETE WHERE batch_id='%s';\\n\" \"$BATCH_ID\" > \"$TMP_SQL\"; "
            "./scripts/ch_run_sql.sh \"$TMP_SQL\"; rm -f \"$TMP_SQL\""
        ),
    )

    # --- load to clickhouse ---
    load_to_clickhouse = BashOperator(
        task_id="load_to_clickhouse",
        bash_command=(
            "cd /opt/mih && "
            "python -m src.pipeline.gold_to_clickhouse_local "
            "--input \"{{ ti.xcom_pull(task_ids='compute_gold_path') }}\""
        ),
    )

    # --- sql + md report ---
    sql_reports = BashOperator(
        task_id="sql_reports",
        bash_command="cd /opt/mih && make report",
    )

    compute_report_window_task = PythonOperator(
        task_id="compute_report_window",
        python_callable=compute_report_window,
        op_kwargs={"last_hours": "{{ params.report_last_hours }}"},
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
            "echo \"${REPORT_FILE}\""
        ),
    )

    # --- upload report to MinIO ---
    # best-effort: если в airflow-контейнере нет mc, не валим DAG
    upload_report_to_minio = BashOperator(
        task_id="upload_report_to_minio",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/mih; "
            "REPORT_FILE=\"{{ (ti.xcom_pull(task_ids='md_report') or '') | trim }}\"; "
            "if [ -z \"${REPORT_FILE}\" ]; then "
            "  echo \"[ERROR] md_report XCom is empty; cannot upload deterministically\" 1>&2; "
            "  exit 1; "
            "fi; "
            "DT_FROM=\"{{ ti.xcom_pull(task_ids='compute_report_window')['dt_from'] }}\"; "
            "DT_TO=\"{{ ti.xcom_pull(task_ids='compute_report_window')['dt_to'] }}\"; "
            "echo \"[INFO] Uploading report: ${REPORT_FILE} (dt_from=${DT_FROM}, dt_to=${DT_TO})\" 1>&2; "
            "python -m src.utils.minio_reports "
            "--last-hours '{{ params.report_last_hours }}' "
            "--report-path \"${REPORT_FILE}\" "
            "--dt-from \"${DT_FROM}\" "
            "--dt-to \"${DT_TO}\""
        ),
        params={"report_last_hours": 1},
    )

    # --- minimal observability log (best-effort) ---
    quality_summary_log = BashOperator(
        task_id="quality_summary_log",
        bash_command="cd /opt/mih && make health && make quality || true",
    )

    # --- Dependencies ---
    wait_clickhouse >> wait_minio >> pick_latest_silver
    pick_latest_silver >> precheck_silver >> validate_silver >> silver_to_gold >> compute_gold_path_task
    compute_gold_path_task >> validate_gold >> quality_gate >> compute_batch_id_task >> delete_batch_in_ch >> load_to_clickhouse
    load_to_clickhouse >> sql_reports >> compute_report_window_task >> md_report >> upload_report_to_minio >> quality_summary_log