from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "mih",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="mih_etl_latest_strict",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 12, 1),
    schedule=None,          # вручную запускаем, чтобы спокойно проверить
    catchup=False,
    max_active_runs=1,
    tags=["mih", "etl"],
) as dag:
    etl_latest_strict = BashOperator(
        task_id="etl_latest_strict",
        bash_command="cd /opt/mih && make etl-latest-strict",
    )

    etl_latest_strict