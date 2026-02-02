from __future__ import annotations

from datetime import timedelta
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mih",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="mih_collect_tg",
    default_args=default_args,
    description="MIH collect Telegram: raw -> silver",
    start_date=pendulum.datetime(2025, 12, 16, 0, 0, 0, tz="Europe/Vilnius"),
    schedule="*/10 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mih", "tg", "collect"],
) as dag:
    # 1) Собираем raw. В конце выводим путь к raw-файлу (последняя строка) в XCom.
    tg_raw = BashOperator(
        task_id="tg_raw",
        do_xcom_push=True,
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/mih; "
            # единый BATCH_ID на прогон (UTC)
            "BATCH_ID=\"{{ data_interval_end.in_timezone('UTC').strftime('%Y-%m-%dT%H:%M:%SZ') }}\"; "
            # на всякий случай принудим combined, чтобы имя было предсказуемее
            "TG_COMBINED=1 BATCH_ID=\"$BATCH_ID\" make tg-raw 1>&2; "
            # берём самый свежий combined raw
            "RAW=$(ls -1t data/raw/articles_*_telegram_combined.json 2>/dev/null | head -n 1); "
            "test -n \"$RAW\" || (echo \"[ERROR] No tg raw produced\" 1>&2; exit 1); "
            "echo \"$RAW\""
        ),
    )

    # 2) Делаем silver из конкретного raw (того, что только что собрали)
    tg_silver = BashOperator(
        task_id="tg_silver",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/mih; "
            "BATCH_ID=\"{{ data_interval_end.in_timezone('UTC').strftime('%Y-%m-%dT%H:%M:%SZ') }}\"; "
            "RAW_IN=\"{{ ti.xcom_pull(task_ids='tg_raw') }}\"; "
            "test -n \"$RAW_IN\" || (echo \"[ERROR] tg_raw XCom is empty\" 1>&2; exit 1); "
            "RAW_IN=\"$RAW_IN\" TG_COMBINED=1 BATCH_ID=\"$BATCH_ID\" make tg-silver"
        ),
    )

    tg_raw >> tg_silver