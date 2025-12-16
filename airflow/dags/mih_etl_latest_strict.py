from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


PROJECT_DIR = Path("/opt/mih")
SILVER_DIR = PROJECT_DIR / "data" / "silver"


def pick_latest_silver_file() -> str:
    """
    Возвращает относительный путь к самому свежему silver JSON-файлу.
    Airflow положит return в XCom, дальше используем в BashOperator.
    """
    if not SILVER_DIR.exists():
        raise FileNotFoundError(f"Silver dir not found: {SILVER_DIR}")

    # Подстрой под свой паттерн, если нужно:
    candidates = sorted(SILVER_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No silver files found in: {SILVER_DIR}")

    latest = candidates[-1]
    # В Makefile/скриптах у тебя используются относительные пути от /opt/mih
    return str(latest.relative_to(PROJECT_DIR))


default_args = {
    "owner": "mih",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="mih_etl_latest_strict",
    default_args=default_args,
    description="MIH ETL: latest silver -> gold -> clickhouse -> reports (strict)",
    start_date=days_ago(1),
    schedule=None,  # оставляем ручной запуск; потом поставим cron
    catchup=False,
    max_active_runs=1,
    tags=["mih"],
) as dag:

    pick_latest_silver = PythonOperator(
        task_id="pick_latest_silver",
        python_callable=pick_latest_silver_file,
    )

    validate_silver = BashOperator(
        task_id="validate_silver",
        bash_command=(
            "cd /opt/mih && "
            "make validate-silver IN={{ ti.xcom_pull(task_ids='pick_latest_silver') }}"
        ),
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=(
            "cd /opt/mih && "
            "python -m src.pipeline.silver_to_gold_local "
            "--input {{ ti.xcom_pull(task_ids='pick_latest_silver') }}"
        ),
    )

    # Если хочешь, можно вычислять имя gold файла в коде и передавать дальше.
    # Но у тебя уже есть make-цель load/gate, которая сама знает, что делать после генерации.
    # Здесь используем make etl для простоты на данном этапе.
    # Более "чисто" — вычислить путь gold и передать в XCom, но это следующий рефакторинг.

    # Вариант A: строго повторяем логику Makefile (сам построит gold и возьмёт нужный файл)
    # Вариант B: сделать отдельные шаги validate-gold / gate / load с явным IN.
    # Ниже — вариант B (прозрачнее).

    # Предполагаем, что silver_to_gold_local пишет parquet с тем же базовым именем.
    # Если имя отличается — лучше вынести в функцию compute_gold_path().
    compute_gold_path = BashOperator(
        task_id="compute_gold_path",
        bash_command=(
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "silver = Path('/opt/mih') / Path(\"{{ ti.xcom_pull(task_ids='pick_latest_silver') }}\")\n"
            "out = silver.name.replace('_clean.json', '_processed.parquet')\n"
            "print(f\"data/gold/{out}\")\n"
            "PY"
        ),
        do_xcom_push=True,
    )

    validate_gold = BashOperator(
        task_id="validate_gold",
        bash_command=(
            "cd /opt/mih && "
            "make validate-gold IN={{ ti.xcom_pull(task_ids='compute_gold_path') }}"
        ),
    )

    quality_gate = BashOperator(
        task_id="quality_gate",
        bash_command=(
            "cd /opt/mih && "
            "make gate IN={{ ti.xcom_pull(task_ids='compute_gold_path') }} STRICT=1"
        ),
    )

    load_to_clickhouse = BashOperator(
        task_id="load_to_clickhouse",
        bash_command=(
            "cd /opt/mih && "
            "python -m src.pipeline.gold_to_clickhouse_local "
            "--input {{ ti.xcom_pull(task_ids='compute_gold_path') }}"
        ),
    )

    sql_reports = BashOperator(
        task_id="sql_reports",
        bash_command="cd /opt/mih && make sql-reports",
    )

    md_report = BashOperator(
        task_id="md_report",
        bash_command="cd /opt/mih && make md-report LAST_HOURS=6",
    )

    # Граф зависимостей
    pick_latest_silver >> validate_silver >> silver_to_gold >> compute_gold_path
    compute_gold_path >> validate_gold >> quality_gate >> load_to_clickhouse
    load_to_clickhouse >> sql_reports >> md_report