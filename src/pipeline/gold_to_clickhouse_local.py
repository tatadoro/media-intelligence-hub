from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Загрузка gold Parquet в ClickHouse с защитой от дублей через load_log + фактическую проверку таблицы."
    )
    p.add_argument("--input", type=str, required=True, help="Путь к parquet-файлу (например data/gold/...parquet)")
    p.add_argument(
        "--container",
        type=str,
        default="clickhouse",
        help="Имя контейнера ClickHouse (по умолчанию clickhouse)",
    )
    p.add_argument("--user", type=str, default="admin", help="ClickHouse user")
    p.add_argument("--password", type=str, default="__REDACTED__", help="ClickHouse password")
    p.add_argument("--database", type=str, default="media_intel", help="ClickHouse database")
    p.add_argument("--table", type=str, default="articles", help="Целевая таблица")
    p.add_argument("--layer", type=str, default="gold", help="Имя слоя для load_log")
    return p.parse_args()


def _docker_ch_cmd(args: argparse.Namespace) -> list[str]:
    return [
        "docker",
        "exec",
        args.container,
        "clickhouse-client",
        "--database",
        args.database,
        "-u",
        args.user,
        "--password",
        args.password,
    ]


def _docker_ch_cmd_stdin(args: argparse.Namespace) -> list[str]:
    return [
        "docker",
        "exec",
        "-i",
        args.container,
        "clickhouse-client",
        "--database",
        args.database,
        "-u",
        args.user,
        "--password",
        args.password,
    ]


def ch_query(args: argparse.Namespace, query: str) -> str:
    cmd = _docker_ch_cmd(args) + ["--query", query]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout.strip()


def ch_int(args: argparse.Namespace, query: str, default: int = 0) -> int:
    out = ch_query(args, query)
    if not out:
        return default
    return int(out)


def validate_ingest_object_name(input_path: Path, object_name: str) -> None:
    """
    Проверяем, что в parquet есть колонка ingest_object_name,
    и что ВСЕ значения равны имени файла.

    Делаем это без чтения всего датасета в память: идём по row_group.
    """
    pf = pq.ParquetFile(str(input_path))
    schema_names = set(pf.schema_arrow.names)

    if "ingest_object_name" not in schema_names:
        raise ValueError(
            "В parquet нет колонки ingest_object_name. Пайплайн gold должен её добавлять."
        )

    uniq: set[str] = set()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["ingest_object_name"])
        col = t.column(0).to_pylist()
        for x in col:
            if x is None:
                continue
            uniq.add(str(x))
            if len(uniq) > 1:
                break
        if len(uniq) > 1:
            break

    if uniq != {object_name}:
        raise ValueError(
            "ingest_object_name внутри parquet не совпадает с именем файла.\n"
            f"Файл: {object_name}\n"
            f"Уникальные значения в колонке: {sorted(list(uniq))[:10]}"
        )


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден parquet: {input_path}")

    object_name = input_path.name

    # 0) Валидация batch id внутри parquet
    validate_ingest_object_name(input_path, object_name)

    # 1) Проверки "уже загружено?"
    in_log_q = (
        "SELECT count() "
        "FROM load_log "
        f"WHERE layer = '{args.layer}' AND object_name = '{object_name}'"
    )
    log_exists = ch_int(args, in_log_q, default=0) > 0

    in_table_q = (
        "SELECT count() "
        f"FROM {args.table} "
        f"WHERE ingest_object_name = '{object_name}'"
    )
    table_cnt = ch_int(args, in_table_q, default=0)

    if table_cnt > 0:
        # Данные уже в articles — значит повторно лить нельзя.
        if not log_exists:
            # Самовосстановление: данные есть, лога нет — добавим лог.
            log_q = (
                "INSERT INTO load_log (layer, object_name, loaded_at, rows_loaded) "
                f"VALUES ('{args.layer}', '{object_name}', now(), {table_cnt})"
            )
            ch_query(args, log_q)
            print(
                f"[OK] Данные уже были в {args.table} ({table_cnt} строк). "
                "Лога не было — добавили запись в load_log."
            )
        else:
            print(
                f"[SKIP] {object_name} уже загружен: в таблице {args.table} есть "
                f"{table_cnt} строк и есть запись в load_log."
            )
        return

    if log_exists and table_cnt == 0:
        # Пограничный кейс: лог есть, данных нет (например, руками чистили таблицу)
        print("[WARN] Есть запись в load_log, но в таблице нет строк для этого object_name. Будем грузить заново.")

    # 2) INSERT parquet (через stdin)
    insert_q = f"INSERT INTO {args.table} FORMAT Parquet"
    cmd = _docker_ch_cmd_stdin(args) + ["--query", insert_q]

    print(f"[INFO] Загружаем {input_path} -> {args.database}.{args.table}")
    with input_path.open("rb") as f:
        subprocess.run(cmd, check=True, stdin=f)

    # 3) Фиксируем фактическое число строк, оказавшихся в таблице
    table_cnt_after = ch_int(
        args,
        f"SELECT count() FROM {args.table} WHERE ingest_object_name = '{object_name}'",
        default=0,
    )

    # 4) Пишем лог БЕЗ DELETE/UPDATE: load_log может не поддерживать мутации (у тебя это и падало).
    # Дубли в load_log допустимы: в view/отчётах ты всё равно берёшь max(loaded_at) для батча.
    log_q = (
        "INSERT INTO load_log (layer, object_name, loaded_at, rows_loaded) "
        f"VALUES ('{args.layer}', '{object_name}', now(), {table_cnt_after})"
    )
    ch_query(args, log_q)

    print(f"[OK] Загружено строк (факт в таблице): {table_cnt_after}. Запись в load_log добавлена.")


if __name__ == "__main__":
    main()