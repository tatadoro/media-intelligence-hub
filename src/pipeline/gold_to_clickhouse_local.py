from __future__ import annotations

import os
import argparse
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Загрузка gold Parquet в ClickHouse с защитой от дублей через load_log + фактическую проверку таблицы."
    )
    p.add_argument("--input", type=str, required=True, help="Путь к parquet-файлу (например data/gold/...parquet)")
    p.add_argument(
        "--batch-id",
        type=str,
        required=True,
        help="Идентификатор батча (одинаковый для всех строк текущей загрузки)",
    )
    p.add_argument(
        "--container",
        type=str,
        default=os.getenv("CH_CONTAINER", "clickhouse"),
        help="Имя контейнера ClickHouse (по умолчанию clickhouse)",
    )
    p.add_argument(
        "--database",
        type=str,
        default=os.getenv(
            "CH_DATABASE",
            os.getenv("CLICKHOUSE_DB", os.getenv("CLICKHOUSE_DATABASE", "media_intel")),
        ),
        help="ClickHouse database",
    )
    p.add_argument(
        "--table",
        type=str,
        default="articles",
        help="Target ClickHouse table (default: articles)",
    )
    p.add_argument(
        "--layer",
        type=str,
        default="gold",
        help="Логическое имя слоя для дедупа/аудита (по умолчанию: gold)",
    )
    p.add_argument(
        "--user",
        type=str,
        default=os.getenv("CH_USER", os.getenv("CLICKHOUSE_USER", "admin")),
        help="ClickHouse user",
    )
    p.add_argument(
        "--password",
        type=str,
        default=os.getenv("CH_PASSWORD", os.getenv("CLICKHOUSE_PASSWORD", "")),
        help="ClickHouse password (через CH_PASSWORD или CLICKHOUSE_PASSWORD)",
    )
    return p.parse_args()


def _require_password(args: argparse.Namespace) -> None:
    if not args.password:
        raise SystemExit(
            "[ERROR] ClickHouse password is not set. Set CH_PASSWORD (or CLICKHOUSE_PASSWORD) in environment / .env."
        )


def _docker_ch_cmd(args: argparse.Namespace) -> list[str]:
    _require_password(args)
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
    _require_password(args)
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
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"[ERROR] ClickHouse query failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}\n"
        ) from e


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
        raise ValueError("В parquet нет колонки ingest_object_name. Пайплайн gold должен её добавлять.")

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


def _parquet_batch_id_state(input_path: Path) -> tuple[bool, int, int, set[str]]:
    """
    Возвращает:
      has_col, total_rows, empty_rows, uniq_non_empty_values
    """
    pf = pq.ParquetFile(str(input_path))
    names = set(pf.schema_arrow.names)
    if "batch_id" not in names:
        return False, 0, 0, set()

    total = 0
    empty = 0
    uniq: set[str] = set()

    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["batch_id"])
        for x in t.column(0).to_pylist():
            total += 1
            s = "" if x is None else str(x)
            if s == "":
                empty += 1
            else:
                uniq.add(s)

    return True, total, empty, uniq


def ensure_batch_id_in_parquet(input_path: Path, batch_id: str) -> tuple[Path, bool]:
    """
    Гарантирует, что batch_id присутствует и заполнен одинаково во всех строках.
    Если нужно — создаёт временный parquet и возвращает (temp_path, True).
    Иначе возвращает (input_path, False).
    """
    has_col, total, empty, uniq = _parquet_batch_id_state(input_path)

    # Нужно пересобрать, если:
    # - колонки batch_id нет
    # - или все значения пустые
    # - или есть непустые, но не равны ожидаемому batch_id
    need_rewrite = False
    if not has_col:
        need_rewrite = True
        reason = "parquet без batch_id"
    elif total > 0 and empty == total:
        need_rewrite = True
        reason = "parquet batch_id пустой во всех строках"
    elif uniq and uniq != {batch_id}:
        need_rewrite = True
        reason = f"parquet batch_id != ожидаемого (uniq={sorted(list(uniq))[:5]})"
    else:
        # uniq либо пуст (но не все строки пустые — редкий случай), либо ровно {batch_id}
        # и/или есть пустые значения вперемешку с batch_id — это тоже плохо, перепишем.
        if empty > 0:
            need_rewrite = True
            reason = "parquet batch_id частично пустой"

    if not need_rewrite:
        print(f"[INFO] batch_id валиден в parquet: {input_path.name}")
        return input_path, False

    tmp_dir = Path(tempfile.gettempdir())
    stem = input_path.stem
    safe_batch = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in batch_id)[:120]
    tmp_path = tmp_dir / f"{stem}__with_batch_id__{safe_batch}.parquet"

    print(f"[INFO] {reason}: создали временный файл для загрузки: {tmp_path}")

    table = pq.read_table(str(input_path))
    batch_arr = pa.array([batch_id] * table.num_rows, type=pa.string())

    if "batch_id" in table.schema.names:
        idx = table.schema.get_field_index("batch_id")
        table = table.set_column(idx, "batch_id", batch_arr)
    else:
        table = table.append_column("batch_id", batch_arr)

    pq.write_table(table, str(tmp_path))
    return tmp_path, True


def main() -> None:
    args = parse_args()
    _require_password(args)

    project_root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден parquet: {input_path}")

    object_name = input_path.name

    # 0) Валидация ingest_object_name внутри parquet
    validate_ingest_object_name(input_path, object_name)

    # 0.1) Гарантируем batch_id (создаём temp parquet при необходимости)
    load_path, is_temp = ensure_batch_id_in_parquet(input_path, args.batch_id)

    # 1) Проверки "уже загружено?"
    in_log_q = (
        "SELECT count() "
        f"FROM {args.database}.load_log "
        f"WHERE layer = '{args.layer}' AND object_name = '{object_name}'"
    )
    log_exists = ch_int(args, in_log_q, default=0) > 0

    in_table_q = (
        "SELECT count() "
        f"FROM {args.database}.{args.table} "
        f"WHERE ingest_object_name = '{object_name}'"
    )
    table_cnt = ch_int(args, in_table_q, default=0)

    if table_cnt > 0:
        last_rows_q = (
            "SELECT rows_loaded "
            f"FROM {args.database}.load_log "
            f"WHERE layer = '{args.layer}' AND object_name = '{object_name}' "
            "ORDER BY loaded_at DESC "
            "LIMIT 1"
        )
        last_rows = ch_int(args, last_rows_q, default=-1) if log_exists else -1

        if not log_exists or last_rows == 0:
            fix_q = (
                f"INSERT INTO {args.database}.load_log (layer, object_name, loaded_at, rows_loaded) "
                f"VALUES ('{args.layer}', '{object_name}', now(), {table_cnt})"
            )
            ch_query(args, fix_q)
            print(
                f"[FIX] Данные уже в {args.table} ({table_cnt} строк). "
                "Лог отсутствовал или был некорректным (rows_loaded=0) — добавили корректную запись в load_log."
            )
        else:
            print(
                f"[SKIP] {object_name} уже загружен: в таблице {args.table} есть "
                f"{table_cnt} строк и есть запись в load_log."
            )
        if is_temp and load_path.exists():
            load_path.unlink(missing_ok=True)
        return

    if log_exists and table_cnt == 0:
        print("[WARN] Есть запись в load_log, но в таблице нет строк для этого object_name. Будем грузить заново.")

    # 2) INSERT parquet (через stdin)
    insert_q = f"INSERT INTO {args.database}.{args.table} FORMAT Parquet"
    cmd = _docker_ch_cmd_stdin(args) + ["--query", insert_q]

    print(f"[INFO] Загружаем {load_path} -> {args.database}.{args.table}")
    with load_path.open("rb") as f:
        subprocess.run(cmd, check=True, stdin=f)

    if is_temp:
        try:
            load_path.unlink(missing_ok=True)
            print(f"[INFO] Удалили временный файл: {load_path}")
        except Exception:
            pass

    # 3) Фиксируем фактическое число строк, оказавшихся в таблице
    table_cnt_after = ch_int(
        args,
        f"SELECT count() FROM {args.database}.{args.table} WHERE ingest_object_name = '{object_name}'",
        default=0,
    )

    # 4) Пишем лог вставкой (без мутаций)
    log_q = (
        f"INSERT INTO {args.database}.load_log (layer, object_name, loaded_at, rows_loaded) "
        f"VALUES ('{args.layer}', '{object_name}', now(), {table_cnt_after})"
    )
    ch_query(args, log_q)

    print(f"[OK] Загружено строк (факт в таблице): {table_cnt_after}. Запись в load_log добавлена.")


if __name__ == "__main__":
    main()