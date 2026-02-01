from __future__ import annotations

import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def _docker_ch_cmd(args: argparse.Namespace) -> list[str]:
    """Команда для clickhouse-client внутри контейнера БЕЗ stdin.

    Важно: НЕ используем `-i`, чтобы docker exec не держал stdin открытым.
    Это устраняет зависания для коротких запросов (DDL/DML), выполняемых через `--query`.
    """
    cmd = ["docker", "exec", args.container, "clickhouse-client"]
    cmd += ["--host", args.host]
    cmd += ["--port", str(args.port)]
    cmd += ["--user", args.user]
    cmd += ["--password", args.password]
    cmd += ["--database", args.database]
    return cmd


def _docker_ch_cmd_stdin(args: argparse.Namespace) -> list[str]:
    """Команда для clickhouse-client внутри контейнера СО stdin.

    Здесь `-i` нужен, потому что мы подаём parquet через stdin (FORMAT Parquet).
    """
    cmd = ["docker", "exec", "-i", args.container, "clickhouse-client"]
    cmd += ["--host", args.host]
    cmd += ["--port", str(args.port)]
    cmd += ["--user", args.user]
    cmd += ["--password", args.password]
    cmd += ["--database", args.database]
    return cmd


def _sql_escape(s: str) -> str:
    # Для строк в SQL-литерале (single quotes) удваиваем апостроф.
    return (s or "").replace("'", "''")


def ch_query(args: argparse.Namespace, query: str) -> str:
    """Выполняет одиночный SQL запрос в ClickHouse.

    Ключевое правило: для одиночных запросов используем docker exec БЕЗ `-i`
    и передаём SQL только через `--query`. Тогда clickhouse-client не ждёт stdin.
    """
    cmd = _docker_ch_cmd(args) + ["--query", query]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout


def ch_table_columns(args: argparse.Namespace) -> list[str]:
    q = (
        "SELECT name "
        "FROM system.columns "
        f"WHERE database='{_sql_escape(args.database)}' AND table='{_sql_escape(args.table)}' "
        "ORDER BY position "
        "FORMAT TSVRaw"
    )
    out = ch_query(args, q)
    cols = [c.strip() for c in out.splitlines() if c.strip()]
    if not cols:
        raise RuntimeError(
            f"[ERROR] Не смогли получить список колонок из system.columns для {args.database}.{args.table}."
        )
    return cols


def parquet_columns(path: Path) -> set[str]:
    pf = pq.ParquetFile(str(path))
    return set(pf.schema_arrow.names)


def parquet_num_rows(path: Path) -> int:
    """Быстрое количество строк в parquet без чтения таблицы целиком."""
    pf = pq.ParquetFile(str(path))
    md = pf.metadata
    if md is None:
        return pf.read().num_rows
    return int(md.num_rows)


def build_insert_columns(args: argparse.Namespace, parquet_path: Path) -> list[str]:
    tbl_cols = ch_table_columns(args)
    pq_cols = parquet_columns(parquet_path)

    cols = [c for c in tbl_cols if c in pq_cols]
    if not cols:
        raise RuntimeError(
            f"[ERROR] Нет пересечения колонок таблицы {args.database}.{args.table} и parquet: {parquet_path}"
        )

    must = {"id", "published_at", "source"}
    missing_must = [c for c in sorted(must) if c not in cols]
    if missing_must:
        print(f"[WARN] В parquet нет ожидаемых колонок {missing_must}. INSERT может упасть, проверь схему.")

    return cols


def _sanitize_for_filename(s: str, max_len: int = 140) -> str:
    safe = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in s)
    return safe[:max_len] if len(safe) > max_len else safe


def validate_ingest_object_name(input_path: Path, object_name: str) -> None:
    pf = pq.ParquetFile(str(input_path))
    names = set(pf.schema_arrow.names)
    if "ingest_object_name" not in names:
        raise ValueError("В parquet нет колонки ingest_object_name")

    for rg in range(pf.num_row_groups):
        col = pf.read_row_group(rg, columns=["ingest_object_name"]).column(0)
        vals = col.to_pylist()
        if any(v != object_name for v in vals):
            bad = next((v for v in vals if v != object_name), None)
            raise ValueError(
                f"ingest_object_name mismatch in row_group={rg}: expected={object_name}, got={bad}"
            )


def validate_batch_id(input_path: Path, batch_id: str) -> None:
    if not batch_id:
        return

    pf = pq.ParquetFile(str(input_path))
    names = set(pf.schema_arrow.names)
    if "batch_id" not in names:
        raise ValueError("В parquet нет колонки batch_id")

    for rg in range(pf.num_row_groups):
        col = pf.read_row_group(rg, columns=["batch_id"]).column(0)
        vals = col.to_pylist()
        if any(v != batch_id for v in vals):
            bad = next((v for v in vals if v != batch_id), None)
            raise ValueError(f"batch_id mismatch in row_group={rg}: expected={batch_id}, got={bad}")


def _write_parquet_with_extra_string_col(
    input_path: Path,
    col_name: str,
    col_value: str,
) -> Path:
    pf = pq.ParquetFile(str(input_path))
    table = pf.read()

    if col_name in table.column_names:
        return input_path

    arr = pa.array([col_value] * table.num_rows, type=pa.string())
    table2 = table.append_column(col_name, arr)

    tmpdir = Path(tempfile.gettempdir())
    out_name = f"{input_path.stem}__with_{col_name}__{_sanitize_for_filename(col_value)}.parquet"
    out_path = tmpdir / out_name
    pq.write_table(table2, out_path)
    return out_path


def _write_parquet_force_string_col(
    input_path: Path,
    col_name: str,
    col_value: str,
) -> Path:
    """Принудительно ставит значение строковой колонки (создаёт или заменяет)."""
    pf = pq.ParquetFile(str(input_path))
    table = pf.read()

    arr = pa.array([col_value] * table.num_rows, type=pa.string())

    if col_name in table.column_names:
        idx = table.schema.get_field_index(col_name)
        table2 = table.set_column(idx, col_name, arr)
    else:
        table2 = table.append_column(col_name, arr)

    tmpdir = Path(tempfile.gettempdir())
    out_name = f"{input_path.stem}__forced_{col_name}__{_sanitize_for_filename(col_value)}.parquet"
    out_path = tmpdir / out_name
    pq.write_table(table2, out_path)
    return out_path


def ensure_ingest_object_name_in_parquet(input_path: Path, object_name: str) -> Tuple[Path, bool]:
    pf = pq.ParquetFile(str(input_path))
    if "ingest_object_name" in set(pf.schema_arrow.names):
        return input_path, False
    out = _write_parquet_with_extra_string_col(input_path, "ingest_object_name", object_name)
    return out, True


def ensure_batch_id_equals_in_parquet(input_path: Path, batch_id: str) -> Tuple[Path, bool]:
    """Обеспечивает batch_id == заданному значению во всех строках parquet."""
    if not batch_id:
        return input_path, False

    pf = pq.ParquetFile(str(input_path))
    names = set(pf.schema_arrow.names)

    if "batch_id" not in names:
        out = _write_parquet_with_extra_string_col(input_path, "batch_id", batch_id)
        return out, True

    # проверим быстро по row groups
    all_ok = True
    for rg in range(pf.num_row_groups):
        col = pf.read_row_group(rg, columns=["batch_id"]).column(0)
        vals = col.to_pylist()
        if any(v != batch_id for v in vals):
            all_ok = False
            break

    if all_ok:
        return input_path, False

    out = _write_parquet_force_string_col(input_path, "batch_id", batch_id)
    return out, True


def _cleanup_temp(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Загрузка gold Parquet в ClickHouse (локально) с batch_id + авто-колонками. "
            "После успешной загрузки пишет запись в media_intel.load_log (можно отключить флагом). "
            "По умолчанию загрузка идемпотентна по object_name (если уже есть в load_log, повторно не грузим). "
            "Чтобы форсировать перезаливку, используйте --force-load или env MIH_FORCE_LOAD=1."
        )
    )
    p.add_argument("--input", required=True, help="Путь к gold parquet")
    p.add_argument("--batch-id", default="", help="batch_id (будет принудительно проставлен в parquet при загрузке)")
    p.add_argument("--database", default=os.getenv("CH_DATABASE", "media_intel"))
    p.add_argument("--table", default="articles")
    p.add_argument("--container", default="clickhouse")
    p.add_argument("--host", default=os.getenv("CH_HOST_DOCKER", "clickhouse"))
    p.add_argument("--port", type=int, default=int(os.getenv("CH_PORT_DOCKER", "8123")))
    p.add_argument("--user", default=os.getenv("CH_USER", "admin"))
    p.add_argument("--password", default=os.getenv("CH_PASSWORD", "admin12345"))
    p.add_argument(
        "--no-load-log",
        action="store_true",
        help="Не писать запись в таблицу load_log после загрузки parquet",
    )
    p.add_argument(
        "--force-load",
        action="store_true",
        help="Форсировать загрузку даже если object_name уже есть в load_log (аналог MIH_FORCE_LOAD=1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # clickhouse-client внутри контейнера: если ходим в тот же контейнер — лучше localhost + native port
    if args.host == args.container:
        args.host = "localhost"

    # Если указали HTTP-порт (8123/18123), для clickhouse-client берём native 9000
    if args.port in {8123, 18123}:
        native_port = int(os.getenv("CH_PORT_NATIVE_DOCKER", os.getenv("CH_NATIVE_PORT_DOCKER", "9000")))
        print(f"[WARN] Port {args.port} is HTTP/external; using native port {native_port} for clickhouse-client.")
        args.port = native_port

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    object_name = input_path.name

    # ------------------------------------------------------------------
    # Идемпотентность по object_name:
    # если этот parquet уже загружали (есть запись в load_log), повторно не вставляем,
    # иначе появляются дубли в articles при повторных запусках.
    # ------------------------------------------------------------------
    force = bool(getattr(args, "force_load", False)) or (os.getenv("MIH_FORCE_LOAD", "0") == "1")
    if not force:
        try:
            obj_esc = _sql_escape(object_name)
            q = (
                f"SELECT count() "
                f"FROM {args.database}.load_log "
                f"WHERE layer='gold' AND object_name='{obj_esc}' "
                f"FORMAT TSVRaw"
            )
            out = ch_query(args, q).strip()
            already = int(out) if out else 0
            if already > 0:
                print(
                    f"[INFO] Skip load: object already loaded in load_log: {object_name} "
                    f"(use --force-load or MIH_FORCE_LOAD=1 to force)"
                )
                return
        except Exception as e:
            # Если проверка не удалась (например, нет таблицы load_log), не блокируем загрузку.
            print(f"[WARN] Не смогли проверить идемпотентность через load_log для {object_name}: {e}")

    temp_paths: list[Path] = []

    # 1) ingest_object_name
    load_path, t1 = ensure_ingest_object_name_in_parquet(input_path, object_name)
    if t1:
        temp_paths.append(load_path)

    # 2) batch_id: принудительно равен args.batch_id (если передан)
    load_path2, t2 = ensure_batch_id_equals_in_parquet(load_path, args.batch_id)
    if t2:
        temp_paths.append(load_path2)

    # 3) проверки
    validate_ingest_object_name(load_path2, object_name)
    if args.batch_id:
        validate_batch_id(load_path2, args.batch_id)

    # 4) INSERT (parquet ∩ table)
    insert_cols = build_insert_columns(args, load_path2)
    cols_sql = ", ".join(insert_cols)
    insert_q = f"INSERT INTO {args.database}.{args.table} ({cols_sql}) FORMAT Parquet"
    cmd = _docker_ch_cmd_stdin(args) + ["--query", insert_q]

    extra = sorted(list(parquet_columns(load_path2) - set(insert_cols)))
    if extra:
        print(
            f"[INFO] Parquet содержит лишние колонки (их игнорируем при INSERT): {extra[:20]}"
            + (" ..." if len(extra) > 20 else "")
        )

    print(f"[INFO] Загружаем {load_path2} -> {args.database}.{args.table}")
    with load_path2.open("rb") as f:
        subprocess.run(cmd, check=True, stdin=f)

    # 4.1) load_log (короткий запрос без stdin)
    if not getattr(args, "no_load_log", False):
        try:
            rows_loaded = parquet_num_rows(load_path2)
            obj_esc = _sql_escape(object_name)
            log_q = (
                f"INSERT INTO {args.database}.load_log (layer, object_name, loaded_at, rows_loaded) "
                f"VALUES ('gold', '{obj_esc}', now(), {rows_loaded})"
            )
            print("[DEBUG] load_log SQL:", log_q)
            ch_query(args, log_q)
            print(f"[INFO] load_log: записали layer=gold object_name={object_name} rows_loaded={rows_loaded}")
        except Exception as e:
            print(f"[WARN] Не смогли записать load_log после загрузки parquet: {e}")

    # 5) cleanup
    if temp_paths:
        _cleanup_temp(temp_paths)
        for p in temp_paths:
            print(f"[INFO] Удалили временный файл: {p}")


if __name__ == "__main__":
    main()