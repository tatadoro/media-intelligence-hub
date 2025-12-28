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
    cmd = ["docker", "exec", "-i", args.container]
    cmd += ["clickhouse-client"]
    cmd += ["--host", args.host]
    cmd += ["--port", str(args.port)]
    cmd += ["--user", args.user]
    cmd += ["--password", args.password]
    cmd += ["--database", args.database]
    return cmd


def _docker_ch_cmd_stdin(args: argparse.Namespace) -> list[str]:
    # тот же cmd, но чтение данных идёт через stdin -> clickhouse-client
    return _docker_ch_cmd(args)


def _sql_escape(s: str) -> str:
    return s.replace("'", "''")


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


def ch_int(args: argparse.Namespace, query: str) -> int:
    out = ch_query(args, query)
    if not out:
        return 0
    return int(out.strip())


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


def ensure_ingest_object_name_in_parquet(input_path: Path, object_name: str) -> Tuple[Path, bool]:
    pf = pq.ParquetFile(str(input_path))
    if "ingest_object_name" in set(pf.schema_arrow.names):
        return input_path, False
    out = _write_parquet_with_extra_string_col(input_path, "ingest_object_name", object_name)
    return out, True


def ensure_batch_id_in_parquet(input_path: Path, batch_id: str) -> Tuple[Path, bool]:
    pf = pq.ParquetFile(str(input_path))
    if "batch_id" in set(pf.schema_arrow.names):
        return input_path, False
    out = _write_parquet_with_extra_string_col(input_path, "batch_id", batch_id)
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
        description="Загрузка gold Parquet в ClickHouse (локально) с batch_id + авто-колонками."
    )
    p.add_argument("--input", required=True, help="Путь к gold parquet")
    p.add_argument("--batch-id", default="", help="batch_id (если в parquet нет batch_id — добавим)")
    p.add_argument("--database", default=os.getenv("CH_DATABASE", "media_intel"))
    p.add_argument("--table", default="articles")
    p.add_argument("--container", default="clickhouse")
    p.add_argument("--host", default=os.getenv("CH_HOST_DOCKER", "clickhouse"))
    # В .env часто лежит HTTP-порт 8123; для clickhouse-client нужен нативный 9000.
    p.add_argument("--port", type=int, default=int(os.getenv("CH_PORT_DOCKER", "8123")))
    p.add_argument("--user", default=os.getenv("CH_USER", "admin"))
    p.add_argument("--password", default=os.getenv("CH_PASSWORD", "admin12345"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- ВАЖНО: clickhouse-client использует нативный протокол (обычно 9000), а 8123 — это HTTP.
    # Если мы выполняем clickhouse-client внутри того же контейнера (docker exec),
    # корректнее подключаться к localhost и нативному порту.
    if args.host == args.container:
        args.host = "localhost"

    if args.port == 8123:
        native_port = int(os.getenv("CH_PORT_NATIVE_DOCKER", os.getenv("CH_NATIVE_PORT_DOCKER", "9000")))
        print(f"[WARN] Port 8123 is HTTP; using native port {native_port} for clickhouse-client.")
        args.port = native_port

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    object_name = input_path.name
    temp_paths: list[Path] = []

    # 1) Гарантируем ingest_object_name
    load_path, temp1 = ensure_ingest_object_name_in_parquet(input_path, object_name)
    if temp1:
        temp_paths.append(load_path)

    # 2) Гарантируем batch_id (если передан)
    if args.batch_id:
        load_path2, temp2 = ensure_batch_id_in_parquet(load_path, args.batch_id)
        if temp2:
            temp_paths.append(load_path2)
    else:
        load_path2 = load_path

    # 3) Проверка ingest_object_name
    validate_ingest_object_name(load_path2, object_name)

    # 4) INSERT parquet (через stdin) — берём пересечение колонок (parquet ∩ table)
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

    # 5) Чистим временные файлы
    if temp_paths:
        _cleanup_temp(temp_paths)
        for p in temp_paths:
            print(f"[INFO] Удалили временный файл: {p}")


if __name__ == "__main__":
    main()