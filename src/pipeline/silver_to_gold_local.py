from __future__ import annotations

from pathlib import Path
import argparse
import os
import pandas as pd

from src.processing.summarization import enrich_articles_with_summary_and_keywords


CH_TZ = "Europe/Moscow"
MIN_PUBLISHED_AT_UTC = pd.Timestamp("2000-01-01", tz="UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Преобразовать silver-данные в gold-витрину (summary + keywords)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к silver-файлу (JSON/Parquet/CSV), например data/silver/articles_20251211_153500_clean.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Куда сохранить gold Parquet. Если не указан — путь будет выведен из input (data/gold/..._processed.parquet).",
    )
    parser.add_argument(
        "--upload-minio",
        action="store_true",
        help="Если указан, после сохранения gold-файла загрузить его в MinIO (bucket media-intel, prefix gold/).",
    )
    return parser.parse_args()


def infer_output_path(input_path: Path) -> Path:
    """
    Если вход: data/silver/articles_20251211_153500_clean.json,
    то выход: data/gold/articles_20251211_153500_processed.parquet
    """
    data_dir = input_path.parent.parent  # .../data
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    if stem.endswith("_clean"):
        stem = stem.replace("_clean", "_processed")
    else:
        stem = f"{stem}_processed"

    return gold_dir / f"{stem}.parquet"


def load_silver_df(input_path: Path) -> pd.DataFrame:
    """
    Загружаем silver-файл в DataFrame по расширению.
    """
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(input_path, lines=(suffix == ".jsonl"))
    if suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


def normalize_published_at(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализуем published_at:
    - парсим в datetime (в UTC)
    - валидируем (не допускаем NaT и мусор < 2000)
    - приводим к TZ Europe/Moscow и делаем tz-naive

    Важно: так значения корректно лягут в ClickHouse DateTime64(..., 'Europe/Moscow')
    без сдвигов по времени.
    """
    if "published_at" not in df.columns:
        raise ValueError("В silver нет колонки published_at.")

    dt_utc = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    bad = dt_utc.isna()
    if bad.any():
        cols = [c for c in ["id", "title", "link", "published_at"] if c in df.columns]
        sample = df.loc[bad, cols].head(10)
        raise ValueError(
            f"published_at не распарсился в {int(bad.sum())} строках. Примеры:\n"
            f"{sample.to_string(index=False)}"
        )

    too_old = dt_utc < MIN_PUBLISHED_AT_UTC
    if too_old.any():
        cols = [c for c in ["id", "title", "link", "published_at"] if c in df.columns]
        sample = df.loc[too_old, cols].head(10)
        raise ValueError(
            f"Найдены published_at < 2000-01-01 в {int(too_old.sum())} строках. Примеры:\n"
            f"{sample.to_string(index=False)}"
        )

    out = df.copy()
    # Переводим в Europe/Moscow и делаем tz-naive (для Parquet/ClickHouse удобнее)
    out["published_at"] = dt_utc.dt.tz_convert(CH_TZ).dt.tz_localize(None)
    return out


def dedup_batch_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Убираем дубли ВНУТРИ одного входного файла (батча) ДО обогащения.

    Ключ как в ClickHouse:
      dedup_key = if(empty(link), id, link)

    Приоритет победителя:
      1) есть body (len(clean_text) >= 40)
      2) clean_text длиннее
      3) published_at свежее
    """
    out = df.copy()

    if "id" not in out.columns:
        raise ValueError("В данных нет колонки id (нужна для dedup_key).")

    if "link" not in out.columns:
        out["link"] = ""

    if "clean_text" not in out.columns:
        # лучше так, чем падать — но если хочешь строго, можно raise
        out["clean_text"] = ""

    link = out["link"].fillna("").astype(str)
    out["dedup_key"] = link.where(link.str.len() > 0, out["id"].astype(str))

    out["clean_len"] = out["clean_text"].fillna("").astype(str).str.len()
    out["has_body"] = (out["clean_len"] >= 40).astype("int8")

    if "published_at" not in out.columns:
        raise ValueError("Нет published_at — dedup по свежести невозможен.")
    out["published_at_dt"] = pd.to_datetime(out["published_at"], errors="coerce")

    before = len(out)
    out = out.sort_values(
        by=["dedup_key", "has_body", "clean_len", "published_at_dt"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["dedup_key"], keep="first")
    after = len(out)

    print(f"[INFO] Dedup inside batch (silver): {before} -> {after} (dropped {before - after})")

    return out.drop(columns=["dedup_key", "clean_len", "has_body", "published_at_dt"], errors="ignore")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    # input: поддерживаем относительные пути от корня проекта
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")

    print("[INFO] Преобразуем silver -> gold (summary + keywords)")
    print(f"silver: {input_path}")

    df_silver = load_silver_df(input_path)

    # 1) нормализуем время
    df_silver = normalize_published_at(df_silver)

    # 2) NEW: dedup внутри батча ДО обогащения
    df_silver = dedup_batch_silver(df_silver)

    # 3) обогащение
    df_gold = enrich_articles_with_summary_and_keywords(df_silver)

    # output: если задан — тоже поддерживаем относительный путь от корня проекта
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = infer_output_path(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # batch id внутри gold (всегда строкой)
    df_gold["ingest_object_name"] = str(output_path.name)

    print(f"gold:   {output_path}")
    df_gold.to_parquet(output_path, index=False)

    print("[OK] Готово")

    if args.upload_minio:
        import subprocess

        # Проверим, что mc доступен
        try:
            subprocess.run(["mc", "--version"], check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Команда 'mc' не найдена. Установи MinIO Client (mc) и повтори."
            ) from e

        # Конфиг MinIO из окружения (совместимо с .env.example и Makefile)
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        bucket = os.getenv("MINIO_BUCKET", "media-intel")
        alias = os.getenv("MINIO_ALIAS", "local")

        if not minio_access_key or not minio_secret_key:
            raise ValueError(
                "Не заданы MINIO_ACCESS_KEY/MINIO_SECRET_KEY. "
                "Заполни .env (см. .env.example) и повтори."
            )

        # Убедимся, что alias существует: если нет — создадим
        out = subprocess.run(
            ["mc", "alias", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        if f"{alias} " not in out and f"{alias}\t" not in out:
            subprocess.run(
                ["mc", "alias", "set", alias, minio_endpoint, minio_access_key, minio_secret_key],
                check=True,
                capture_output=True,
                text=True,
            )

        dst = f"{alias}/{bucket}/gold/{output_path.name}"

        print(f"[INFO] Загружаем gold в MinIO: {dst}")
        subprocess.run(["mc", "cp", str(output_path), dst], check=True)
        print("[OK] Gold загружен в MinIO")

if __name__ == "__main__":
    main()