from __future__ import annotations

from pathlib import Path
import argparse
import os
import pandas as pd

from src.processing.summarization import enrich_articles_with_summary_and_keywords

CH_TZ = "Europe/Moscow"
MIN_PUBLISHED_AT_UTC = pd.Timestamp("2000-01-01", tz="UTC")

# Фильтрация низкокачественного контента перед обогащением
EXCLUDE_BODY_STATUS = {"parsed_empty", "fetch_failed", "no_link", "too_short"}
MIN_CLEAN_TEXT_CHARS = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Преобразование silver -> gold: summary + keywords (TF-IDF) и выгрузка в MinIO (опционально)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Путь к silver-файлу (json/jsonl/parquet/csv).",
    )
    parser.add_argument(
        "-o",
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
    Если вход: data/silver/articles_20251210_155554_enriched_clean.json
    Выход:     data/gold/articles_20251210_155554_enriched_clean_processed.parquet
    """
    project_root = Path(__file__).resolve().parents[2]
    gold_dir = project_root / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.name
    for suf in (".jsonl", ".json", ".parquet", ".csv"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    return gold_dir / f"{stem}_processed.parquet"


def load_silver_df(input_path: Path) -> pd.DataFrame:
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
    """
    out = df.copy()

    if "published_at" not in out.columns:
        out["published_at"] = None

    published_at_utc = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
    published_at_utc = published_at_utc.where(published_at_utc >= MIN_PUBLISHED_AT_UTC)

    published_at_msk = published_at_utc.dt.tz_convert(CH_TZ).dt.tz_localize(None)
    out["published_at"] = published_at_msk
    return out


def _safe_col_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Возвращает Series для колонки `col`.
    Если колонки нет — возвращает пустые строки нужной длины.
    """
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df), index=df.index)


def dedup_batch_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedup внутри одного файла/батча.
    Ключ: (source + link) или (source + title + published_at) как fallback.

    Выбираем лучшую запись по:
    1) есть ли body/raw_text
    2) длина clean_text
    3) published_at (последняя)
    """
    out = df.copy()

    def _dedup_key(row: pd.Series) -> str:
        src = str(row.get("source") or "")
        link = str(row.get("link") or "")
        if link:
            return f"{src}|{link}"
        title = str(row.get("title") or "")
        published = str(row.get("published_at") or "")
        return f"{src}|{title}|{published}"

    out["dedup_key"] = out.apply(_dedup_key, axis=1)

    clean_text_s = _safe_col_series(out, "clean_text").fillna("").astype(str)
    raw_text_s = _safe_col_series(out, "raw_text").fillna("").astype(str)
    body_s = _safe_col_series(out, "body").fillna("").astype(str)

    out["clean_len"] = clean_text_s.str.len()
    out["has_body"] = (raw_text_s.str.len() > 0) | (body_s.str.len() > 0)

    # для стабильного sort используем mergesort
    out["published_at_dt"] = pd.to_datetime(out.get("published_at"), errors="coerce")

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


def filter_low_quality_for_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Отсекаем записи, которые не стоит отправлять в gold:
    - явно помеченные как 'too_short' / 'parsed_empty' / etc. (если колонка body_status есть)
    - с слишком коротким clean_text (страница/парсер вернули мало контента)

    Функция безопасна: если body_status нет — фильтрация идёт только по длине clean_text.
    """
    out = df.copy()
    before = len(out)

    if "body_status" in out.columns:
        mask_bad = out["body_status"].astype(str).str.strip().isin(EXCLUDE_BODY_STATUS)
        dropped = int(mask_bad.sum())
        if dropped:
            out = out.loc[~mask_bad].copy()
            print(f"[INFO] Drop by body_status: {dropped} rows (excluded={sorted(EXCLUDE_BODY_STATUS)})")

    clean_text_s = _safe_col_series(out, "clean_text").fillna("").astype(str)
    clean_len = clean_text_s.str.len()
    mask_short = clean_len < MIN_CLEAN_TEXT_CHARS
    dropped = int(mask_short.sum())
    if dropped:
        out = out.loc[~mask_short].copy()
        print(f"[INFO] Drop by clean_text length: {dropped} rows (min_len={MIN_CLEAN_TEXT_CHARS})")

    after = len(out)
    print(f"[INFO] Quality filter before gold: {before} -> {after} (dropped {before - after})")
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

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

    # 2) dedup внутри батча ДО обогащения
    df_silver = dedup_batch_silver(df_silver)

    # 3) фильтрация низкокачественных записей (too_short/empty)
    df_silver = filter_low_quality_for_gold(df_silver)

    # output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = infer_output_path(input_path)

    # 4) если после фильтра пусто — не вызываем enrichment
    if df_silver.empty:
        print("[WARN] После фильтрации quality не осталось строк. Сохраняем пустой gold.")
        df_gold = df_silver.copy()
        if "summary" not in df_gold.columns:
            df_gold["summary"] = pd.Series(dtype="string")
        if "keywords" not in df_gold.columns:
            df_gold["keywords"] = pd.Series(dtype="object")
    else:
        df_gold = enrich_articles_with_summary_and_keywords(df_silver)

    print(f"gold:  {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_gold.to_parquet(output_path, index=False)

    if args.upload_minio:
        import subprocess

        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        bucket = os.getenv("MINIO_BUCKET", "media-intel")
        alias = os.getenv("MINIO_ALIAS", "local")

        if not minio_access_key or not minio_secret_key:
            raise ValueError(
                "Не заданы MINIO_ACCESS_KEY/MINIO_SECRET_KEY. Заполни .env (см. .env.example) и повтори."
            )

        try:
            subprocess.run(["mc", "--version"], check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise FileNotFoundError("Команда 'mc' не найдена. Установи MinIO Client (mc) и повтори.") from e

        out = subprocess.run(["mc", "alias", "list"], check=True, capture_output=True, text=True).stdout
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