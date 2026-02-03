from __future__ import annotations

import os
import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.processing.cleaning import clean_articles_df

REQUIRED_RAW_COLUMNS = [
    "id",
    "title",
    "link",
    "source",
    "published_at",
    "raw_text",
]


def find_latest_raw_file(raw_dir: Path) -> Path:
    """
    Находит самый новый raw-файл формата articles_*.json в каталоге raw_dir.

    Критерий "самый новый" — по времени модификации файла (mtime).
    Если файлов нет, поднимает FileNotFoundError.
    """
    candidates = list(raw_dir.glob("articles_*.json"))
    if not candidates:
        raise FileNotFoundError(f"Не найдено ни одного файла articles_*.json в {raw_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_domain_from_url(url: str) -> str:
    """
    Извлекает домен из URL. Возвращает '' если извлечь не удалось.
    """
    url = (url or "").strip()
    if not url:
        return ""
    try:
        p = urlparse(url)
        return (p.netloc or "").lower().strip()
    except Exception:
        return ""


def _normalize_source_for_silver(rec: dict) -> None:
    """
    Нормализуем поле source в silver, чтобы дальше можно было надёжно
    определять тип источника (telegram vs rss).

    Правило:
    - если source уже telegram:* или rss:* -> не трогаем
    - если есть channel -> telegram:<channel>
    - иначе, если source выглядит как домен (tvzvezda.ru) или можно извлечь домен из link
      -> rss:<domain>
    - если удалось определить тип -> проставляем source_type
    """
    src = str(rec.get("source") or "").strip()
    src_l = src.lower()

    # Уже нормализовано
    if src_l.startswith("telegram:"):
        rec["source_type"] = rec.get("source_type") or "telegram"
        return
    if src_l.startswith("rss:"):
        rec["source_type"] = rec.get("source_type") or "rss"
        return

    # Telegram: если есть channel
    ch = str(rec.get("channel") or "").strip()
    if ch:
        rec["source"] = f"telegram:{ch}"
        rec["source_type"] = "telegram"
        return

    # RSS: source часто доменом (tvzvezda.ru), иначе домен извлечём из link
    link = str(rec.get("link") or rec.get("raw_url") or "").strip()

    domain = ""
    if src and ("." in src) and (" " not in src) and ("/" not in src) and (":" not in src):
        domain = src.lower()
    else:
        domain = _extract_domain_from_url(link)

    if domain:
        rec["source"] = f"rss:{domain}"
        rec["source_type"] = "rss"
        return

    # Фолбек
    rec["source_type"] = rec.get("source_type") or "other"


def transform_raw_to_silver(input_path: Path, output_path: Path) -> None:
    """
    Шаг пайплайна: читаем raw JSON, чистим тексты, сохраняем cleaned JSON.

    input_path  — путь к сырым данным (data/raw/...json)
    output_path — путь к файлу silver (data/silver/...json)
    """
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)

    # Telegram/raw источники иногда приходят без id. Проставим fallback:
    # 1) uid
    # 2) link/raw_url
    # 3) стабильный индексный id
    if "id" not in df.columns or df["id"].isna().all():
        if "uid" in df.columns:
            df["id"] = df["uid"].astype(str)
        elif "link" in df.columns:
            df["id"] = df["link"].astype(str)
        elif "raw_url" in df.columns:
            df["id"] = df["raw_url"].astype(str)
        else:
            df["id"] = [f"raw_{i}" for i in range(len(df))]
    else:
        # дособерём пустые id из uid/link/raw_url
        missing = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
        if missing.any():
            if "uid" in df.columns:
                df.loc[missing, "id"] = df.loc[missing, "uid"].astype(str)
            elif "link" in df.columns:
                df.loc[missing, "id"] = df.loc[missing, "link"].astype(str)
            elif "raw_url" in df.columns:
                df.loc[missing, "id"] = df.loc[missing, "raw_url"].astype(str)

    _validate_required_columns(df, REQUIRED_RAW_COLUMNS, context="raw")

    df_clean = clean_articles_df(
        df,
        text_column="raw_text",  # важно, чтобы совпадало с названием колонки в rss_collector
        new_column="clean_text",
    )

    cleaned_records = df_clean.to_dict(orient="records")

    # Нормализуем source/source_type для устойчивой аналитики (TG vs RSS) на следующих шагах
    for rec in cleaned_records:
        if isinstance(rec, dict):
            _normalize_source_for_silver(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)


def build_output_path_from_input(input_path: Path) -> Path:
    """
    По пути raw-файла строим путь к silver-файлу.

    data/raw/articles_20251209_212948.json
    → data/silver/articles_20251209_212948_clean.json
    """
    filename = input_path.name
    if filename.endswith(".json"):
        filename_clean = filename[:-5] + "_clean.json"
    else:
        filename_clean = filename + "_clean"

    project_root = Path(__file__).resolve().parents[2]
    silver_dir = project_root / "data" / "silver"
    return silver_dir / filename_clean


def _validate_required_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] {context} data is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    empty_cols = [c for c in required if df[c].isna().all()]
    if empty_cols:
        raise ValueError(
            f"[ERROR] {context} data has empty required columns: {empty_cols}. "
            "Ensure the source provides these fields."
        )


def main() -> None:
    """
    Точка входа для шага raw -> silver.

    Запуск:
      - с явным input:
          python -m src.pipeline.clean_raw_to_silver_local \
              --input data/raw/articles_20251211_153500.json

      - без input (возьмёт последний raw-файл в data/raw):
          python -m src.pipeline.clean_raw_to_silver_local
    """
    parser = argparse.ArgumentParser(
        description="Преобразование raw JSON-файла со статьями в silver-слой (добавление clean_text)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help=(
            "Путь к raw JSON-файлу (от корня проекта или абсолютный). "
            "Если не указан, будет использован последний файл в data/raw."
        ),
    )
    parser.add_argument(
        "--upload-minio",
        action="store_true",
        help="Если указан, после сохранения silver-файла загрузить его в MinIO (bucket media-intel, prefix silver/).",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = project_root / input_path
    else:
        input_path = find_latest_raw_file(raw_dir)
        print(f"[INFO] input не указан, используем последний raw-файл: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной raw-файл: {input_path}")

    output_path = build_output_path_from_input(input_path)

    print("[INFO] Преобразуем raw -> silver")
    print(f"       raw:    {input_path}")
    print(f"       silver: {output_path}")

    transform_raw_to_silver(input_path, output_path)

    if args.upload_minio:
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

        out = subprocess.run(
            ["mc", "alias", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        if alias not in out:
            subprocess.run(
                ["mc", "alias", "set", alias, minio_endpoint, minio_access_key, minio_secret_key],
                check=True,
                capture_output=True,
                text=True,
            )

        dst = f"{alias}/{bucket}/silver/{output_path.name}"
        print(f"[INFO] Загружаем silver в MinIO: {dst}")
        subprocess.run(["mc", "cp", str(output_path), dst], check=True)
        print("[OK] Silver загружен в MinIO")


if __name__ == "__main__":
    main()
