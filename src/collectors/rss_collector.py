from __future__ import annotations

from pathlib import Path
from datetime import datetime
from src.utils.s3_client import upload_json_bytes, MINIO_BUCKET
import json
import yaml
import feedparser
import pandas as pd

# Путь к корню проекта: .../media_intel_hub
BASE_DIR = Path(__file__).resolve().parents[2]
SETTINGS_PATH = BASE_DIR / "config" / "settings.yaml"

def load_settings() -> dict:
    """Читает настройки проекта из config/settings.yaml."""
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Один источник на День 2: RSS новостей Lenta.ru
RSS_URL = "https://lenta.ru/rss/news"
SOURCE_NAME = "lenta.ru"


def fetch_rss(url: str = RSS_URL):
    """
    Скачивает RSS-ленту и возвращает объект feedparser.
    """
    feed = feedparser.parse(url)

    if feed.bozo:  # флаг, что при парсинге были ошибки
        # feed.bozo_exception можно залогировать позже
        print("Warning: RSS feed has parsing issues")

    return feed


def feed_to_dataframe(feed, source_name: str = SOURCE_NAME) -> pd.DataFrame:
    """
    Превращает RSS-ленту в pandas.DataFrame
    с минимальным набором полей.
    """
    items = []

    for entry in feed.entries:
        # 1. Пытаемся взять текст: сначала summary/description, если есть
        summary = entry.get("summary") or entry.get("description")

        # 2. Если summary нет или он пустой — берём title как fallback
        raw_text = summary or entry.get("title") or ""

        item = {
            # id: сначала пробуем явный id, если нет — используем ссылку
            "id": entry.get("id") or entry.get("link"),
            "title": entry.get("title"),
            "link": entry.get("link"),
            # published_at: в RSS бывает published или updated
            "published_at": entry.get("published") or entry.get("updated"),
            "source": source_name,
            "raw_text": raw_text,
        }
        items.append(item)

    df = pd.DataFrame(items)
    return df


def save_dataframe_to_json(df: pd.DataFrame, raw_dir_cfg: str | None = None) -> str:
    """
    Сохраняет DataFrame с сырыми статьями в локальный JSON.

    :param df: данные со статьями
    :param raw_dir_cfg: относительный путь к каталогу raw из конфига (например, "data/raw").
                        Если не задан, используется значение по умолчанию "data/raw".
    :return: путь к сохранённому файлу в виде строки
    """
    if raw_dir_cfg is None:
        raw_dir_cfg = "data/raw"

    # Каталог относительно корня проекта
    raw_dir = BASE_DIR / raw_dir_cfg
    raw_dir.mkdir(parents=True, exist_ok=True)

    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = raw_dir / f"articles_{ts_str}.json"

    df.to_json(filename, orient="records", force_ascii=False, indent=2)

    return str(filename)

def save_raw_to_s3(df: pd.DataFrame, source_name: str) -> str:
    """
    Сохраняем сырые статьи в raw-зону Data Lake (MinIO).

    Структура ключа:
    raw/YYYY-MM-DD/source_name/articles_YYYYMMDD_HHMMSS.json
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y%m%d_%H%M%S")

    key = f"raw/{date_str}/{source_name}/articles_{ts_str}.json"

    json_str = df.to_json(orient="records", force_ascii=False, indent=2)
    upload_json_bytes(MINIO_BUCKET, key, json_str)

    return key

def main() -> None:
    # Загружаем настройки
    settings = load_settings()
    storage_cfg = settings.get("storage", {})
    data_cfg = settings.get("data", {})

    raw_backend = storage_cfg.get("raw_backend", "s3")
    raw_dir_cfg = data_cfg.get("raw_dir", "data/raw")

    print(f"[CONFIG] RAW backend = {raw_backend}")
    print(f"Fetching RSS from {RSS_URL} ...")

    feed = fetch_rss()

    print("Converting feed to DataFrame ...")
    df = feed_to_dataframe(feed)

    print(f"Got {len(df)} items")
    if df.empty:
        print("No items to save, exiting.")
        return

    # Ветвление по конфигу: куда писать сырые данные
    if raw_backend in ("s3", "both"):
        print("Saving raw data to MinIO (S3)...")
        key = save_raw_to_s3(df, source_name=SOURCE_NAME)
        print(f"Saved {len(df)} items to s3://{MINIO_BUCKET}/{key}")

    if raw_backend in ("local", "both"):
        print("Saving local backup JSON ...")
        filename = save_dataframe_to_json(df, raw_dir_cfg)
        print(f"Saved local copy with {len(df)} items to {filename}")

    if raw_backend not in ("s3", "local", "both"):
        print(f"[WARN] Unknown raw_backend='{raw_backend}', nothing was saved.")


if __name__ == "__main__":
    main()