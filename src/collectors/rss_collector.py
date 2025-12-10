from __future__ import annotations

from pathlib import Path
from datetime import datetime
from src.utils.s3_client import upload_json_bytes, MINIO_BUCKET

import feedparser
import pandas as pd



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
        item = {
            # id: сначала пробуем явный id, если нет — используем ссылку
            "id": entry.get("id") or entry.get("link"),
            "title": entry.get("title"),
            "link": entry.get("link"),
            # published_at: в RSS бывает published или updated
            "published_at": entry.get("published") or entry.get("updated"),
            "source": source_name,
            # raw_text: берём summary / description, что есть
            "raw_text": entry.get("summary")
            or entry.get("description")
            or "",
        }
        items.append(item)

    df = pd.DataFrame(items)
    return df


def save_dataframe_to_json(
    df: pd.DataFrame,
    folder: str = "data/raw",
) -> Path:
    """
    Сохраняет DataFrame в JSON-файл вида
    data/raw/articles_YYYYMMDD_HHMMSS.json
    и возвращает путь к файлу.
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = folder_path / f"articles_{ts}.json"

    # orient="records" — список объектов
    # force_ascii=False — чтобы кириллица не была \u043f\u0440\u0438...
    df.to_json(filename, orient="records", force_ascii=False)

    return filename

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

def main():
    print(f"Fetching RSS from {RSS_URL} ...")
    feed = fetch_rss()

    print("Converting feed to DataFrame ...")
    df = feed_to_dataframe(feed)

    print(f"Got {len(df)} items")
    if df.empty:
        print("No items to save, exiting.")
        return

    # 1) Сохраняем в MinIO (raw-зона Data Lake)
    print("Saving raw data to MinIO (S3)...")
    key = save_raw_to_s3(df, source_name=SOURCE_NAME)
    print(f"Saved {len(df)} items to s3://{MINIO_BUCKET}/{key}")

    # 2) По желанию — локальный бэкап в data/raw
    print("Saving local backup JSON ...")
    filename = save_dataframe_to_json(df)
    print(f"Saved local copy with {len(df)} items to {filename}")


if __name__ == "__main__":
    main()