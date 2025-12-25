from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import feedparser
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.s3_client import MINIO_BUCKET, upload_json_bytes

# Путь к корню проекта: .../media_intel_hub
BASE_DIR = Path(__file__).resolve().parents[2]
SETTINGS_PATH = BASE_DIR / "config" / "settings.yaml"


@dataclass(frozen=True)
class HttpCfg:
    timeout: float = 15.0
    retries_total: int = 3
    backoff_factor: float = 0.6


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Читает настройки проекта из config/settings.yaml."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_raw_backend(settings: dict[str, Any]) -> str:
    storage_cfg = settings.get("storage", {}) or {}
    backend = os.getenv("RAW_BACKEND") or storage_cfg.get("raw_backend", "s3")
    return str(backend).strip().lower()


def _get_raw_dir(settings: dict[str, Any]) -> str:
    data_cfg = settings.get("data", {}) or {}
    raw_dir = os.getenv("RAW_DIR") or data_cfg.get("raw_dir", "data/raw")
    return str(raw_dir).strip()


def _get_rss_sources(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Ожидаем структуру:
    sources:
      rss:
        - name: "lenta.ru"
          url: "https://lenta.ru/rss/news"
          enabled: true

    ВАЖНО: возвращаем ВСЕ источники (и enabled=false тоже),
    чтобы --only мог тестировать выключенные в конфиге.
    """
    sources_cfg = settings.get("sources", {}) or {}
    rss_list = sources_cfg.get("rss", []) or []
    return list(rss_list)


def make_session(cfg: HttpCfg) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
        }
    )
    retry = Retry(
        total=cfg.retries_total,
        connect=cfg.retries_total,
        read=cfg.retries_total,
        backoff_factor=cfg.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def fetch_rss_bytes(
    session: requests.Session, url: str, timeout: float
) -> Tuple[bytes, str, int, str, Optional[str]]:
    """
    Скачивает RSS по HTTP и возвращает:
    (content_bytes, final_url, status_code, content_type, error_str).
    Ошибки сети/DNS не падают исключением наружу.
    """
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        return (r.content or b""), r.url, int(r.status_code), r.headers.get("Content-Type", ""), None
    except requests.RequestException as e:
        return b"", url, 0, "", f"{type(e).__name__}: {e}"


def feed_to_dataframe(feed: feedparser.FeedParserDict, source_name: str) -> pd.DataFrame:
    """Превращает RSS-ленту (feedparser) в DataFrame с минимальным набором полей."""
    items: list[dict[str, Any]] = []

    for entry in feed.entries:
        summary = getattr(entry, "summary", None) or getattr(entry, "description", None)
        raw_text = str(summary or getattr(entry, "title", "") or "").strip()

        link = str(getattr(entry, "link", "") or "").strip()
        eid = getattr(entry, "id", None) or link

        published = getattr(entry, "published", None) or getattr(entry, "updated", None) or ""

        items.append(
            {
                "id": str(eid or "").strip(),
                "title": str(getattr(entry, "title", "") or "").strip(),
                "link": link,
                "published_at": str(published).strip(),
                "source": source_name,
                "raw_text": raw_text,
            }
        )

    return pd.DataFrame(items)


def _looks_like_html(content: bytes) -> bool:
    head = (content[:2000] or b"").lower()
    return b"<html" in head or b"<!doctype html" in head


def _strip_tag(tag: str) -> str:
    # {namespace}tag -> tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child_text(item: ET.Element, names: tuple[str, ...]) -> str:
    for el in item.iter():
        if _strip_tag(el.tag) in names:
            return (el.text or "").strip()
    return ""


def _fallback_parse_items_xml(content: bytes) -> list[dict[str, Any]]:
    """
    Fallback, когда feedparser возвращает 0 entries, но RSS на самом деле есть.
    Парсим <item> вручную через xml.etree (без внешних зависимостей).
    """
    if not content or _looks_like_html(content):
        return []

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    out: list[dict[str, Any]] = []
    for el in root.iter():
        if _strip_tag(el.tag) != "item":
            continue

        title = _find_child_text(el, ("title",))
        link = _find_child_text(el, ("link",))
        desc = _find_child_text(el, ("description", "summary"))
        pub = _find_child_text(el, ("pubDate", "published", "updated"))

        if not title and not link:
            continue

        out.append({"title": title, "link": link, "raw_text": desc, "published_at": pub})

    return out


def _fallback_items_to_df(items: list[dict[str, Any]], source_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for it in items:
        link = str(it.get("link") or "").strip()
        rows.append(
            {
                "id": link or "",
                "title": str(it.get("title") or "").strip(),
                "link": link,
                "published_at": str(it.get("published_at") or "").strip(),
                "source": source_name,
                "raw_text": str(it.get("raw_text") or "").strip(),
            }
        )
    return pd.DataFrame(rows)


def save_dataframe_to_json(
    df: pd.DataFrame,
    raw_dir_cfg: str,
    source_name: str,
    now_utc: datetime,
) -> str:
    """Сохраняет DataFrame с сырыми статьями в локальный JSON (data/raw)."""
    raw_dir = BASE_DIR / raw_dir_cfg
    raw_dir.mkdir(parents=True, exist_ok=True)

    ts_str = now_utc.strftime("%Y%m%d_%H%M%S")
    filename = raw_dir / f"{source_name.replace('.', '_')}_{ts_str}.json"

    df.to_json(filename, orient="records", force_ascii=False, indent=2)
    return str(filename)


def save_raw_to_s3(df: pd.DataFrame, source_name: str, now_utc: datetime) -> str:
    """
    Сохраняем сырые статьи в raw-зону Data Lake (MinIO).

    Структура ключа:
    raw/YYYY-MM-DD/source_name/articles_YYYYMMDD_HHMMSS.json
    """
    date_str = now_utc.strftime("%Y-%m-%d")
    ts_str = now_utc.strftime("%Y%m%d_%H%M%S")

    key = f"raw/{date_str}/{source_name}/articles_{ts_str}.json"
    json_str = df.to_json(orient="records", force_ascii=False, indent=2)
    upload_json_bytes(MINIO_BUCKET, key, json_str)
    return key


def collect_one_source(
    session: requests.Session,
    name: str,
    url: str,
    http_cfg: HttpCfg,
) -> pd.DataFrame:
    print(f"\n[RSS] Fetching {name} from {url} ...")

    content, final_url, status, content_type, err = fetch_rss_bytes(session, url, http_cfg.timeout)

    if err:
        print(f"[WARN] Request failed: url={url} ({err})")
        print(f"[WARN] Failed to fetch RSS: status={status}, final_url={final_url}")
        return pd.DataFrame()

    if status >= 400 or not content:
        print(f"[WARN] Failed to fetch RSS: status={status}, final_url={final_url}")
        return pd.DataFrame()

    feed = feedparser.parse(content)

    if getattr(feed, "bozo", False):
        exc = getattr(feed, "bozo_exception", None)
        print(f"[WARN] RSS feed has parsing issues: {url}" + (f" ({exc})" if exc else ""))

    df = feed_to_dataframe(feed, source_name=name)

    # Фикс для “HTTP 200, но entries=0”
    if df.empty:
        items = _fallback_parse_items_xml(content)
        if items:
            print(f"[WARN] feedparser returned 0 entries, used XML fallback: items={len(items)}")
            df = _fallback_items_to_df(items, source_name=name)
        else:
            head = content[:200].decode("utf-8", errors="replace").replace("\n", " ")
            print(
                f"[WARN] Got 0 items. status={status}, content-type={content_type}, final_url={final_url}"
            )
            print(f"[WARN] Response head: {head}")

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect RSS sources -> raw (S3/MinIO or local).")
    p.add_argument("--only", type=str, default=None, help="Collect only one source by name (e.g. rbc.ru)")
    p.add_argument("--check", action="store_true", help="Fetch+parse only; do not save")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    raw_backend = _get_raw_backend(settings)
    raw_dir_cfg = _get_raw_dir(settings)
    sources = _get_rss_sources(settings)

    # Fallback: если sources.rss не задан — оставим поведение “один источник”
    if not sources:
        sources = [{"name": "lenta.ru", "url": "https://lenta.ru/rss/news", "enabled": True}]

    only = args.only.strip() if args.only else None
    if only:
        sources = [s for s in sources if str(s.get("name", "")).strip() == only]

    if not sources:
        print(f"[CONFIG] RAW backend = {raw_backend}")
        if only:
            print(f"[WARN] Source '{only}' not found in config/settings.yaml")
        else:
            print("[WARN] No RSS sources found. Check config/settings.yaml")
        return

    # Логика запуска:
    # - если --only указан: запускаем выбранный источник даже если enabled=false (это “режим теста”)
    # - иначе: запускаем только enabled=true
    if only:
        run_sources = sources
    else:
        run_sources = [s for s in sources if bool(s.get("enabled", True))]

    print(f"[CONFIG] RAW backend = {raw_backend}")
    print(f"[CONFIG] Enabled RSS sources: {len(run_sources)}")

    if not run_sources:
        print("[WARN] No enabled RSS sources found. Check config/settings.yaml")
        return

    now_utc = datetime.now(timezone.utc)
    http_cfg = HttpCfg()
    session = make_session(http_cfg)

    for src in run_sources:
        name = str(src.get("name") or "").strip()
        url = str(src.get("url") or "").strip()
        enabled_flag = bool(src.get("enabled", True))

        if not name or not url:
            print("[WARN] Bad source config (missing name/url), skipping")
            continue

        if only and not enabled_flag:
            print(f"[CONFIG] NOTE: source '{name}' is disabled in settings.yaml, running anyway because --only is set")

        df = collect_one_source(session, name=name, url=url, http_cfg=http_cfg)

        print("[RSS] Converting feed to DataFrame ...")
        print(f"[RSS] Got {len(df)} items")

        if df.empty:
            print(f"[RSS] No items for {name}, skipping save.")
            continue

        if args.check:
            continue

        if raw_backend in ("s3", "both"):
            print("[SAVE] Saving raw data to MinIO (S3)...")
            key = save_raw_to_s3(df, source_name=name, now_utc=now_utc)
            print(f"[SAVE] Saved {len(df)} items to s3://{MINIO_BUCKET}/{key}")

        if raw_backend in ("local", "both"):
            print("[SAVE] Saving local backup JSON ...")
            filename = save_dataframe_to_json(df, raw_dir_cfg, source_name=name, now_utc=now_utc)
            print(f"[SAVE] Saved local copy with {len(df)} items to {filename}")

        if raw_backend not in ("s3", "local", "both"):
            print(f"[WARN] Unknown raw_backend='{raw_backend}', nothing was saved.")
            return


if __name__ == "__main__":
    main()