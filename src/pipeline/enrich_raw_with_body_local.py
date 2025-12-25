from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Если raw_text короче порога — считаем это RSS-тизером и всё равно пытаемся скачать body по link
MIN_RAW_TEXT_CHARS = 400

# Минимальная длина, чтобы считать скачанный body "качественным"
MIN_BODY_TEXT_CHARS = 400


@dataclass
class FetchCfg:
    timeout: float = 10.0
    sleep: float = 0.2
    max_chars: int = 50_000
    bs_parser: str = "html.parser"  # "lxml" если установлен


def find_latest_raw_file(raw_dir: Path) -> Path:
    candidates = list(raw_dir.glob("articles_*.json"))
    if not candidates:
        raise FileNotFoundError(f"Не найдено ни одного файла articles_*.json в {raw_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_output_path_from_input(input_path: Path, limit: Optional[int] = None) -> Path:
    """
    data/raw/articles_20251210_153232.json
    -> data/raw_enriched/articles_20251210_153232_enriched.json
    или (если limit задан)
    -> data/raw_enriched/articles_20251210_153232_enriched_200.json
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "data" / "raw_enriched"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = input_path.name
    stem = name[:-5] if name.endswith(".json") else name

    suffix = f"_enriched_{limit}.json" if limit is not None else "_enriched.json"
    return out_dir / f"{stem}{suffix}"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
    )
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _clean_text_lines(lines: list[str], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for line in lines:
        t = " ".join(line.split())
        if len(t) < 2:
            continue
        parts.append(t)
        total += len(t) + 1
        if total >= max_chars:
            break
    return "\n".join(parts).strip()


def extract_body_lenta(html: str, cfg: FetchCfg) -> str:
    """Пытаемся вытащить текст статьи Lenta несколькими селекторами."""
    soup = BeautifulSoup(html, cfg.bs_parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    selectors = [
        "div.topic-body__content p",
        "div.topic-body__content-text",
        "div.topic__text p",
        "article p",
    ]

    for sel in selectors:
        nodes = soup.select(sel)
        lines = [n.get_text(" ", strip=True) for n in nodes if n.get_text(strip=True)]
        text = _clean_text_lines(lines, cfg.max_chars)
        if len(text) >= 200:
            return text

    return ""


def extract_body_ria(html: str, cfg: FetchCfg) -> str:
    """Пытаемся вытащить текст статьи РИА несколькими селекторами."""
    soup = BeautifulSoup(html, cfg.bs_parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    selectors = [
        "div.article__text p",
        "div.article__body p",
        "div.article__text",
        "article p",
    ]

    for sel in selectors:
        nodes = soup.select(sel)
        lines = [n.get_text(" ", strip=True) for n in nodes if n.get_text(strip=True)]
        text = _clean_text_lines(lines, cfg.max_chars)
        if len(text) >= 200:
            return text

    return ""


def extract_body_tass(html: str, cfg: FetchCfg) -> str:
    soup = BeautifulSoup(html, cfg.bs_parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    selectors = [
        "div.news__content",
        "div.news-content",
        "div.article__content",
        "article",
    ]

    for sel in selectors:
        root = soup.select_one(sel)
        if not root:
            continue

        lines: list[str] = []

        # 1) p
        for p in root.select("p"):
            t = " ".join(p.get_text(" ", strip=True).split())
            if len(t) >= 20:
                lines.append(t)

        # 2) li
        for li in root.select("li"):
            t = " ".join(li.get_text(" ", strip=True).split())
            if len(t) >= 20:
                lines.append(t)

        text = _clean_text_lines(lines, cfg.max_chars)

        # 3) fallback: если p/li не дали нормального текста — берём весь текст контейнера
        if len(text) < 200:
            raw = root.get_text("\n", strip=True)
            raw_lines = [" ".join(x.split()) for x in raw.splitlines()]
            raw_lines = [x for x in raw_lines if len(x) >= 20]
            text = _clean_text_lines(raw_lines, cfg.max_chars)

        if len(text) >= 200:
            return text

    return ""

def extract_body_generic(html: str, cfg: FetchCfg) -> str:
    soup = BeautifulSoup(html, cfg.bs_parser)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    ps = soup.find_all("p")
    lines: list[str] = []
    for p in ps:
        t = p.get_text(" ", strip=True)
        t = " ".join(t.split())
        if len(t) >= 60:
            lines.append(t)

    return _clean_text_lines(lines, cfg.max_chars)


def fetch_html(session: requests.Session, url: str, timeout: float) -> Optional[str]:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=False)
        # редирект — считаем, что контент недоступен без auth
        if 300 <= r.status_code < 400:
            return None
        if r.status_code >= 400:
            return None
        return r.text
    except requests.RequestException:
        return None


def _choose_extractor(url: str):
    host = urlparse(url).netloc.lower()
    if "lenta.ru" in host:
        return extract_body_lenta
    if host.endswith("ria.ru") or ".ria.ru" in host:
        return extract_body_ria
    if host.endswith("tass.ru") or ".tass.ru" in host:
        return extract_body_tass
    return None


def enrich_records(
    records: list[dict[str, Any]],
    cfg: FetchCfg,
    limit: Optional[int] = None,
    force_fetch: bool = False,
) -> list[dict[str, Any]]:
    session = make_session()

    out: list[dict[str, Any]] = []
    n = 0

    for rec in records:
        n += 1
        if limit is not None and n > limit:
            break

        link = str(rec.get("link") or "").strip()
        raw_text = str(rec.get("raw_text") or "").strip()
        body_existing = str(rec.get("body") or "").strip()

        # 1) Если body уже сохранён ранее — пропускаем (если не включён force_fetch)
        if (not force_fetch) and body_existing:
            rec["body_status"] = "already_present"
            rec["body_len"] = len(body_existing)
            out.append(rec)
            continue

        # 2) Если raw_text реально длинный — считаем уже "достаточно" (если не включён force_fetch)
        #    иначе это, скорее всего, RSS-тизер, и нужно качать body
        if (not force_fetch) and raw_text and len(raw_text) >= MIN_RAW_TEXT_CHARS:
            rec["body_status"] = "already_present"
            rec["body_len"] = len(raw_text)
            out.append(rec)
            continue

        # 3) Если raw_text есть, но короткий — сохраняем его отдельно как rss_text
        if raw_text:
            rec.setdefault("rss_text", raw_text)

        if not link:
            rec["body_status"] = "no_link"
            rec["body_len"] = 0
            out.append(rec)
            continue

        html = fetch_html(session, link, cfg.timeout)
        if not html:
            rss_text = str(rec.get("rss_text") or raw_text or "").strip()
            if rss_text and len(rss_text) >= MIN_RAW_TEXT_CHARS:
                rec["raw_text"] = rss_text
                rec["body_status"] = "ok_rss"
                rec["body_len"] = len(rss_text)
            else:
                rec["body_status"] = "fetch_failed"
                rec["body_len"] = 0
            out.append(rec)
            time.sleep(cfg.sleep)
            continue

        extractor = _choose_extractor(link)
        text = extractor(html, cfg) if extractor else ""

        if not text:
            text = extract_body_generic(html, cfg)

        if text and len(text) >= MIN_BODY_TEXT_CHARS:
            # сохраняем и в body, и в raw_text (для downstream clean_raw_to_silver)
            rec["body"] = text
            rec["raw_text"] = text
            rec["body_status"] = "ok"
            rec["body_len"] = len(text)
        elif text:
            # текст вытащили, но он подозрительно короткий: сохраняем, но помечаем как too_short
            rec["body"] = text
            rec["raw_text"] = text
            rec["body_status"] = "too_short"
            rec["body_len"] = len(text)
        else:
            # не затираем rss-тизер (если он был)
            if "raw_text" in rec and isinstance(rec["raw_text"], str) and rec["raw_text"].strip():
                pass
            else:
                rec["raw_text"] = rec.get("rss_text", "")
            rec["body_status"] = "parsed_empty"
            rec["body_len"] = 0

        out.append(rec)
        time.sleep(cfg.sleep)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Обогащение raw-слоя: скачать body статьи по link и сохранить raw_enriched."
    )
    p.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        help="Путь к raw JSON (data/raw/articles_*.json). Если не указан — берём самый свежий в data/raw.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Путь к выходному raw_enriched JSON. Если не указан — сформируется автоматически в data/raw_enriched/.",
    )
    p.add_argument("--limit", type=int, default=None, help="Ограничить число записей (для теста).")
    p.add_argument("--timeout", type=float, default=10.0, help="Timeout для HTTP запросов (сек).")
    p.add_argument("--sleep", type=float, default=0.2, help="Пауза между запросами (сек).")
    p.add_argument("--max-chars", type=int, default=50000, help="Макс. длина сохраняемого текста (символов).")
    p.add_argument(
        "--bs-parser",
        type=str,
        default="html.parser",
        choices=["html.parser", "lxml"],
        help="Парсер BeautifulSoup. html.parser работает без зависимостей; lxml быстрее, но требует установленного lxml.",
    )
    p.add_argument(
        "--force-fetch",
        action="store_true",
        help="Принудительно скачивать body по link даже если raw_text уже есть (полезно для RSS-тизеров).",
    )
    p.add_argument(
        "--upload-minio",
        action="store_true",
        help="Если указан, после сохранения raw_enriched файла загрузить его в MinIO (bucket media-intel, prefix raw_enriched/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = project_root / input_path
    else:
        input_path = find_latest_raw_file(raw_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной raw файл: {input_path}")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = build_output_path_from_input(input_path, limit=args.limit)

    print("[INFO] raw -> raw_enriched")
    print(f"       raw:         {input_path}")
    print(f"       raw_enriched:{output_path}")

    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Ожидали список словарей в raw JSON.")

    cfg = FetchCfg(
        timeout=float(args.timeout),
        sleep=float(args.sleep),
        max_chars=int(args.max_chars),
        bs_parser=str(args.bs_parser),
    )
    enriched = enrich_records(
        raw_data,
        cfg=cfg,
        limit=args.limit,
        force_fetch=bool(args.force_fetch),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in enriched if r.get("body_status") == "ok")
    too_short = sum(1 for r in enriched if r.get("body_status") == "too_short")
    empty = sum(1 for r in enriched if r.get("body_status") in {"parsed_empty", "fetch_failed", "no_link"})
    already = sum(1 for r in enriched if r.get("body_status") == "already_present")
    print(
        f"[OK] Готово. ok={ok}, too_short={too_short}, already_present={already}, "
        f"empty_or_failed={empty}, total={len(enriched)}"
    )

    if args.upload_minio:
        # Проверим, что mc доступен
        try:
            subprocess.run(["mc", "--version"], check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Команда 'mc' не найдена. Установи MinIO Client (mc) и повтори."
            ) from e

        bucket = os.getenv("MINIO_BUCKET", "media-intel")
        alias = os.getenv("MINIO_ALIAS", "local")

        # Если alias уже настроен в mc — ключи не нужны, просто грузим
        aliases_out = subprocess.run(
            ["mc", "alias", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        if alias in aliases_out:
            dst = f"{alias}/{bucket}/raw_enriched/{output_path.name}"
            print(f"[INFO] Загружаем enriched raw в MinIO: {dst}")
            subprocess.run(["mc", "cp", str(output_path), dst], check=True)
            print("[OK] Enriched raw загружен в MinIO")
            return

        # Иначе попробуем создать alias из окружения (если переменные заданы)
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        minio_access_key = (
            os.getenv("MINIO_ACCESS_KEY")
            or os.getenv("MINIO_ROOT_USER")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        minio_secret_key = (
            os.getenv("MINIO_SECRET_KEY")
            or os.getenv("MINIO_ROOT_PASSWORD")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        if not minio_access_key or not minio_secret_key:
            raise ValueError(
                f"mc alias '{alias}' не настроен и не заданы ключи MinIO в окружении. "
                "Сделай одно из двух: "
                "1) настрой alias вручную: mc alias set local http://localhost:9000 <user> <pass> "
                "или 2) экспортируй переменные MINIO_ROOT_USER/MINIO_ROOT_PASSWORD (или MINIO_ACCESS_KEY/MINIO_SECRET_KEY)."
            )

        subprocess.run(
            ["mc", "alias", "set", alias, minio_endpoint, minio_access_key, minio_secret_key],
            check=True,
            capture_output=True,
            text=True,
        )

        dst = f"{alias}/{bucket}/raw_enriched/{output_path.name}"
        print(f"[INFO] Загружаем enriched raw в MinIO: {dst}")
        subprocess.run(["mc", "cp", str(output_path), dst], check=True)
        print("[OK] Enriched raw загружен в MinIO")


if __name__ == "__main__":
    main()