from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
    if name.endswith(".json"):
        stem = name[:-5]
    else:
        stem = name

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
    """
    Пытаемся вытащить текст статьи Lenta несколькими селекторами.
    Если не получается — вернём пустую строку (дальше будет fallback).
    """
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
        r = session.get(url, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.text
    except requests.RequestException:
        return None


def enrich_records(
    records: list[dict[str, Any]],
    cfg: FetchCfg,
    limit: Optional[int] = None,
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

        if raw_text:
            rec["body_status"] = "already_present"
            rec["body_len"] = len(raw_text)
            out.append(rec)
            continue

        if not link:
            rec["body_status"] = "no_link"
            rec["body_len"] = 0
            out.append(rec)
            continue

        html = fetch_html(session, link, cfg.timeout)
        if not html:
            rec["body_status"] = "fetch_failed"
            rec["body_len"] = 0
            out.append(rec)
            time.sleep(cfg.sleep)
            continue

        text = extract_body_lenta(html, cfg)
        if not text:
            text = extract_body_generic(html, cfg)

        rec["raw_text"] = text
        rec["body_status"] = "ok" if text else "parsed_empty"
        rec["body_len"] = len(text)
        out.append(rec)

        time.sleep(cfg.sleep)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обогащение raw-слоя: скачать body статьи по link и сохранить raw_enriched.")
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
    enriched = enrich_records(raw_data, cfg=cfg, limit=args.limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in enriched if r.get("body_status") == "ok")
    empty = sum(1 for r in enriched if r.get("body_status") in {"parsed_empty", "fetch_failed", "no_link"})
    print(f"[OK] Готово. ok={ok}, empty_or_failed={empty}, total={len(enriched)}")

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

        dst = f"{alias}/{bucket}/raw_enriched/{output_path.name}"

        print(f"[INFO] Загружаем enriched raw в MinIO: {dst}")
        subprocess.run(["mc", "cp", str(output_path), dst], check=True)
        print("[OK] Enriched raw загружен в MinIO")


if __name__ == "__main__":
    main()