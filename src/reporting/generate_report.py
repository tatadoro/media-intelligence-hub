from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import requests

from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # корень репо
except Exception:
    # если dotenv не установлен или .env отсутствует — просто работаем дальше
    pass


@dataclass
class CHConfig:
    host: str = os.getenv("CH_HOST", os.getenv("CLICKHOUSE_HOST", "localhost"))
    port: int = int(os.getenv("CH_PORT", os.getenv("CLICKHOUSE_PORT", "18123")))
    database: str = os.getenv("CH_DB", os.getenv("CLICKHOUSE_DB", "media_intel"))
    user: str = os.getenv("CH_USER", os.getenv("CLICKHOUSE_USER", "default"))
    password: str = os.getenv("CH_PASSWORD", os.getenv("CLICKHOUSE_PASSWORD", ""))


def ch_query_tsv(cfg: CHConfig, query: str) -> List[List[str]]:
    """
    Выполняет запрос в ClickHouse через HTTP и возвращает rows как список списков строк.
    Формат ответа — TSV.
    """
    url = f"http://{cfg.host}:{cfg.port}/"
    params = {
        "database": cfg.database,
        "query": query.strip() + "\nFORMAT TSV",
    }
    auth = (cfg.user, cfg.password) if (cfg.user or cfg.password) else None

    r = requests.get(url, params=params, auth=auth, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if not text:
        return []

    rows = []
    for line in text.splitlines():
        rows.append(line.split("\t"))
    return rows


def safe_filename(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H%M")


def build_report(cfg: CHConfig) -> str:
    # 1) Сводка
    summary_q = """
    SELECT
      count() AS articles,
      uniqExact(source) AS sources,
      min(published_at) AS min_dt,
      max(published_at) AS max_dt,
      round(avg(text_length_chars), 1) AS avg_len_chars,
      round(avg(num_sentences), 2) AS avg_sentences,
      round(avg(num_keywords), 2) AS avg_keywords
    FROM articles
    """
    summary = ch_query_tsv(cfg, summary_q)[0]
    articles, sources, min_dt, max_dt, avg_len, avg_sent, avg_kw = summary

    # 2) Топ источников
    sources_q = """
    SELECT source, count() AS n
    FROM articles
    GROUP BY source
    ORDER BY n DESC
    LIMIT 20
    """
    sources_rows = ch_query_tsv(cfg, sources_q)

    # 3) Динамика по часу
    hourly_q = """
    SELECT toStartOfHour(published_at) AS hour, count() AS n
    FROM articles
    GROUP BY hour
    ORDER BY hour
    """
    hourly_rows = ch_query_tsv(cfg, hourly_q)

    # 4) Топ ключевых слов
    keywords_q = r"""
    WITH
      arrayFilter(x -> lengthUTF8(x) > 1,
        arrayMap(x -> trim(BOTH ' ' FROM x),
          splitByRegexp('[,;|\n]+', lowerUTF8(keywords))
        )
      ) AS kws
    SELECT kw, count() AS n
    FROM articles
    ARRAY JOIN kws AS kw
    WHERE kw != ''
    GROUP BY kw
    ORDER BY n DESC
    LIMIT 30
    """
    keywords_rows = ch_query_tsv(cfg, keywords_q)

    lines: List[str] = []
    lines.append("# Media Intelligence Hub — Daily Report")
    lines.append("")
    lines.append(f"**Период данных:** {min_dt} → {max_dt} ({cfg.database})")
    lines.append(f"**Материалов:** {articles}")
    lines.append(f"**Источников:** {sources} (топ см. ниже)")
    lines.append("")
    lines.append("## Сводные метрики текста")
    lines.append(f"- Средняя длина текста: **{avg_len}** символов")
    lines.append(f"- Среднее число предложений: **{avg_sent}**")
    lines.append(f"- Среднее число ключевых слов: **{avg_kw}**")
    lines.append("")
    lines.append("## Топ источников")
    for src, n in sources_rows:
        lines.append(f"- {src} ({n})")
    lines.append("")
    lines.append("## Динамика публикаций по часу")
    for hour, n in hourly_rows:
        lines.append(f"- {hour} — {n}")
    lines.append("")
    lines.append("## Топ ключевых слов")
    for kw, n in keywords_rows:
        lines.append(f"- {kw} ({n})")

    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = CHConfig()

    # Куда сохранять
    outdir = Path(os.getenv("REPORTS_DIR", "reports"))
    outdir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    fname = f"daily_report_{safe_filename(now)}.md"
    outpath = outdir / fname

    report_md = build_report(cfg)
    outpath.write_text(report_md, encoding="utf-8")

    print(str(outpath))


if __name__ == "__main__":
    main()