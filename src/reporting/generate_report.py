from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # repo root
except Exception:
    pass


def _running_in_docker() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and "docker" in cgroup.read_text(encoding="utf-8", errors="ignore"):
            return True
    except Exception:
        pass
    return False


def _default_host_port() -> Tuple[str, int]:
    # If explicitly set in env -> use it
    if os.getenv("CH_HOST") or os.getenv("CLICKHOUSE_HOST"):
        host = os.getenv("CH_HOST", os.getenv("CLICKHOUSE_HOST", "localhost"))
        port = int(os.getenv("CH_PORT", os.getenv("CLICKHOUSE_PORT", "8123")))
        return host, port

    # If inside docker (airflow container) -> connect by service name
    if _running_in_docker():
        return "clickhouse", 8123

    # Local default: host-mapped port from docker-compose.yml
    return "localhost", 18123


@dataclass
class CHConfig:
    host: str
    port: int
    tz: str
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "CHConfig":
        host, port = _default_host_port()

        tz = os.getenv("CH_TZ", os.getenv("APP_TZ", "Europe/Moscow"))

        database = os.getenv(
            "CH_DATABASE",
            os.getenv(
                "CH_DB",
                os.getenv(
                    "CLICKHOUSE_DB",
                    os.getenv("CLICKHOUSE_DATABASE", "media_intel"),
                ),
            ),
        )

        user = os.getenv("CH_USER", os.getenv("CLICKHOUSE_USER", "admin"))
        password = os.getenv("CH_PASSWORD", os.getenv("CLICKHOUSE_PASSWORD", ""))

        if not password:
            raise RuntimeError(
                "ClickHouse password is not set. Set CH_PASSWORD (or CLICKHOUSE_PASSWORD) in environment / .env."
            )

        return cls(
            host=host,
            port=int(port),
            tz=tz,
            database=database,
            user=user,
            password=password,
        )


def ch_query_tsv(cfg: CHConfig, query: str) -> List[List[str]]:
    url = f"http://{cfg.host}:{cfg.port}/"
    params = {"database": cfg.database, "query": query.strip() + "\nFORMAT TSV"}
    auth = (cfg.user, cfg.password) if (cfg.user or cfg.password) else None

    r = requests.get(url, params=params, auth=auth, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if not text:
        return []
    return [line.split("\t") for line in text.splitlines()]


def safe_filename(dt: datetime) -> str:
    """
    Стабильный timestamp для имени файла.
    Всегда приводим к UTC, чтобы имя совпадало в контейнере (UTC) и на хосте (локальная TZ).
    """
    if dt.tzinfo is None:
        # fallback: если вдруг передали naive datetime
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%d_%H%M%S_%f")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Markdown report from ClickHouse.")
    p.add_argument("--from", dest="dt_from", default=None, help='Start datetime, e.g. "2025-12-10 14:00:00"')
    p.add_argument("--to", dest="dt_to", default=None, help='End datetime, e.g. "2025-12-10 19:00:00"')
    p.add_argument("--last-hours", dest="last_hours", type=int, default=None, help="Report for last N hours")
    p.add_argument("--top-k", dest="top_k", type=int, default=30, help="Top K keywords/sources")
    p.add_argument(
        "--table",
        dest="table",
        default="articles",
        choices=["articles", "articles_dedup"],
        help="Source table",
    )
    p.add_argument("--outdir", dest="outdir", default=None, help="Output directory (default: reports/)")
    return p.parse_args()


def build_report(
    cfg: CHConfig,
    table: str,
    dt_from: str | None,
    dt_to: str | None,
    last_hours: int | None,
    top_k: int,
) -> str:
    where_parts: List[str] = []

    if dt_from:
        where_parts.append(f"published_at >= toDateTime64('{dt_from}', 9, '{cfg.tz}')")
    if dt_to:
        where_parts.append(f"published_at <  toDateTime64('{dt_to}', 9, '{cfg.tz}')")
    if last_hours is not None:
        where_parts.append(
            "published_at >= (SELECT max(published_at) FROM {t}) - INTERVAL {h} HOUR".format(
                t=table, h=last_hours
            )
        )
        where_parts.append("published_at <= (SELECT max(published_at) FROM {t})".format(t=table))

    where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    summary_q = f"""
    SELECT
      count() AS articles,
      uniqExact(source) AS sources,
      min(published_at) AS min_dt,
      max(published_at) AS max_dt,
      round(avg(text_length_chars), 1) AS avg_len_chars,
      round(avg(num_sentences), 2) AS avg_sentences,
      round(avg(num_keywords), 2) AS avg_keywords
    FROM {table}
    {where_sql}
    """
    summary_rows = ch_query_tsv(cfg, summary_q)
    if not summary_rows:
        min_dt = max_dt = "—"
        articles = "0"
        sources = "0"
        avg_len = avg_sent = avg_kw = "—"
    else:
        articles, sources, min_dt, max_dt, avg_len, avg_sent, avg_kw = summary_rows[0]

    sources_q = f"""
    SELECT source, count() AS n
    FROM {table}
    {where_sql}
    GROUP BY source
    ORDER BY n DESC
    LIMIT {top_k}
    """
    sources_rows = ch_query_tsv(cfg, sources_q)

    hourly_q = f"""
    SELECT toStartOfHour(published_at) AS hour, count() AS n
    FROM {table}
    {where_sql}
    GROUP BY hour
    ORDER BY hour
    """
    hourly_rows = ch_query_tsv(cfg, hourly_q)

    prewhere_sql = ""
    if where_parts:
        prewhere_sql = "PREWHERE " + " AND ".join(where_parts)

    keywords_q = rf"""
    WITH
      arrayFilter(x -> lengthUTF8(x) > 1,
        arrayMap(x -> trim(BOTH ' ' FROM x),
          splitByRegexp('[,;|\n]+', lowerUTF8(keywords))
        )
      ) AS kws
    SELECT kw, count() AS n
    FROM {table}
    ARRAY JOIN kws AS kw
    {prewhere_sql}
    WHERE kw != ''
    GROUP BY kw
    ORDER BY n DESC
    LIMIT {top_k}
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
    args = parse_args()
    cfg = CHConfig.from_env()

    outdir = Path(args.outdir or os.getenv("REPORTS_DIR", "reports"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Важно: используем UTC, чтобы имя файла было стабильным и одинаковым в контейнере и на хосте
    now = datetime.now(timezone.utc)
    fname = f"daily_report_{safe_filename(now)}.md"
    outpath = outdir / fname

    report_md = build_report(
        cfg,
        table=args.table,
        dt_from=args.dt_from,
        dt_to=args.dt_to,
        last_hours=args.last_hours,
        top_k=args.top_k,
    )
    outpath.write_text(report_md, encoding="utf-8")
    print(str(outpath))


if __name__ == "__main__":
    main()