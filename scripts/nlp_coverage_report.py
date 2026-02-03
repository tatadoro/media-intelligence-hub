from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd
import os


@dataclass
class CoverageMetrics:
    total: int
    lang_ok: int
    keyphrases_ok: int
    sentiment_ok: int
    lang_ok_ratio: float
    keyphrases_ok_ratio: float
    sentiment_ok_ratio: float


def _count_scripts(text: str) -> tuple[int, int]:
    cyr = 0
    lat = 0
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x052F:
            cyr += 1
        elif 0x0041 <= o <= 0x007A:
            lat += 1
    return cyr, lat


def _lang_reason(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    cyr, lat = _count_scripts(t)
    total = cyr + lat
    if total < 15:
        return "short_text"
    if total >= 40:
        cyr_r = cyr / total if total else 0.0
        lat_r = lat / total if total else 0.0
        if 0.3 <= cyr_r <= 0.7 and 0.3 <= lat_r <= 0.7:
            return "mixed_script"
    return "unknown"


def _ratio(ok: int, total: int) -> str:
    return f"{ok}/{total} ({(ok / total):.0%})" if total else "0/0 (0%)"


def _metrics(df: pd.DataFrame) -> CoverageMetrics:
    total = len(df)
    if total == 0:
        return CoverageMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0)
    lang_ok = int(((df["lang"].fillna("").astype(str) != "") & (df["lang"] != "unknown")).sum())
    keyphrases_ok = int((df["keyphrases_ok"].fillna(0).astype(int) > 0).sum())
    sentiment_ok = int((df["sentiment_ok"].fillna(0).astype(int) > 0).sum())
    return CoverageMetrics(
        total,
        lang_ok,
        keyphrases_ok,
        sentiment_ok,
        (lang_ok / total) if total else 0.0,
        (keyphrases_ok / total) if total else 0.0,
        (sentiment_ok / total) if total else 0.0,
    )


def _sample_rows(df: pd.DataFrame, mask, cols: List[str], max_rows: int = 5) -> List[Dict[str, str]]:
    if df.empty:
        return []
    sub = df.loc[mask, cols].head(max_rows)
    out: List[Dict[str, str]] = []
    for _, row in sub.iterrows():
        item = {}
        for c in cols:
            item[c] = "" if pd.isna(row.get(c)) else str(row.get(c))
        out.append(item)
    return out


def build_report(rss_path: Path, tg_path: Path, out_md: Path, out_json: Path) -> None:
    rss = pd.read_parquet(rss_path)
    tg = pd.read_parquet(tg_path)

    rss_m = _metrics(rss)
    tg_m = _metrics(tg)

    # Samples for diagnostics
    rss_lang_bad = _sample_rows(
        rss,
        (rss["lang"].fillna("").astype(str) == "") | (rss["lang"] == "unknown"),
        ["id", "title", "lang", "nlp_text"],
        max_rows=5,
    )
    tg_lang_bad = _sample_rows(
        tg,
        (tg["lang"].fillna("").astype(str) == "") | (tg["lang"] == "unknown"),
        ["id", "title", "lang", "nlp_text"],
        max_rows=5,
    )
    for item in rss_lang_bad:
        item["lang_reason"] = _lang_reason(item.get("nlp_text", ""))
    for item in tg_lang_bad:
        item["lang_reason"] = _lang_reason(item.get("nlp_text", ""))

    rss_sent_bad = _sample_rows(
        rss,
        rss["sentiment_ok"].fillna(0).astype(int) == 0,
        ["id", "title", "sentiment_label", "sentiment_score", "sentiment_confidence", "sentiment_source"],
        max_rows=5,
    )
    tg_sent_bad = _sample_rows(
        tg,
        tg["sentiment_ok"].fillna(0).astype(int) == 0,
        ["id", "title", "sentiment_label", "sentiment_score", "sentiment_confidence", "sentiment_source"],
        max_rows=5,
    )

    rss_kp_bad = _sample_rows(
        rss,
        rss["keyphrases_ok"].fillna(0).astype(int) == 0,
        ["id", "title", "keyphrases"],
        max_rows=5,
    )
    tg_kp_bad = _sample_rows(
        tg,
        tg["keyphrases_ok"].fillna(0).astype(int) == 0,
        ["id", "title", "keyphrases"],
        max_rows=5,
    )

    md_lines: List[str] = []
    md_lines.append(f"# NLP coverage report ({date.today().isoformat()})")
    md_lines.append("")
    md_lines.append("## Inputs")
    md_lines.append(f"- RSS gold: `{rss_path}`")
    md_lines.append(f"- Telegram gold: `{tg_path}`")
    md_lines.append("")
    md_lines.append("## Coverage summary")
    md_lines.append("")
    md_lines.append("RSS:")
    md_lines.append(f"- rows: {rss_m.total}")
    md_lines.append(f"- lang_ok: {_ratio(rss_m.lang_ok, rss_m.total)}")
    md_lines.append(f"- keyphrases_ok: {_ratio(rss_m.keyphrases_ok, rss_m.total)}")
    md_lines.append(f"- sentiment_ok: {_ratio(rss_m.sentiment_ok, rss_m.total)}")
    md_lines.append("")
    md_lines.append("Telegram:")
    md_lines.append(f"- rows: {tg_m.total}")
    md_lines.append(f"- lang_ok: {_ratio(tg_m.lang_ok, tg_m.total)}")
    md_lines.append(f"- keyphrases_ok: {_ratio(tg_m.keyphrases_ok, tg_m.total)}")
    md_lines.append(f"- sentiment_ok: {_ratio(tg_m.sentiment_ok, tg_m.total)}")
    md_lines.append("")
    md_lines.append("## Diagnostics samples")
    md_lines.append("")
    md_lines.append("RSS lang=unknown samples:")
    md_lines.append("```json")
    md_lines.append(json.dumps(rss_lang_bad, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("Telegram lang=unknown samples:")
    md_lines.append("```json")
    md_lines.append(json.dumps(tg_lang_bad, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("RSS sentiment_ok=0 samples:")
    md_lines.append("```json")
    md_lines.append(json.dumps(rss_sent_bad, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("Telegram sentiment_ok=0 samples:")
    md_lines.append("```json")
    md_lines.append(json.dumps(tg_sent_bad, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("RSS keyphrases_ok=0 samples:")
    md_lines.append("```json")
    md_lines.append(json.dumps(rss_kp_bad, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("Telegram keyphrases_ok=0 samples:")
    md_lines.append("```json")
    md_lines.append(json.dumps(tg_kp_bad, ensure_ascii=False, indent=2))
    md_lines.append("```")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    thresholds = {
        "lang_ok_ratio_min": float(os.getenv("NLP_LANG_OK_MIN", "0.9")),
        "keyphrases_ok_ratio_min": float(os.getenv("NLP_KP_OK_MIN", "0.8")),
        "sentiment_ok_ratio_min": float(os.getenv("NLP_SENT_OK_MIN", "0.15")),
    }
    out_json.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "rss": rss_m.__dict__,
                "telegram": tg_m.__dict__,
                "thresholds": thresholds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    rss_path = Path("data/gold/articles_20260201_163246_rss_combined_clean_processed.parquet")
    tg_path = Path("data/gold/articles_20260201_175611_telegram_combined_clean_processed.parquet")

    today = date.today().isoformat()
    out_md = Path("reports") / f"nlp_coverage_report_{today}.md"
    out_json = Path("reports") / f"nlp_coverage_report_{today}.json"

    build_report(rss_path, tg_path, out_md, out_json)
    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()
