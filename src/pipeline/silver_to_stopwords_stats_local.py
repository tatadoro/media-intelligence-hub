from __future__ import annotations

import argparse
from pathlib import Path
import json
from urllib.parse import urlparse

import pandas as pd

from src.processing.stopwords_miner import MinerConfig, mine_tf_df, mine_tf_df_by_source
from src.processing.stopwords_candidates import CandidateConfig, build_candidates
from src.processing.stopwords_build import (
    BuildStopwordsConfig,
    build_stopwords_ru_v001,
    write_stopwords,
)
from src.processing.source_stopwords import SourceStopCfg, source_specific_terms
from src.processing.domain_stopwords import DomainStopCfg, domain_specific_terms


def _source_type_from_source(s: str) -> str:
    """
    Превращаем source из silver в тип источника, чтобы сравнивать TG vs RSS.
    Примеры:
      telegram:ostorozhno_novosti -> telegram
      rss:meduza -> rss
      tvzvezda.ru -> other
    """
    s = (s or "").lower()
    if s.startswith("telegram:"):
        return "telegram"
    if s.startswith("rss:"):
        return "rss"
    return "other"


def iter_silver_docs(paths, text_field: str = "clean_text", lang_field: str = "lang"):
    """
    Yield: (source, source_type, lang, clean_text)

    source ожидаем уже нормализованным в silver:
      - telegram:<channel>
      - rss:<domain>
    Если почему-то source "сырой" (например gazeta.ru) — попытаемся нормализовать.
    """
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for it in items:
            if not isinstance(it, dict):
                continue

            lang = (
                it.get(lang_field)
                or it.get("language")
                or it.get("detected_lang")
                or it.get("lang_code")
                or "unknown"
            )

            src = str(it.get("source") or "").strip().lower()

            # 1) Если в записи уже есть source_type (например "rss"), доверяем ему.
            st = str(it.get("source_type") or "").strip().lower()
            if st in {"rss", "telegram"}:
                source_type = st
            else:
                source_type = _source_type_from_source(src)

            # 2) Эвристика: если source_type == other, но source выглядит как домен — это RSS.
            #    Пример: старые silver содержат source="gazeta.ru" без префикса rss:
            if source_type == "other":
                if src and "." in src and not src.startswith(("rss:", "telegram:")):
                    source_type = "rss"

            # 2b) Эвристика по link: если source неинформативен, берём домен из URL и считаем RSS
            if source_type == "other":
                link = str(it.get("link") or "").strip()
                if link:
                    netloc = urlparse(link).netloc.lower()
                    netloc = netloc[4:] if netloc.startswith("www.") else netloc
                    if netloc and "." in netloc:
                        source_type = "rss"
                        if not src:
                            src = netloc

            # 3) Нормализуем source как ключ (для доменов RSS и каналов TG)
            #    Если source пустой — подставим source_type, чтобы не ломать агрегацию
            source = src if src else source_type

            # 4) Принудительно добавляем префиксы, чтобы всё было единообразно
            if source_type == "rss" and not source.startswith("rss:"):
                source = f"rss:{source}"
            if source_type == "telegram" and not source.startswith("telegram:"):
                source = f"telegram:{source}"

            txt = it.get(text_field) or ""
            yield (source, source_type, str(lang), txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-glob", default="data/silver/articles_*_clean.json")
    ap.add_argument("--out-dir", default="data/nlp/stopwords")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--min-clean-chars", type=int, default=400)
    ap.add_argument("--no-lemmatize-ru", action="store_true")
    args = ap.parse_args()

    silver_paths = sorted(Path(".").glob(args.silver_glob))
    if not silver_paths:
        raise SystemExit(f"No files matched glob: {args.silver_glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MinerConfig(
        min_clean_chars=args.min_clean_chars,
        do_lemmatize_ru=not args.no_lemmatize_ru,
        max_docs=None if args.max_docs == 0 else args.max_docs,
    )

    # Читаем silver один раз (чтобы использовать и для общего подсчёта, и для подсчёта по источникам)
    docs = list(iter_silver_docs(silver_paths))

    # -----------------------------
    # 1) Общая статистика (lang, term)
    # -----------------------------
    docs_lang = ((lang, txt) for (_source, _source_type, lang, txt) in docs)
    stats = mine_tf_df(docs_lang, cfg)

    parquet_path = out_dir / "stats_all_langs.parquet"
    stats.to_parquet(parquet_path, index=False)

    if stats.empty:
        print("[WARN] Stats is empty: no docs passed filters (min_clean_chars?).")
        return

    stats["lang"] = stats["lang"].fillna("unknown").astype(str)

    # ru-preview: если нет ru, но корпус весь unknown — используем unknown как ru-like
    mask_ru = stats["lang"].str.startswith("ru", na=False)
    ru = stats.loc[mask_ru].copy()

    if ru.empty:
        only_unknown = stats["lang"].nunique() == 1 and (stats["lang"].iloc[0] == "unknown")
        if only_unknown:
            print("[WARN] No ru lang in silver; using unknown as ru-like preview.")
            ru = stats.loc[stats["lang"] == "unknown"].copy()

    preview_path = out_dir / "preview_ru_top.csv"
    ru.head(500).to_csv(preview_path, index=False)

    print(f"[OK] Wrote: {parquet_path}")
    print(f"[OK] Wrote: {preview_path}")
    print("[INFO] Top ru-like terms by df_ratio/tf:")
    print(ru.head(20).to_string(index=False))

    # Кандидаты общих стоп-слов (ru-like)
    cand_cfg = CandidateConfig(df_ratio_min=0.60, tf_min=50, max_terms=2000)
    cands = build_candidates(ru, cand_cfg)
    cands_path = out_dir / "candidates_ru_like.csv"
    cands.to_csv(cands_path, index=False)
    print(f"[OK] Wrote: {cands_path}")
    print("[INFO] Top candidates:")
    print(cands.head(30).to_string(index=False))

    # -----------------------------
    # 2) Статистика по источникам (source + source_type)
    # -----------------------------
    stats_src = mine_tf_df_by_source(docs, cfg)
    parquet_src_path = out_dir / "stats_by_source.parquet"
    stats_src.to_parquet(parquet_src_path, index=False)
    print(f"[OK] Wrote: {parquet_src_path}")

    # -----------------------------
    # 2a) Source-specific кандидаты TG vs RSS (по source_type)
    # -----------------------------
    present_types = set(stats_src["source_type"].astype(str).unique().tolist())
    need = {"telegram", "rss"}

    # Заготовим пустые DF на случай, если сравнить нечего
    tg_terms = pd.DataFrame(columns=["term", "tf_hi", "df_ratio_hi", "tf_lo", "df_ratio_lo"])
    rss_terms = pd.DataFrame(columns=["term", "tf_hi", "df_ratio_hi", "tf_lo", "df_ratio_lo"])

    if not need.issubset(present_types):
        print(f"[WARN] Not enough sources for TG/RSS comparison. present_sources={sorted(present_types)}")
    else:
        cfg_src = SourceStopCfg(df_ratio_hi=0.25, df_ratio_lo=0.03, tf_min=30, max_terms=500)

        # Важно: source_specific_terms ожидает stats, где есть source_type/lang/term/tf/df/df_ratio
        # Оно агрегирует по source_type.
        tg_terms = source_specific_terms(stats_src, src_hi="telegram", src_lo="rss", cfg=cfg_src, lang="unknown")
        rss_terms = source_specific_terms(stats_src, src_hi="rss", src_lo="telegram", cfg=cfg_src, lang="unknown")

    tg_csv = out_dir / "candidates_source_tg.csv"
    rss_csv = out_dir / "candidates_source_rss.csv"
    tg_terms.to_csv(tg_csv, index=False)
    rss_terms.to_csv(rss_csv, index=False)
    print(f"[OK] Wrote: {tg_csv}")
    print(f"[OK] Wrote: {rss_csv}")

    # Пишем авто blacklists в config
    def _write_terms_txt(p: Path, terms: list[str]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(terms) + ("\n" if terms else ""), encoding="utf-8")

    tg_auto_path = Path("config/stopwords/blacklist_ru_source_tg_auto.txt")
    rss_auto_path = Path("config/stopwords/blacklist_ru_source_rss_auto.txt")

    _write_terms_txt(tg_auto_path, tg_terms.get("term", pd.Series([], dtype=str)).astype(str).str.lower().tolist())
    _write_terms_txt(rss_auto_path, rss_terms.get("term", pd.Series([], dtype=str)).astype(str).str.lower().tolist())

    print(f"[OK] Updated: {tg_auto_path} ({len(tg_terms)} terms)")
    print(f"[OK] Updated: {rss_auto_path} ({len(rss_terms)} terms)")

    # -----------------------------
    # 2b) Domain-specific RSS кандидаты: rss:<domain> vs остальной RSS
    # -----------------------------
    rss_sources = (
        stats_src.loc[stats_src["source_type"].astype(str) == "rss", "source"]
        .astype(str)
        .value_counts()
    )

    if rss_sources.empty:
        print("[WARN] No RSS sources found for domain-specific stopwords.")
    else:
        top_rss_sources = rss_sources.head(20).index.tolist()

        dom_cfg = DomainStopCfg(df_ratio_hi=0.25, df_ratio_lo=0.03, tf_min=20, max_terms=200)

        rss_out_dir = Path("config/stopwords/rss_domains")
        rss_out_dir.mkdir(parents=True, exist_ok=True)

        for dom in top_rss_sources:
            cand = domain_specific_terms(stats_src, domain_source=str(dom), cfg=dom_cfg, lang="unknown")

            safe_dom = str(dom).replace("rss:", "").replace("/", "_")

            # CSV для дебага
            cand_csv = out_dir / f"candidates_rss_domain_{safe_dom}.csv"
            cand.to_csv(cand_csv, index=False)

            # TXT auto-blacklist
            out_txt_dom = rss_out_dir / f"blacklist_ru_rss_{safe_dom}_auto.txt"
            terms = cand["term"].astype(str).str.lower().tolist() if not cand.empty else []
            out_txt_dom.write_text("\n".join(terms) + ("\n" if terms else ""), encoding="utf-8")

        print(f"[OK] Wrote domain RSS stopwords to: {rss_out_dir}")

    # -----------------------------
    # 3) Итоговый стоп-лист (пока v002 = v001 по сборке; позже подключим auto в build)
    # -----------------------------
    cfg_sw = BuildStopwordsConfig()
    final_sw = build_stopwords_ru_v001(cfg_sw)

    out_txt = out_dir / "stopwords_ru_v002.txt"
    write_stopwords(out_txt, final_sw)

    print(f"[OK] Wrote: {out_txt} ({len(final_sw)} terms)")
    print("[INFO] stopwords_ru_v002 head:")
    print("\n".join(final_sw[:30]))


if __name__ == "__main__":
    main()