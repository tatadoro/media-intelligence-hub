#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def iter_silver_docs(path: str) -> Iterable[Dict[str, Any]]:
    """
    Поддерживаем два формата:
    1) list[dict]
    2) dict{"items": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                yield x
        return

    if isinstance(data, dict):
        items = data.get("items", [])
        if isinstance(items, list):
            for x in items:
                if isinstance(x, dict):
                    yield x
        return


def get_text(doc: Dict[str, Any]) -> str:
    # silver обычно хранит clean_text; на всякий случай фолбэки
    return (
        doc.get("clean_text")
        or doc.get("text")
        or doc.get("content")
        or ""
    )


def get_lang(doc: Dict[str, Any]) -> str:
    # ожидаем, что в silver есть lang; иначе грубая эвристика
    lang = (doc.get("lang") or "").lower().strip()
    if lang in ("ru", "en"):
        return lang

    txt = get_text(doc)
    if not txt:
        return "unknown"
    # эвристика: если кириллицы больше, считаем ru
    cyr = sum(1 for ch in txt if "А" <= ch <= "я" or ch in "Ёё")
    lat = sum(1 for ch in txt if "A" <= ch <= "z" or "a" <= ch <= "Z")
    if cyr >= max(20, 2 * lat):
        return "ru"
    if lat >= max(20, 2 * cyr):
        return "en"
    return "unknown"


def is_good_token(t: str, min_len: int, max_len: int) -> bool:
    if not t:
        return False
    if len(t) < min_len or len(t) > max_len:
        return False
    # только буквы (регекс уже дал буквы), но оставим проверку на всякий случай
    if not t.isalpha():
        return False
    return True


def write_stopwords(path: str, words: List[Tuple[str, float, int]], meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated stopwords (corpus-based)\n")
        f.write(f"# generated_at_utc: {meta['generated_at_utc']}\n")
        f.write(f"# silver_files: {meta['silver_files']}\n")
        f.write(f"# docs_total: {meta['docs_total']}\n")
        f.write(f"# lang: {meta['lang']}\n")
        f.write(f"# min_df_ratio: {meta['min_df_ratio']}\n")
        f.write(f"# min_len: {meta['min_len']}, max_len: {meta['max_len']}\n")
        f.write(f"# top_n: {meta['top_n']}\n")
        f.write("# format: one token per line\n\n")
        for w, ratio, df in words:
            f.write(f"{w}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-glob", required=True, help="Glob for silver json files, e.g. data/silver/articles_*_clean.json")
    ap.add_argument("--out-ru", default="config/stopwords_auto_ru.txt")
    ap.add_argument("--out-en", default="config/stopwords_auto_en.txt")
    ap.add_argument("--min-df-ratio", type=float, default=0.60)
    ap.add_argument("--top-n", type=int, default=600)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=20)
    args = ap.parse_args()

    files = sorted(glob.glob(args.silver_glob))
    if not files:
        raise SystemExit(f"[ERROR] no files by glob: {args.silver_glob}")

    # df по языкам
    df_ru: Counter[str] = Counter()
    df_en: Counter[str] = Counter()
    docs_ru = 0
    docs_en = 0

    for fp in files:
        for doc in iter_silver_docs(fp):
            lang = get_lang(doc)
            txt = get_text(doc)
            if not txt:
                continue

            toks = tokenize(txt)
            if not toks:
                continue

            uniq = set(t for t in toks if is_good_token(t, args.min_len, args.max_len))

            if lang == "ru":
                docs_ru += 1
                df_ru.update(uniq)
            elif lang == "en":
                docs_en += 1
                df_en.update(uniq)
            else:
                # unknown игнорируем в авто-стопах (можно пересмотреть позже)
                continue

    def pick(df: Counter[str], n_docs: int) -> List[Tuple[str, float, int]]:
        if n_docs <= 0:
            return []
        out: List[Tuple[str, float, int]] = []
        for w, c in df.items():
            ratio = c / n_docs
            if ratio >= args.min_df_ratio:
                out.append((w, ratio, c))
        # сортировка: выше ratio, потом df, потом слово
        out.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return out[: args.top_n]

    ru_words = pick(df_ru, docs_ru)
    en_words = pick(df_en, docs_en)

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    write_stopwords(
        args.out_ru,
        ru_words,
        {
            "generated_at_utc": ts,
            "silver_files": len(files),
            "docs_total": docs_ru,
            "lang": "ru",
            "min_df_ratio": args.min_df_ratio,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "top_n": args.top_n,
        },
    )
    write_stopwords(
        args.out_en,
        en_words,
        {
            "generated_at_utc": ts,
            "silver_files": len(files),
            "docs_total": docs_en,
            "lang": "en",
            "min_df_ratio": args.min_df_ratio,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "top_n": args.top_n,
        },
    )

    print(f"[OK] wrote: {args.out_ru} (n={len(ru_words)}) docs_ru={docs_ru}")
    print(f"[OK] wrote: {args.out_en} (n={len(en_words)}) docs_en={docs_en}")


if __name__ == "__main__":
    main()