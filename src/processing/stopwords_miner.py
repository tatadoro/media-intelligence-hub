# src/processing/stopwords_miner.py
from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Optional
import pandas as pd

_WORD_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_TECH_RE = re.compile(r"\b(amp|utm_[a-z_]+|t\.me|telegram)\b", re.IGNORECASE)


@dataclass
class MinerConfig:
    min_clean_chars: int = 400
    min_token_len: int = 2
    keep_token_len_1: Tuple[str, ...] = ("я", "и", "в", "к", "с", "у", "о", "а", "не")  # дальше можно уточнять
    do_lemmatize_ru: bool = True
    max_docs: Optional[int] = None  # None = все


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.replace("ё", "е")
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _HASHTAG_RE.sub(" ", t)
    t = _TECH_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _ru_lemmatize(tokens: List[str]) -> List[str]:
    # Ленивая инициализация, чтобы модуль импортировался даже без pymorphy2
    try:
        import pymorphy2
    except Exception:
        return tokens

    morph = pymorphy2.MorphAnalyzer()
    lemmas = []
    for tok in tokens:
        p = morph.parse(tok)
        if not p:
            continue
        lemmas.append(p[0].normal_form)
    return lemmas


def mine_tf_df(
    docs: Iterable[Tuple[str, str]],
    cfg: MinerConfig,
) -> pd.DataFrame:
    """
    docs: iterable of (lang, clean_text)
    returns: DataFrame with columns: lang, term, tf, df, df_ratio
    """
    tf: Dict[Tuple[str, str], int] = Counter()
    df: Dict[Tuple[str, str], int] = Counter()
    doc_counts: Dict[str, int] = Counter()

    for i, (lang, text) in enumerate(docs, start=1):
        if cfg.max_docs is not None and i > cfg.max_docs:
            break

        if not isinstance(text, str) or len(text) < cfg.min_clean_chars:
            continue

        lang = (lang or "unknown").lower()
        norm = normalize_text(text)
        toks = tokenize(norm)

        # длина токенов + исключения
        filtered = []
        for t in toks:
            if len(t) >= cfg.min_token_len:
                filtered.append(t)
            elif t in cfg.keep_token_len_1:
                filtered.append(t)

        if not filtered:
            continue

        if lang.startswith("ru") and cfg.do_lemmatize_ru:
            terms = _ru_lemmatize(filtered)
        else:
            terms = filtered

        doc_counts[lang] += 1

        uniq = set(terms)

        for term in terms:
            tf[(lang, term)] += 1

        for term in uniq:
            df[(lang, term)] += 1

    rows = []
    for key, tf_val in tf.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue

        lang, term = key
        lang = "unknown" if lang is None else str(lang)
        term = "" if term is None else str(term)

        df_val = int(df.get((lang, term), 0))
        n_docs = int(doc_counts.get(lang, 0))
        df_ratio = (df_val / n_docs) if n_docs else 0.0

        rows.append((lang, term, int(tf_val), df_val, float(df_ratio)))

    out = pd.DataFrame(rows, columns=["lang", "term", "tf", "df", "df_ratio"])
    out.sort_values(["lang", "df_ratio", "tf"], ascending=[True, False, False], inplace=True)
    return out


def mine_tf_df_by_source(
    docs: Iterable[Tuple[str, str, str, str]],
    cfg: MinerConfig,
) -> pd.DataFrame:
    """
    docs: iterable of (source, source_type, lang, clean_text)

    Важно:
    - source ожидаем нормализованным из silver:
        telegram:<channel> или rss:<domain>
      (если вдруг пусто — можно подставлять source_type)

    returns columns: source, source_type, lang, term, tf, df, df_ratio
    """
    tf = Counter()
    df = Counter()
    doc_counts = Counter()

    for i, (source, source_type, lang, text) in enumerate(docs, start=1):
        if cfg.max_docs is not None and i > cfg.max_docs:
            break
        if not isinstance(text, str) or len(text) < cfg.min_clean_chars:
            continue

        source_type = (source_type or "other").lower()
        lang = (lang or "unknown").lower()

        source = (source or "").strip().lower()
        if not source:
            # Фолбек, если в silver нет source (или он пуст)
            source = source_type

        norm = normalize_text(text)
        toks = tokenize(norm)

        filtered = []
        for t in toks:
            if len(t) >= cfg.min_token_len:
                filtered.append(t)
            elif t in cfg.keep_token_len_1:
                filtered.append(t)

        if not filtered:
            continue

        if lang.startswith("ru") and cfg.do_lemmatize_ru:
            terms = _ru_lemmatize(filtered)
        else:
            terms = filtered

        # Счётчик документов внутри конкретного источника
        doc_counts[(source, source_type, lang)] += 1

        uniq = set(terms)
        for term in terms:
            tf[(source, source_type, lang, term)] += 1
        for term in uniq:
            df[(source, source_type, lang, term)] += 1

    rows = []
    for (source, source_type, lang, term), tf_val in tf.items():
        df_val = int(df.get((source, source_type, lang, term), 0))
        n_docs = int(doc_counts.get((source, source_type, lang), 0))
        df_ratio = (df_val / n_docs) if n_docs else 0.0
        rows.append((source, source_type, lang, term, int(tf_val), df_val, float(df_ratio)))

    out = pd.DataFrame(rows, columns=["source", "source_type", "lang", "term", "tf", "df", "df_ratio"])
    out.sort_values(
        ["source_type", "source", "lang", "df_ratio", "tf"],
        ascending=[True, True, True, False, False],
        inplace=True,
    )
    return out