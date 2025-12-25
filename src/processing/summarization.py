"""
Text enrichment: extractive summary + TF-IDF keywords (RU-friendly).

Goals:
- keep dependencies minimal (scikit-learn + pymorphy3);
- be robust on short/dirty news texts;
- produce deterministic output for pipelines (raw/silver/gold).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Basic RU stopwords (small but pragmatic)
# ----------------------------
RU_STOPWORDS = {
    "и", "в", "во", "на", "не", "что", "это", "как", "а", "но", "или", "то",
    "у", "к", "по", "из", "за", "от", "до", "над", "под", "при", "про",
    "для", "без", "с", "со", "об", "о", "обо", "же", "ли", "бы", "быть",
    "он", "она", "оно", "они", "мы", "вы", "я", "ты", "его", "ее", "их",
    "этот", "эта", "эти", "тот", "та", "те", "такой", "такая", "такие",
    "там", "тут", "здесь", "вот", "уже", "еще", "ещё", "все", "всё",
    "только", "лишь", "очень", "сам", "сама", "сами",
    "который", "которая", "которые", "которых", "которому", "которым",
    "так", "же", "ни", "да", "нет",
    "сегодня", "вчера", "завтра",
}

# ----------------------------
# Domain stopwords: frequent "news noise" words (esp. Telegram/short news)
# Keep it small; extend empirically from your corpus.
# ----------------------------
DOMAIN_STOPWORDS = {
    "конец", "близкий", "похоже", "кажется",
    "ранее", "сейчас", "сегодня", "вчера", "завтра",
    "сообщить", "сообщаться", "заявить", "заявлять", "рассказать",
    "отметить", "уточнить", "добавить",
    "якобы", "возможно", "вероятно",
    "стало", "стать",
    "дневной", "облёт", "облет", "пересмешник", "щебетать",
    "канал", "телеграм", "telegram", "вечер",
    "слово", "утренний", "полдень"
}


def _norm_word(w: str) -> str:
    """Normalize for stable stopword matching: lower + ё→е."""
    return w.lower().replace("ё", "е")


RU_STOPWORDS_NORM = {_norm_word(w) for w in RU_STOPWORDS}
DOMAIN_STOPWORDS_NORM = {_norm_word(w) for w in DOMAIN_STOPWORDS}

# Extra forms that often appear after lemmatization / different spellings
DOMAIN_STOPWORDS_NORM.update(
    {_norm_word(w) for w in ["дневный", "дневной", "кремлевский", "кремлёвский"]}
)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}", re.UNICODE)

# Sentence split: keep it simple but stable.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])")

_morph = MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _lemma_and_pos(token_lower: str) -> Tuple[str, str]:
    """
    Return (lemma, POS) for a token using pymorphy3.
    Cached for speed because news corpora repeat words a lot.
    """
    p = _morph.parse(token_lower)
    if not p:
        return token_lower, ""
    best = p[0]
    lemma = best.normal_form or token_lower
    pos = str(best.tag.POS) if best.tag else ""
    return lemma, pos


def normalize_for_tfidf(
    text: str,
    *,
    keep_pos: Optional[Sequence[str]] = ("NOUN", "ADJF", "ADJS"),
    stopwords: Optional[set] = None,
) -> str:
    """
    Turn raw text into a space-separated string of lemmas.

    - tokenizes (letters only, len>=2)
    - lowercases
    - lemmatizes (pymorphy3)
    - optional POS filtering (by pymorphy3 tag.POS)
    - stopwords filtering

    Returns a string for vectorizers.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Support custom stopwords, but match them normalized
    custom_sw_norm = {_norm_word(w) for w in stopwords} if stopwords else None

    tokens = _TOKEN_RE.findall(text.lower())

    lemmas: List[str] = []
    for tok in tokens:
        lemma, pos = _lemma_and_pos(tok)
        lemma_norm = _norm_word(lemma)

        # stopwords (normalized)
        if custom_sw_norm is not None:
            if lemma_norm in custom_sw_norm:
                continue
        else:
            if lemma_norm in RU_STOPWORDS_NORM:
                continue

        # domain noise stopwords (normalized)
        if lemma_norm in DOMAIN_STOPWORDS_NORM:
            continue

        if keep_pos is not None:
            if pos and pos not in keep_pos:
                continue

        # drop very short lemmas after normalization
        if len(lemma_norm) < 2:
            continue

        lemmas.append(lemma_norm)

    return " ".join(lemmas)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    We don't try to be perfect Russian NLP here; we want predictable behavior in pipeline.
    """
    if not isinstance(text, str):
        return []
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    parts = _SENT_SPLIT_RE.split(text)
    # final cleanup
    return [p.strip() for p in parts if p and p.strip()]


@dataclass(frozen=True)
class SummaryConfig:
    max_sentences: int = 3
    max_chars: int = 700
    redundancy_threshold: float = 0.65  # cosine sim; lower => more diverse


def extractive_summary(text: str, cfg: SummaryConfig = SummaryConfig()) -> str:
    """
    Extractive summary using sentence TF-IDF scoring + redundancy control.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    # Short texts: return as-is (but clipped)
    if len(sentences) <= cfg.max_sentences:
        s = " ".join(sentences)
        return s[: cfg.max_chars].strip()

    # Build sentence-level TF-IDF inside a single document
    norm_sents = [normalize_for_tfidf(s, keep_pos=None) for s in sentences]
    # In case everything got filtered out
    if sum(1 for s in norm_sents if s.strip()) == 0:
        s = " ".join(sentences[: cfg.max_sentences])
        return s[: cfg.max_chars].strip()

    vect = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vect.fit_transform(norm_sents)  # (n_sent, vocab)

    # Score sentence by total TF-IDF mass; add mild bias to early sentences
    base = np.asarray(X.sum(axis=1)).ravel()
    position_bonus = np.linspace(1.0, 0.7, num=len(sentences))  # early sentences slightly preferred
    scores = base * position_bonus

    # Select top sentences with redundancy filtering
    order = np.argsort(-scores)

    chosen: List[int] = []
    for idx in order:
        if len(chosen) >= cfg.max_sentences:
            break

        if not chosen:
            chosen.append(int(idx))
            continue

        sims = cosine_similarity(X[idx], X[chosen]).ravel()
        if np.max(sims) < cfg.redundancy_threshold:
            chosen.append(int(idx))

    # If redundancy blocked too much, fill by best remaining
    if len(chosen) < cfg.max_sentences:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
                if len(chosen) >= cfg.max_sentences:
                    break

    chosen_sorted = sorted(chosen)
    summary = " ".join(sentences[i] for i in chosen_sorted).strip()

    return summary[: cfg.max_chars].strip()


@dataclass(frozen=True)
class KeywordsConfig:
    top_k: int = 8
    ngram_range: Tuple[int, int] = (1, 3)
    max_features: int = 50_000
    max_df: float = 0.85
    min_df: int = 1
    # post-filtering
    min_term_len: int = 3
    max_terms: int = 8


def _postfilter_terms(terms: List[str], cfg: KeywordsConfig) -> List[str]:
    """
    Post-filter candidate terms:
    - drop too short
    - drop terms that start/end with stopwords
    - drop terms containing domain stopwords (noise)
    - drop junk patterns like "слово X"
    - de-duplicate and avoid "nested" keywords:
      if a longer phrase is selected, drop single-word/sub-phrases contained in it.
    - if too few phrases remain, softly backfill with single-word terms (non-nested)
    """
    cleaned: List[str] = []
    seen = set()

    for t in terms:
        t = t.strip()
        if not t:
            continue
        if len(t) < cfg.min_term_len:
            continue

        # drop junk patterns like "слово X"
        if t.startswith("слово "):
            continue

        words = t.split()
        if not words:
            continue

        w0 = _norm_word(words[0])
        wL = _norm_word(words[-1])

        # drop if starts/ends with stopwords (normalized)
        if w0 in RU_STOPWORDS_NORM or wL in RU_STOPWORDS_NORM:
            continue

        # drop if contains domain stopwords anywhere (normalized)
        if any(_norm_word(w) in DOMAIN_STOPWORDS_NORM for w in words):
            continue

        if t in seen:
            continue

        seen.add(t)
        cleaned.append(t)

    if not cleaned:
        return []

    # prefer longer / more specific phrases first
    cleaned.sort(key=lambda s: (len(s.split()), len(s)), reverse=True)

    # anti-nesting: keep phrase, drop subphrases / singletons inside it
    selected: List[str] = []
    for t in cleaned:
        t_norm = f" {t} "
        if any(t_norm in f" {s} " for s in selected):
            continue
        selected.append(t)
        if len(selected) >= cfg.max_terms:
            break

    # soft backfill: add non-nested single terms if we have too few
    if len(selected) < cfg.max_terms:
        for t in cleaned:
            if t in selected:
                continue
            if len(t.split()) != 1:
                continue
            t_norm = f" {t} "
            if any(t_norm in f" {s} " for s in selected):
                continue
            selected.append(t)
            if len(selected) >= cfg.max_terms:
                break

    return selected


def extract_keywords_tfidf(texts: Sequence[str], cfg: KeywordsConfig = KeywordsConfig()) -> List[str]:
    """
    Global TF-IDF over a batch of documents, then top-k terms per document.
    Returns list of '; '-joined keywords for each doc.
    """
    norm_texts = [normalize_for_tfidf(t, keep_pos=("NOUN", "ADJF", "ADJS")) for t in texts]

    vect = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
    )
    X = vect.fit_transform(norm_texts)
    feat = np.array(vect.get_feature_names_out())

    results: List[str] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            results.append("")
            continue
        data = row.data
        idxs = row.indices
        order = np.argsort(-data)[: cfg.top_k]
        top_terms = [feat[idxs[j]] for j in order]
        top_terms = _postfilter_terms(top_terms, cfg)
        results.append("; ".join(top_terms))

    return results


def enrich_articles_with_summary_and_keywords(
    df: pd.DataFrame,
    *,
    text_col: str = "clean_text",
    title_col: str = "title",
    summary_col: str = "summary",
    keywords_col: str = "keywords",
    nlp_text_col: str = "nlp_text",
    summary_cfg: SummaryConfig = SummaryConfig(),
    keywords_cfg: KeywordsConfig = KeywordsConfig(),
) -> pd.DataFrame:
    """
    Adds:
      - nlp_text: title + clean_text (for stable NLP input)
      - summary: extractive summary
      - keywords: TF-IDF keywords (batch-level)

    Also adds helper features for compatibility:
      - text_length_chars
      - num_sentences
      - num_keywords

    Returns a COPY of df (no inplace modifications).
    """
    out = df.copy()

    title = out[title_col].fillna("").astype(str) if title_col in out.columns else ""
    body = out[text_col].fillna("").astype(str) if text_col in out.columns else ""

    out[nlp_text_col] = (title + ". " + body).str.strip()

    # summary per row
    out[summary_col] = out[nlp_text_col].apply(lambda t: extractive_summary(t, summary_cfg))

    # keywords: batch-level TF-IDF (better for trends + comparability)
    out[keywords_col] = extract_keywords_tfidf(out[nlp_text_col].tolist(), keywords_cfg)

    # --- helper features (compat with ClickHouse schema / old pipeline) ---
    out["text_length_chars"] = out[nlp_text_col].fillna("").astype(str).apply(len)
    out["num_sentences"] = out[nlp_text_col].fillna("").astype(str).apply(
        lambda txt: len(split_into_sentences(txt))
    )
    out["num_keywords"] = out[keywords_col].apply(
        lambda s: 0
        if not isinstance(s, str) or not s.strip()
        else len([x for x in s.split(";") if x.strip()])
    )

    return out