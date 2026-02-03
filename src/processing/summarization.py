"""
Text enrichment: extractive summary + TF-IDF keywords (RU-friendly).

Goals:
- keep dependencies minimal (scikit-learn + pymorphy3);
- be robust on short/dirty news texts;
- produce deterministic output for pipelines (raw/silver/gold).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional NER (B): Natasha. If not installed, we fall back to simple rules (A).
try:
    from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter, MorphVocab  # type: ignore

    _NATASHA_OK = True
except Exception:
    _NATASHA_OK = False


# ----------------------------
# Basic RU stopwords (small but pragmatic)
# ----------------------------
RU_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "не",
    "что",
    "это",
    "как",
    "а",
    "но",
    "или",
    "то",
    "у",
    "к",
    "по",
    "из",
    "за",
    "от",
    "до",
    "над",
    "под",
    "при",
    "про",
    "для",
    "без",
    "с",
    "со",
    "об",
    "о",
    "обо",
    "же",
    "ли",
    "бы",
    "быть",
    "он",
    "она",
    "оно",
    "они",
    "мы",
    "вы",
    "я",
    "ты",
    "его",
    "ее",
    "их",
    "этот",
    "эта",
    "эти",
    "тот",
    "та",
    "те",
    "такой",
    "такая",
    "такие",
    "там",
    "тут",
    "здесь",
    "вот",
    "уже",
    "еще",
    "ещё",
    "все",
    "всё",
    "только",
    "лишь",
    "очень",
    "сам",
    "сама",
    "сами",
    "который",
    "которая",
    "которые",
    "которых",
    "которому",
    "которым",
    "так",
    "же",
    "ни",
    "да",
    "нет",
    "сегодня",
    "вчера",
    "завтра",
}

# Доменные стоп-термины (новости/медиа), которые часто шумят в keywords
DOMAIN_STOP_TERMS = {
    # ru
    "новость",
    "сообщить",
    "сообщать",
    "заявить",
    "заявлять",
    "рассказать",
    "говорить",
    "сказать",
    "пояснить",
    "уточнить",
    "отметить",
    "подчеркнуть",
    "источник",
    "агентство",
    "редакция",
    "интервью",
    "обзор",
    # en
    "news",
    "report",
    "reported",
    "says",
    "said",
    "sources",
    "according",
    "statement",
    "interview",
}

# ----------------------------
# Domain stopwords: frequent "news noise" words (esp. Telegram/short news)
# Keep it small; extend empirically from your corpus.
# ----------------------------
DOMAIN_STOPWORDS = {
    "конец",
    "близкий",
    "похоже",
    "кажется",
    "ранее",
    "сейчас",
    "сегодня",
    "вчера",
    "завтра",
    "сообщить",
    "сообщаться",
    "заявить",
    "заявлять",
    "рассказать",
    "отметить",
    "уточнить",
    "добавить",
    "якобы",
    "возможно",
    "вероятно",
    "стало",
    "стать",
    "дневной",
    "облёт",
    "облет",
    "пересмешник",
    "щебетать",
    "канал",
    "телеграм",
    "telegram",
    "вечер",
    "слово",
    "утренний",
    "полдень",
}


def _norm_word(w: str) -> str:
    """Normalize for stable matching: lower + ё→е."""
    return (w or "").lower().replace("ё", "е")


RU_STOPWORDS_NORM = {_norm_word(w) for w in RU_STOPWORDS}
DOMAIN_STOPWORDS_NORM = {_norm_word(w) for w in DOMAIN_STOPWORDS}
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
    return [p.strip() for p in parts if p and p.strip()]


# ----------------------------
# Entities (B: Natasha NER) + fallback (A: simple rules)
# ----------------------------
_ENTITY_CLEAN_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\- ]+", re.UNICODE)

_CAP_SEQ_RE = re.compile(
    r"\b(?:[А-ЯЁ][а-яё]+|[A-Z][a-z]+)(?:\s+(?:[А-ЯЁ][а-яё]+|[A-Z][a-z]+)){0,2}\b"
)

_FALLBACK_BAD_START = {
    "в",
    "на",
    "это",
    "но",
    "и",
    "по",
    "из",
    "при",
    "для",
    "без",
    "от",
    "до",
}

_GENERIC_ENTITY_SINGLE = {
    # ru
    "президент",
    "премьер",
    "министр",
    "глава",
    "лидер",
    "депутат",
    "сенатор",
    "губернатор",
    "мэр",
    "суд",
    "судья",
    "прокуратура",
    "полиция",
    "ведомство",
    "агентство",
    "комитет",
    "служба",
    "группа",
    "компания",
    "холдинг",
    # en
    "president",
    "prime",
    "minister",
    "leader",
    "governor",
    "mayor",
    "court",
    "police",
    "agency",
    "committee",
    "service",
    "group",
    "company",
    "holding",
}


def normalize_entity(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = _ENTITY_CLEAN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = _norm_word(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


@lru_cache(maxsize=1)
def _get_ner():
    if not _NATASHA_OK:
        return None
    segmenter = Segmenter()
    emb = NewsEmbedding()
    ner = NewsNERTagger(emb)
    vocab = MorphVocab()
    return segmenter, ner, vocab


def extract_entities_ner(text: str, *, max_per_type: int = 10) -> Dict[str, List[str]]:
    out = {"persons": [], "orgs": [], "geo": []}
    if not isinstance(text, str) or not text.strip():
        return out

    ner_pack = _get_ner()
    if ner_pack is None:
        return out

    segmenter, ner, vocab = ner_pack
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner)

    persons: List[str] = []
    orgs: List[str] = []
    geo: List[str] = []

    for span in doc.spans:
        try:
            span.normalize(vocab)
            norm = span.normal
        except TypeError:
        # на случай несовпадений версий
            norm = None

        val = normalize_entity(norm or span.text)
        if not val or len(val) < 2:
            continue
        if val.lower() in _GENERIC_ENTITY_SINGLE:
            continue
        if span.type == "PER":
            persons.append(val)
        elif span.type == "ORG":
            orgs.append(val)
        elif span.type == "LOC":
            geo.append(val)

    out["persons"] = _uniq_keep_order(persons)[:max_per_type]
    out["orgs"] = _uniq_keep_order(orgs)[:max_per_type]
    out["geo"] = _uniq_keep_order(geo)[:max_per_type]
    return out


def extract_entities_fallback(text: str, *, max_per_type: int = 10) -> Dict[str, List[str]]:
    out = {"persons": [], "orgs": [], "geo": []}
    if not isinstance(text, str) or not text.strip():
        return out

    cands = _CAP_SEQ_RE.findall(text)
    cleaned: List[str] = []
    for c in cands:
        c2 = normalize_entity(c)
        if not c2:
            continue
        if c2.lower() in _GENERIC_ENTITY_SINGLE:
            continue
        first = _norm_word(c2.split()[0])
        if first in _FALLBACK_BAD_START:
            continue
        cleaned.append(c2)

    cleaned = _uniq_keep_order(cleaned)
    # Fallback does not reliably classify PER/ORG/LOC. Put into persons for "who" trends.
    out["persons"] = cleaned[:max_per_type]
    return out


def extract_entities(text: str, *, max_per_type: int = 10) -> Dict[str, List[str]]:
    ent = extract_entities_ner(text, max_per_type=max_per_type)
    fallback = extract_entities_fallback(text, max_per_type=max_per_type)

    # Merge: fill empty types from fallback, otherwise union unique (lightly)
    out = {"persons": [], "orgs": [], "geo": []}
    for k in ("persons", "orgs", "geo"):
        items = ent.get(k, [])
        if items:
            merged = _uniq_keep_order(items + fallback.get(k, []))
            out[k] = merged[:max_per_type]
        else:
            out[k] = fallback.get(k, [])[:max_per_type]
    return out


def _sentence_entity_bonus(sentences: List[str], top_entities: List[str]) -> np.ndarray:
    if not sentences or not top_entities:
        return np.zeros(len(sentences), dtype=float)

    ents_norm = [_norm_word(normalize_entity(e)) for e in top_entities if e and normalize_entity(e)]
    ents_norm = [e for e in ents_norm if e]
    if not ents_norm:
        return np.zeros(len(sentences), dtype=float)

    bonuses = np.zeros(len(sentences), dtype=float)
    for i, s in enumerate(sentences):
        s_norm = _norm_word(s)
        hit = 0
        for e in ents_norm:
            if e and e in s_norm:
                hit += 1
        bonuses[i] = float(hit)
    return bonuses


@dataclass(frozen=True)
class SummaryConfig:
    max_sentences: int = 3
    max_chars: int = 700
    redundancy_threshold: float = 0.65  # cosine sim; lower => more diverse
    entity_bonus_weight: float = 0.25   # bonus per entity hit (relative)
    max_entities_for_bonus: int = 6
    min_sentence_chars: int = 25
    min_sentence_tokens: int = 5
    diversity_lambda: float = 0.7       # MMR penalty for similarity
    entity_repeat_penalty: float = 0.15 # penalty per repeated entity


def extractive_summary(text: str, cfg: SummaryConfig = SummaryConfig()) -> str:
    """
    Extractive summary using sentence TF-IDF scoring + redundancy control.
    Plus optional entity coverage bonus (persons/orgs/geo) to make summaries more "news-like".
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    # Filter out low-signal sentences
    def _good_sentence(s: str) -> bool:
        s = s.strip()
        if len(s) < cfg.min_sentence_chars:
            return False
        toks = [m.group(0) for m in _TOKEN_RE.finditer(s)]
        if len(toks) < cfg.min_sentence_tokens:
            return False
        # too much punctuation/noise
        punct = sum(1 for ch in s if ch in "!?:;…")
        if punct >= max(4, len(s) // 25):
            return False
        return True

    filtered = [s for s in sentences if _good_sentence(s)]
    if filtered:
        sentences = filtered

    # Short texts: return as-is (but clipped)
    if len(sentences) <= cfg.max_sentences:
        s = " ".join(sentences)
        return s[: cfg.max_chars].strip()

    # Build sentence-level TF-IDF inside a single document
    norm_sents = [normalize_for_tfidf(s, keep_pos=None) for s in sentences]
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

    # Base score by TF-IDF mass; mild bias to early sentences
    base = np.asarray(X.sum(axis=1)).ravel()
    position_bonus = np.linspace(1.0, 0.7, num=len(sentences))
    scores = base * position_bonus

    # Entity coverage bonus (optional)
    ent = extract_entities(text, max_per_type=10)
    top_entities = (ent["persons"] + ent["orgs"] + ent["geo"])[: cfg.max_entities_for_bonus]
    if top_entities and cfg.entity_bonus_weight > 0:
        hits = _sentence_entity_bonus(sentences, top_entities)
        # normalize hits into [0..1] roughly, then scale by mean TF-IDF
        if hits.max() > 0:
            hits = hits / hits.max()
            scores = scores + cfg.entity_bonus_weight * (scores.mean() + 1e-9) * hits

    order = np.argsort(-scores)

    # MMR selection with entity repeat penalty
    sent_entities: List[List[str]] = []
    for s in sentences:
        ent_s = extract_entities_fallback(s, max_per_type=5).get("persons", [])
        ent_s = [normalize_entity(x).lower() for x in ent_s if x]
        sent_entities.append(ent_s)

    chosen: List[int] = []
    used_entities: set[str] = set()

    for _ in range(min(cfg.max_sentences, len(sentences))):
        best_idx = None
        best_score = None
        for idx in order:
            if int(idx) in chosen:
                continue
            base = scores[idx]
            if chosen:
                sims = cosine_similarity(X[idx], X[chosen]).ravel()
                max_sim = float(np.max(sims)) if sims.size else 0.0
            else:
                max_sim = 0.0

            # penalty for repeated entities
            repeats = sum(1 for e in sent_entities[idx] if e in used_entities)
            pen = cfg.entity_repeat_penalty * repeats

            mmr = float(base) - cfg.diversity_lambda * max_sim - pen
            if best_score is None or mmr > best_score:
                best_score = mmr
                best_idx = int(idx)

        if best_idx is None:
            break
        chosen.append(best_idx)
        for e in sent_entities[best_idx]:
            used_entities.add(e)

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
        if t in DOMAIN_STOP_TERMS:
            continue
        if t.startswith("слово "):
            continue

        words = t.split()
        if not words:
            continue

        w0 = _norm_word(words[0])
        wL = _norm_word(words[-1])

        if w0 in RU_STOPWORDS_NORM or wL in RU_STOPWORDS_NORM:
            continue
        if any(_norm_word(w) in DOMAIN_STOPWORDS_NORM for w in words):
            continue

        if t in seen:
            continue

        seen.add(t)
        cleaned.append(t)

    if not cleaned:
        return []

    cleaned.sort(key=lambda s: (len(s.split()), len(s)), reverse=True)

    selected: List[str] = []
    for t in cleaned:
        t_norm = f" {t} "
        if any(t_norm in f" {s} " for s in selected):
            continue
        selected.append(t)
        if len(selected) >= cfg.max_terms:
            break

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


def _fallback_keywords(texts: Sequence[str], cfg: KeywordsConfig) -> List[str]:
    """
    Simple per-document fallback: pick top tokens by frequency.
    Used when TF-IDF is unstable on tiny batches.
    """
    results: List[str] = []
    for t in texts:
        norm = normalize_for_tfidf(t, keep_pos=("NOUN", "ADJF", "ADJS"))
        toks = [x for x in norm.split() if x]
        if not toks:
            results.append("")
            continue
        counts = Counter(toks)
        cand = [w for w, _ in counts.most_common(cfg.top_k * 5)]
        cand = _postfilter_terms(cand, cfg)
        results.append("; ".join(cand[: cfg.top_k]))
    return results


def extract_keywords_tfidf(texts: Sequence[str], cfg: KeywordsConfig = KeywordsConfig()) -> List[str]:
    """
    Global TF-IDF over a batch of documents, then top-k terms per document.
    Returns list of '; '-joined keywords for each doc.
    """
    norm_texts = [normalize_for_tfidf(t, keep_pos=("NOUN", "ADJF", "ADJS")) for t in texts]

    if len(texts) < 3:
        return _fallback_keywords(texts, cfg)

    max_df = 1.0 if len(texts) < 5 else cfg.max_df

    vect = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=max_df,
    )
    try:
        X = vect.fit_transform(norm_texts)
    except ValueError:
        return _fallback_keywords(texts, cfg)
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

    Adds entities (B: Natasha NER; A: fallback):
      - entities_persons
      - entities_orgs
      - entities_geo
      - num_persons / num_orgs / num_geo

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

    # entities per row
    ents = out[nlp_text_col].apply(lambda t: extract_entities(t, max_per_type=10))
    out["entities_persons"] = ents.apply(lambda d: "; ".join(d.get("persons", [])))
    out["entities_orgs"] = ents.apply(lambda d: "; ".join(d.get("orgs", [])))
    out["entities_geo"] = ents.apply(lambda d: "; ".join(d.get("geo", [])))

    out["num_persons"] = ents.apply(lambda d: len(d.get("persons", [])))
    out["num_orgs"] = ents.apply(lambda d: len(d.get("orgs", [])))
    out["num_geo"] = ents.apply(lambda d: len(d.get("geo", [])))

    # summary per row (now with optional entity bonus)
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
