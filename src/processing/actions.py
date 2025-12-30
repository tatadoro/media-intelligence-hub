from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List

try:
    import pymorphy2

    _MORPH_AVAILABLE = True
    _morph = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH_AVAILABLE = False
    _morph = None  # type: ignore[assignment]


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")

DEFAULT_STOP_VERBS = {
    "быть",
    "стать",
    "мочь",
    "иметь",
    "сказать",
    "говорить",
}


def split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)
    return [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]


@lru_cache(maxsize=50_000)
def _parse_token(token: str):
    if not _MORPH_AVAILABLE or _morph is None:
        return None
    return _morph.parse(token)[0]


def _lemmatize_verbs(tokens: Iterable[str], stop_verbs: set[str]) -> List[str]:
    if not _MORPH_AVAILABLE or _morph is None:
        return []

    out: List[str] = []
    for tok in tokens:
        p = _parse_token(tok)
        if p is None:
            continue
        pos = p.tag.POS
        if pos in {"VERB", "INFN"}:
            lemma = p.normal_form
            if lemma and lemma not in stop_verbs:
                out.append(lemma)
    return out


def _normalize(s: str) -> str:
    return (s or "").strip().lower().replace("ё", "е")


def extract_actions_for_persons(
    text: str,
    persons: List[str],
    stop_verbs: set[str] | None = None,
) -> Dict[str, List[str]]:
    """
    Возвращает dict: person -> список лемм-глаголов,
    извлечённых из предложений, где упоминается персона.

    persons — ожидаем уже канонизированные строки ('дональд трамп', 'зеленский', ...)
    """
    stop_verbs = stop_verbs or set(DEFAULT_STOP_VERBS)

    t = (text or "").strip()
    if not t or not persons:
        return {}

    sents = split_sentences(t)
    if not sents:
        return {}

    # варианты матчинга: полное имя и фамилия (если есть)
    person_variants: Dict[str, List[str]] = {}
    for p in persons:
        p0 = _normalize(p)
        if not p0:
            continue
        parts = p0.split()
        variants = [p0]
        if len(parts) >= 2:
            variants.append(parts[-1])
        # убираем дубли
        person_variants[p0] = list(dict.fromkeys(variants))

    # компилируем regex'ы по границам слова (без подстрочных совпадений)
    # (?<!\w) ... (?!\w) работает надёжнее, чем \b для кириллицы + дефисов
    person_patterns: Dict[str, List[re.Pattern[str]]] = {}
    for p, vars_ in person_variants.items():
        pats: List[re.Pattern[str]] = []
        for v in vars_:
            v = v.strip()
            if not v:
                continue
            pats.append(re.compile(rf"(?<!\w){re.escape(v)}(?!\w)"))
        person_patterns[p] = pats

    res: Dict[str, List[str]] = {p: [] for p in person_patterns.keys()}

    for sent in sents:
        s_low = _normalize(sent)

        matched: List[str] = []
        for p, pats in person_patterns.items():
            if any(pat.search(s_low) for pat in pats):
                matched.append(p)

        if not matched:
            continue

        tokens = [_normalize(m.group(0)) for m in _TOKEN_RE.finditer(sent)]
        verbs = _lemmatize_verbs(tokens, stop_verbs)
        if not verbs:
            continue

        for p in matched:
            res[p].extend(verbs)

    return {p: v for p, v in res.items() if v}