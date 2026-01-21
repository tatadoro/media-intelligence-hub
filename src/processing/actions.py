# src/processing/actions.py
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple
from collections import Counter
try:
    import pymorphy3
    _MORPH_AVAILABLE = True
    _morph = pymorphy3.MorphAnalyzer(lang="ru")
except Exception:
    _MORPH_AVAILABLE = False
    _morph = None  # type: ignore[assignment]

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")
_SPACE_RE = re.compile(r"\s+")

# расширенный stoplist (минимум полезного шума)
DEFAULT_STOP_VERBS = {
    "быть","стать","мочь","иметь","являться","находиться","считаться",
    "сказать","говорить","сообщить","заявить","отметить","подчеркнуть","написать",
    "рассказать","пояснить","уточнить","добавить","признать","объявить",
}

# нормализация “речевых” в один тип
VERB_CANON = {
    "сказать": "заявить",
    "говорить": "заявить",
    "сообщить": "заявить",
    "отметить": "заявить",
    "подчеркнуть": "заявить",
    "заявить": "заявить",
}

def _normalize(s: str) -> str:
    return (s or "").strip().lower().replace("ё", "е")

def split_sentences(text: str) -> List[str]:
    t = _normalize(text)
    if not t:
        return []
    t = _SPACE_RE.sub(" ", t)
    return [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]

@lru_cache(maxsize=50_000)
def _parse_token(token: str):
    if not _MORPH_AVAILABLE or _morph is None:
        return None
    return _morph.parse(token)[0]

def _lemmatize_verbs(tokens: Iterable[str], stop_verbs: set[str], *, include_infinitives: bool = False) -> List[str]:
    if not _MORPH_AVAILABLE or _morph is None:
        return []

    out: List[str] = []
    for tok in tokens:
        p = _parse_token(tok)
        if p is None:
            continue
        pos = p.tag.POS
        if pos == "VERB" or (include_infinitives and pos == "INFN"):
            lemma = (p.normal_form or "").strip()
            if not lemma:
                continue
            lemma = VERB_CANON.get(lemma, lemma)
            if lemma in stop_verbs:
                continue
            out.append(lemma)
    return out

def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """[(token, start, end)] по исходной строке."""
    out: List[Tuple[str,int,int]] = []
    for m in _TOKEN_RE.finditer(text):
        tok = _normalize(m.group(0))
        if tok:
            out.append((tok, m.start(), m.end()))
    return out

def _merge_spans(spans: List[Tuple[int, int]], gap: int = 2) -> List[Tuple[int, int]]:
    """
    Схлопывает пересекающиеся и "почти соседние" span-ы.
    gap — допустимый разрыв между span-ами, чтобы считать их одним упоминанием.
    """
    if not spans:
        return []

    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = spans_sorted[0]

    for s, e in spans_sorted[1:]:
        # если пересекаются или почти соприкасаются (разрыв <= gap) — объединяем
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return merged

def _compile_person_patterns(persons: List[str]) -> Dict[str, List[re.Pattern[str]]]:
    """
    person -> patterns для матчинга:
    - полная строка
    - фамилия (если есть >=2 токена)
    - два последних токена (для составных фамилий/двусловных вариантов)
    - простые падежные варианты фамилии
    """
    out: Dict[str, List[re.Pattern[str]]] = {}

    for p in persons:
        p0 = _normalize(p)
        if not p0:
            continue

        parts = p0.split()
        variants: List[str] = [p0]

        if len(parts) >= 2:
            last = parts[-1]
            prev_last = parts[-2]

            # фамилия
            variants.append(last)

            # два последних токена (например: "ван дер", "иванов петров" и т.п.)
            variants.append(f"{prev_last} {last}")

            # простые падежные варианты фамилии для матчинга
            if not last.endswith(("а", "я")):
                variants.append(last + "а")
                variants.append(last + "я")

        pats: List[re.Pattern[str]] = []
        for v in dict.fromkeys(variants):
            v = v.strip()
            if not v:
                continue

            # делаем дефис "гибким": разрешаем пробелы вокруг обычного дефиса
            v_esc = re.escape(v).replace(r"\-", r"\s*-\s*")

            pats.append(re.compile(rf"(?<!\w){v_esc}(?!\w)"))

        if pats:
            out[p0] = pats

    return out
def extract_actions_for_persons(
    text: str,
    persons: List[str],
    stop_verbs: set[str] | None = None,
    *,
    window_tokens: int = 8,
    top_verbs_per_person: int = 10,
) -> Dict[str, List[str]]:
    """
    person -> список лемм-глаголов вокруг упоминания (окно ±window_tokens).
    """
    if stop_verbs is None:
        stop_verbs = set(DEFAULT_STOP_VERBS)

    t = (text or "").strip()
    if not t or not persons:
        return {}

    persons_norm = []
    for p in persons:
        p0 = _normalize(p)
        if p0 and len(p0) >= 3:
            persons_norm.append(p0)
    persons_norm = list(dict.fromkeys(persons_norm))
    if not persons_norm:
        return {}

    patterns = _compile_person_patterns(persons_norm)
    if not patterns:
        return {}

    res: Dict[str, List[str]] = {p: [] for p in patterns.keys()}

    for sent_norm in split_sentences(t):

        matched: List[Tuple[str, List[Tuple[int,int]]]] = []
        for p, pats in patterns.items():
            spans: List[Tuple[int,int]] = []
            for pat in pats:
                for m in pat.finditer(sent_norm):
                    spans.append((m.start(), m.end()))
            if spans:
                matched.append((p, spans))

        if not matched:
            continue

        toks = _tokenize_with_spans(sent_norm)
        if not toks:
            continue

        for person, spans in matched:
            # найдём токен-индексы, пересекающиеся со span, и возьмём окно вокруг
            person_verbs: List[str] = []
            spans = _merge_spans(spans, gap=2)
            for sa, sb in spans:
                idxs: List[int] = []
                for i, (_tok, a, b) in enumerate(toks):
                    if not (b <= sa or a >= sb):  # пересечение [a,b) и [sa,sb)
                        idxs.append(i)

                if not idxs:
                    continue

                left = max(0, min(idxs) - window_tokens)
                right = min(len(toks), max(idxs) + window_tokens + 1)
                window = [tok for tok, _a, _b in toks[left:right]]

                verbs = _lemmatize_verbs(window, stop_verbs, include_infinitives=False)
                if verbs:
                    person_verbs.extend(verbs)
            if person_verbs:
                # убираем дубли в рамках одного предложения, порядок сохраняем
                res[person].extend(list(dict.fromkeys(person_verbs)))

    # ограничим top-N уникальных на персону (по частоте в тексте)
    out: Dict[str, List[str]] = {}
    for p, verbs in res.items():
        if not verbs:
            continue
        cnt = Counter(verbs)
        top = [v for v, _ in cnt.most_common(top_verbs_per_person)]
        if top:
            out[p] = top
    return out