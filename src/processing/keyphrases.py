"""
Keyphrases extraction (multilingual) using YAKE.

- dependency: yake (moderate, but usually easy)
- output: list of normalized keyphrases (deduped for BI)
"""

from __future__ import annotations

import re
from typing import List, Tuple

try:
    import yake  # type: ignore

    _AVAILABLE = True
except Exception:
    yake = None  # type: ignore
    _AVAILABLE = False

# Optional: better dedup for Russian keyphrases via lemmatization
try:
    import pymorphy3  # type: ignore

    _MORPH = pymorphy3.MorphAnalyzer(lang="ru")
    _MORPH_OK = True
except Exception:
    _MORPH = None
    _MORPH_OK = False

_RE_SPACES = re.compile(r"\s+")
_RE_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+", re.UNICODE)

# Слишком общие «фразы», которые YAKE иногда возвращает, но для BI это шум.
_GENERIC_SINGLE = {
    # ru
    "суд",
    "банк",
    "университет",
    "институт",
    "компания",
    "корпорация",
    "группа",
    "холдинг",
    "министерство",
    "ведомство",
    "правительство",
    "администрация",
    "прокуратура",
    "полиция",
    "комитет",
    "департамент",
    "служба",
    "комиссия",
    "флот",
    # en
    "company",
    "bank",
    "ministry",
    "government",
}


def _norm_phrase(p: str) -> str:
    """Нормализация для сравнения/дедупа."""
    p = _RE_SPACES.sub(" ", (p or "").strip().lower())
    return p


def _tokenize(p: str) -> List[str]:
    return [m.group(0).lower() for m in _RE_TOKEN.finditer(p or "") if m.group(0)]


def _lemma_tokens_ru(tokens: List[str]) -> List[str]:
    if not _MORPH_OK or _MORPH is None:
        return tokens
    out: List[str] = []
    for t in tokens:
        try:
            parses = _MORPH.parse(t)
            if parses:
                out.append((parses[0].normal_form or t).lower())
            else:
                out.append(t)
        except Exception:
            out.append(t)
    return out


def _is_trivial(tokens: List[str]) -> bool:
    """Отсечь очевидный шум (одиночные общие слова)."""
    if not tokens:
        return True
    if len(tokens) == 1:
        t = tokens[0]
        if len(t) < 4:
            return True
        if t in _GENERIC_SINGLE:
            return True
    return False


def _anchor_key(tokens: List[str]) -> str:
    """Якорь для дедупа: первые 2 токена (или 1, если меньше)."""
    if not tokens:
        return ""
    return " ".join(tokens[:2])


def _dedupe_by_anchor(items: List[Tuple[str, float]], *, lang: str) -> List[Tuple[str, float]]:
    """Дедуп по якорю: оставляем лучший (у YAKE score меньше = лучше)."""
    best: dict[str, Tuple[str, float]] = {}
    for phrase, score in items:
        toks = _tokenize(phrase)
        toks_key = _lemma_tokens_ru(toks) if lang == "ru" else toks
        if _is_trivial(toks_key):
            continue
        a = _anchor_key(toks_key)
        if not a:
            continue
        cur = best.get(a)
        if cur is None:
            best[a] = (phrase, score)
        else:
            p0, s0 = cur
            # YAKE: меньше score — лучше; при равенстве берём более длинную фразу
            if score < s0 or (score == s0 and len(phrase) > len(p0)):
                best[a] = (phrase, score)
    return list(best.values())


def _dedupe_by_substring(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Убираем короткие фразы, которые являются подстрокой уже выбранных длинных."""
    kept: List[Tuple[str, float]] = []
    # сначала длинные (и лучшие по score), чтобы они «поглотили» короткие варианты
    items_sorted = sorted(items, key=lambda x: (-len(_norm_phrase(x[0])), x[1]))
    kept_norm: List[str] = []
    for phrase, score in items_sorted:
        p = _norm_phrase(phrase)
        if not p:
            continue
        if any((p != q) and (p in q) for q in kept_norm):
            continue
        kept.append((phrase, score))
        kept_norm.append(p)
    # вернём в порядке «качества» (YAKE score ASC)
    return sorted(kept, key=lambda x: x[1])


def extract_keyphrases(
    text: str,
    *,
    lang: str,
    top_k: int = 15,
    max_ngram: int = 3,
) -> List[str]:
    t = (text or "").strip()
    if not t or not _AVAILABLE or yake is None:
        return []

    # YAKE expects language codes like 'en', 'ru'. For unknown -> 'en'.
    lan = lang if lang and lang != "unknown" else "en"

    # Берём кандидатов с запасом, потому что мы потом сильно фильтруем.
    try:
        kw_extractor = yake.KeywordExtractor(lan=lan, n=max_ngram, top=max(top_k * 3, 30))
        kws = kw_extractor.extract_keywords(t)
    except Exception:
        kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram, top=max(top_k * 3, 30))
        kws = kw_extractor.extract_keywords(t)

    # 1) первичная нормализация + exact dedup
    raw: List[Tuple[str, float]] = []
    seen = set()
    for phrase, score in kws:
        p = _norm_phrase(phrase)
        if len(p) < 3:
            continue
        if p in seen:
            continue
        seen.add(p)
        raw.append((p, float(score) if score is not None else 1.0))

    if not raw:
        return []

    # 2) дедуп по якорю (первые 2 токена, для ru — по леммам)
    filtered = _dedupe_by_anchor(raw, lang=lan)
    if not filtered:
        return []

    # 3) убрать подстроки/короткие варианты
    filtered = _dedupe_by_substring(filtered)

    # 4) финальный top_k (по score)
    return [p for p, _s in sorted(filtered, key=lambda x: x[1])[:top_k]]


def keyphrases_to_str(keyphrases: List[str]) -> str:
    """Compatibility with your CH schema style: 'a;b;c'."""
    return ";".join([p for p in keyphrases if p and p.strip()])


import re
from typing import Iterable, List, Tuple

_RE_SPACES = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w\s\-\.]", re.UNICODE)

_RU_STOP = {
    "и","в","во","на","к","ко","о","об","обо","от","до","из","за","для","по","при","про","у","без",
    "что","это","как","так","то","же","ли","бы","не","ни",
    "его","ее","их","ему","ей","им","ними","она","они","он",
    "год","годы","лет","сегодня","вчера","завтра",
}

def _norm_phrase(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_SPACES.sub(" ", s).strip()
    return s

def _is_noise_phrase(p: str) -> bool:
    if not p:
        return True
    if len(p) < 4:
        return True
    toks = p.split()
    if len(toks) == 1 and toks[0] in _RU_STOP:
        return True
    # фразы из одних стоп-слов
    if toks and all(t in _RU_STOP for t in toks):
        return True
    # одиночные прилагательные/общие слова часто шумят
    if len(toks) == 1 and toks[0].endswith(("ский","ская","ское","ские","ной","ная","ное","ные")):
        return True
    return False

def _drop_substrings(phrases: List[str]) -> List[str]:
    # удаляем фразы, которые являются подстрокой другой фразы
    out: List[str] = []
    for p in sorted(phrases, key=len, reverse=True):
        if any(p != q and p in q for q in out):
            continue
        out.append(p)
    return list(reversed(out))

def postprocess_keyphrases(
    keyphrases: Iterable[Tuple[str, float]] | Iterable[str],
    *,
    top_k: int = 15,
) -> List[str]:
    """
    Принимает либо список (phrase, score), либо список phrase.
    Возвращает очищенный список phrase (top_k).
    """
    items: List[str] = []
    for x in keyphrases:
        if isinstance(x, tuple) and len(x) >= 1:
            items.append(str(x[0]))
        else:
            items.append(str(x))

    normed = []
    seen = set()
    for s in items:
        p = _norm_phrase(s)
        if _is_noise_phrase(p):
            continue
        if p in seen:
            continue
        seen.add(p)
        normed.append(p)

    normed = _drop_substrings(normed)
    return normed[:top_k]