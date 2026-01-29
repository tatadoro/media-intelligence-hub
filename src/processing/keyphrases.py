"""
Keyphrase extraction utilities.

Goal: be robust on RU/EN short news texts and Telegram posts.
We keep the implementation simple and deterministic:
- tokenize -> generate ngrams -> score by tf-idf-like heuristic + penalties
- drop noisy tokens and generic singletons
- dedupe by substring
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Set, Optional


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+", re.UNICODE)

# Часто встречающиеся одиночные слова, которые почти всегда мусор как "ключевая фраза"
_GENERIC_SINGLE: Set[str] = {
    # ru
    "это", "эта", "этот", "эти",
    "так", "там", "тут", "здесь",
    "уже", "еще", "ещё",
    "будет", "быть", "есть",
    "очень", "просто", "сейчас",
    "сегодня", "вчера", "завтра",
    "тогда", "потом", "сразу",
    "снова", "опять",
    "время", "год", "года", "лет",
    "человек", "люди",
    "страна", "страны",
    # en
    "this", "that", "these", "those",
    "here", "there",
    "already", "still", "just",
    "today", "yesterday", "tomorrow",
    "time", "year", "years", "people",
}

# Базовые стоп-слова (минимальный набор, без внешних зависимостей).
_RU_STOP: Set[str] = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты",
    "к","у","же","вы","за","бы","по","только","ее","мне","было","вот","от","меня","еще","нет","о","из",
    "ему","теперь","когда","даже","ну","вдруг","ли","если","уже","или","ни","быть","был","него","до",
    "вас","нибудь","опять","уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где",
    "есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто","чего","раз",
    "тоже","себе","под","будет","ж","тогда","кто","этот","того","потому","этого","какой","совсем","ним",
    "здесь","этом","один","почти","мой","тем","чтобы","нее","сейчас","были","куда","зачем","всех","никогда",
    "можно","при","наконец","два","об","другой","хоть","после","над","больше","тот","через","эти","нас",
    "про","всего","них","какая","много","разве","три","эту","моя","впрочем","хорошо","свою","этой","перед",
    "иногда","лучше","чуть","том","нельзя","такой","им","более","всегда","конечно","всю","между",
}

_EN_STOP: Set[str] = {
    "the","a","an","and","or","but","if","then","else","so","to","of","in","on","at","for","from","by",
    "with","about","as","is","are","was","were","be","been","being","it","its","this","that","these","those",
    "i","you","he","she","we","they","them","his","her","our","their","my","your","me","him","us",
    "not","no","yes","do","does","did","done","have","has","had","will","would","can","could","may","might",
    "there","here","when","where","why","how","what","which","who","whom",
}

_STOP_BY_LANG: Dict[str, Set[str]] = {
    "ru": _RU_STOP,
    "en": _EN_STOP,
}


def _norm_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _is_bad_keyword(phrase: str, lang: str) -> bool:
    p = _norm_phrase(phrase)
    if not p:
        return True

    toks = p.split()
    if len(toks) == 1:
        t = toks[0]
        # слишком короткие
        if len(t) < 2:
            return True
        # стоп-слова и "генерик"
        if t in _STOP_BY_LANG.get(lang, set()):
            return True
        if t in _GENERIC_SINGLE:
            return True

    # если все токены — стоп-слова
    stop = _STOP_BY_LANG.get(lang, set())
    if stop and all(t in stop for t in toks):
        return True

    return False


def _score_ngram(tokens: List[str], n: int, stop: Set[str]) -> Counter:
    """
    Простой скоринг: считаем частоты n-грамм, но penalize если внутри есть стоп-слова.
    """
    c = Counter()
    if n <= 0 or len(tokens) < n:
        return c

    for i in range(len(tokens) - n + 1):
        ng = tokens[i:i+n]
        if any(t in stop for t in ng):
            continue
        # не берём ngram, где все токены слишком короткие
        if all(len(t) <= 2 for t in ng):
            continue
        c[" ".join(ng)] += 1
    return c


def _dedupe_by_substring(phrases: List[str]) -> List[str]:
    """
    Удаляем фразы, которые являются подстрокой другой, более длинной фразы.
    """
    out: List[str] = []
    sorted_ph = sorted({_norm_phrase(p) for p in phrases if p}, key=lambda x: (-len(x), x))
    for p in sorted_ph:
        if any(p != q and p in q for q in out):
            continue
        out.append(p)
    # возвращаем в “естественном” порядке: короткие выше после дедупа не нужно,
    # поэтому просто отсортируем по длине и алфавиту для стабильности
    return sorted(out, key=lambda x: (-len(x), x))


def extract_keyphrases(
    text: str,
    lang: str = "ru",
    top_k: int = 15,
    ngrams: Tuple[int, ...] = (1, 2, 3),
) -> List[str]:
    """
    Извлекает ключевые фразы из текста.
    Возвращает список строк (фраз).
    """
    lang = (lang or "ru").lower()
    stop = _STOP_BY_LANG.get(lang, set())

    tokens = _tokenize(text)
    if not tokens:
        return []

    scored = Counter()
    for n in ngrams:
        scored.update(_score_ngram(tokens, n, stop))

    # базовая сортировка: частота, затем длина (длиннее чуть выше), затем лексикографически
    cand = [p for p, _ in scored.most_common()]
    # фильтры качества
    cand = [p for p in cand if not _is_bad_keyword(p, lang)]
    # дедуп по подстроке
    cand = _dedupe_by_substring(cand)

    return cand[:top_k]


def postprocess_keyphrases(keyphrases: Iterable[str]) -> str:
    """
    Нормализует и склеивает ключевые фразы в строку (как у тебя в пайплайне).
    """
    uniq = []
    seen = set()
    for k in keyphrases or []:
        k2 = _norm_phrase(k)
        if not k2 or k2 in seen:
            continue
        seen.add(k2)
        uniq.append(k2)
    return ";".join(uniq)