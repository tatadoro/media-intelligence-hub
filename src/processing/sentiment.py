from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Optional
import re

# --- optional morphology (better for Russian lexicon fallback) ---
try:
    import pymorphy3  # type: ignore

    _MORPH = pymorphy3.MorphAnalyzer(lang="ru")
    _MORPH_OK = True
except Exception:
    _MORPH = None
    _MORPH_OK = False

_TOKEN_RE = re.compile(r"[а-яa-zё]+", re.IGNORECASE)

_NEGATORS: Set[str] = {"не", "ни", "нет", "без"}

# Базовый MVP-словарик (можешь расширять)
_POS: Set[str] = {
    "отлично",
    "прекрасно",
    "классно",
    "супер",
    "здорово",
    "хорошо",
    "понравиться",
    "рад",
    "радовать",
    "успех",
    "удаться",
    "лучший",
    "счастливый",
    "красивый",
    "замечательно",
    "улучшение",
    "рост",
    "победа",
}

_NEG: Set[str] = {
    "ужасно",
    "плохо",
    "отвратительно",
    "кошмар",
    "провал",
    "скандал",
    "ненавидеть",
    "разочарование",
    "разочаровать",
    "трагедия",
    "кризис",
    "провальный",
    "ужасный",
    "неудача",
    "арест",
    "обвинение",
    "обвинить",
    "убийство",
    "смерть",
    "ранение",
    "катастрофа",
    "удар",
    "взрыв",
}


@dataclass(frozen=True)
class SentimentResult:
    label: str              # 'pos'|'neu'|'neg' или '' если не считали
    score: Optional[float]  # None, если не посчиталось; иначе [-1..1], для neu обычно 0.0
    ok: bool                # True, если реально считали (текст достаточный + модель/лексикон отработали)
    confidence: Optional[float]  # 0..1, None если не посчитали
    source: str             # transformers|lexicon|empty


def _norm(text: str) -> str:
    return (text or "").strip().lower().replace("ё", "е")


def _simple_lemma(tok: str) -> str:
    tok = tok.strip().lower().replace("ё", "е")
    if not tok:
        return ""
    if _MORPH_OK and _MORPH is not None:
        p = _MORPH.parse(tok)
        if p:
            return p[0].normal_form
    return tok


# --- transformers sentiment (optional) ---
_TF_MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"
_TF_PIPE = None
_TF_AVAILABLE: Optional[bool] = None


def _get_tf_pipe():
    """
    Ленивая инициализация transformers pipeline.
    Возвращает None, если transformers/модель недоступны.
    """
    global _TF_PIPE, _TF_AVAILABLE
    if _TF_AVAILABLE is False:
        return None
    if _TF_PIPE is not None:
        return _TF_PIPE
    try:
        from transformers import pipeline  # type: ignore

        _TF_PIPE = pipeline(
            "text-classification",
            model=_TF_MODEL_NAME,
            tokenizer=_TF_MODEL_NAME,
            truncation=True,
        )
        _TF_AVAILABLE = True
        return _TF_PIPE
    except Exception:
        _TF_AVAILABLE = False
        return None


def _lexicon_sentiment(text: str) -> SentimentResult:
    """
    Лексиконный фолбек.
    ok=True только если удалось извлечь токены и посчитать score.
    """
    t = _norm(text)
    if not t:
        return SentimentResult("", None, False, None, "empty")

    toks = [m.group(0) for m in _TOKEN_RE.finditer(t)]
    if not toks:
        return SentimentResult("", None, False, None, "empty")

    score = 0.0
    window_neg = 0

    for raw in toks:
        tok = _simple_lemma(raw)
        if not tok:
            continue

        if tok in _NEGATORS:
            window_neg = 2
            continue

        delta = 0.0
        if tok in _POS:
            delta = 1.0
        elif tok in _NEG:
            delta = -1.0

        if delta != 0.0:
            if window_neg > 0:
                delta = -delta
            score += delta

        if window_neg > 0:
            window_neg -= 1

    # нормализация в [-1..1]
    score = max(-1.0, min(1.0, score / 3.0))

    if score > 0.15:
        return SentimentResult("pos", float(score), True, min(1.0, abs(score)), "lexicon")
    if score < -0.15:
        return SentimentResult("neg", float(score), True, min(1.0, abs(score)), "lexicon")
    return SentimentResult("neu", 0.0, True, 0.1, "lexicon")


def analyze_sentiment(
    text: str,
    *,
    min_chars: int = 120,
    min_tokens: int = 8,
    force_transformers: bool = False,
) -> SentimentResult:
    """
    Главная функция sentiment.
    - Если текста мало: ok=False, score=None, label=""
    - Иначе пытаемся transformers, при проблемах — лексикон.
    """
    t = _norm(text)
    if not t:
        return SentimentResult("", None, False, None, "empty")

    # quality gate: короткие тексты не считаем, чтобы не плодить "нейтрал 0.0" по мусору
    toks = [m.group(0) for m in _TOKEN_RE.finditer(t)]
    if len(t) < min_chars or len(toks) < min_tokens:
        return SentimentResult("", None, False, None, "empty")

    pipe = _get_tf_pipe()
    if pipe is None and force_transformers:
        return SentimentResult("", None, False, None, "transformers_missing")

    if pipe is not None:
        try:
            r = pipe(t[:2000])[0]
            label_raw = str(r.get("label", "")).lower()
            prob = float(r.get("score", 0.0))

            if "pos" in label_raw or "positive" in label_raw:
                return SentimentResult("pos", prob, True, prob, "transformers")
            if "neg" in label_raw or "negative" in label_raw:
                return SentimentResult("neg", -prob, True, prob, "transformers")
            return SentimentResult("neu", 0.0, True, prob, "transformers")
        except Exception:
            if force_transformers:
                return SentimentResult("", None, False, None, "transformers_error")
            return _lexicon_sentiment(t)

    return _lexicon_sentiment(t)
