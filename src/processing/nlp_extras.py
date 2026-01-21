"""
DataFrame enrichment: lang + keyphrases + sentiment.

Adds columns:
- lang
- keyphrases (string "a;b;c")
- keyphrases_n (UInt16)      # сколько фраз в строке
- keyphrases_ok (UInt8)      # 1 если извлечение прошло успешно
- sentiment_label (pos|neu|neg)
- sentiment_score (nullable float)
- sentiment_ok (UInt8)
- sentiment_source (string)  # summary|text|empty
- sentiment_input_len (UInt16)
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from src.processing.language import detect_lang
from src.processing.keyphrases import extract_keyphrases, postprocess_keyphrases
from src.processing.sentiment import analyze_sentiment


def _safe_text(x: Any) -> str:
    """NaN/None -> '' и защищаемся от str(nan) == 'nan'."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    if s.strip().lower() == "nan":
        return ""
    return s


def add_lang_keyphrases_sentiment(
    df: pd.DataFrame,
    *,
    text_col: str = "nlp_text",
    summary_col: str = "summary",
    keyphrases_top_k: int = 15,
    keyphrases_candidate_k: int = 30,
    keyphrases_max_ngram: int = 3,
) -> pd.DataFrame:
    """
    Добавляет в датафрейм:
      - lang: язык текста
      - keyphrases: ключевые фразы (YAKE), строка "a;b;c"
      - keyphrases_n: число фраз
      - keyphrases_ok: флаг успешного извлечения (1/0)
      - sentiment_label: метка тональности (pos|neu|neg)
      - sentiment_score: score (nullable float)
      - sentiment_ok: флаг успешного расчёта (UInt8)
      - sentiment_source: откуда брали текст для sentiment (summary|text|empty)
      - sentiment_input_len: длина текста, по которому считали sentiment (UInt16)

    Важно: функция НИКОГДА не должна возвращать None.
    При любой ошибке возвращает df с гарантированными колонками (возможно пустыми).
    """
    out = df.copy()

    # --- гарантируем контракт колонок (даже если дальше что-то упадёт) ---
    n = len(out)

    def _ensure(col: str, series: pd.Series) -> None:
        if col not in out.columns:
            out[col] = series

    _ensure("lang", pd.Series([""] * n, index=out.index, dtype="string"))
    _ensure("keyphrases", pd.Series([""] * n, index=out.index, dtype="string"))
    _ensure("keyphrases_n", pd.Series([0] * n, index=out.index, dtype="UInt16"))
    _ensure("keyphrases_ok", pd.Series([0] * n, index=out.index, dtype="UInt8"))

    _ensure("sentiment_label", pd.Series([""] * n, index=out.index, dtype="string"))
    _ensure("sentiment_score", pd.Series([None] * n, index=out.index, dtype="Float64"))
    _ensure("sentiment_ok", pd.Series([0] * n, index=out.index, dtype="UInt8"))
    _ensure("sentiment_source", pd.Series([""] * n, index=out.index, dtype="string"))
    _ensure("sentiment_input_len", pd.Series([0] * n, index=out.index, dtype="UInt16"))

    if out.empty:
        return out

    # --- безопасные входы текста ---
    text_raw = out.get(text_col, pd.Series([""] * n, index=out.index))
    summary_raw = out.get(summary_col, pd.Series([""] * n, index=out.index))

    text_list = [_safe_text(x) for x in text_raw.tolist()]
    summary_list = [_safe_text(x) for x in summary_raw.tolist()]

    # 1) lang
    langs: List[str] = []
    for t in text_list:
        try:
            lg, _score = detect_lang(t)
        except Exception:
            lg = "unknown"
        langs.append(lg)
    out["lang"] = pd.Series(langs, index=out.index, dtype="string")

    # 2) keyphrases (YAKE) -> postprocess -> "a;b;c"
    kp_str: List[str] = []
    kp_n: List[int] = []
    kp_ok: List[int] = []

    for t, lg in zip(text_list, langs):
        if not (t or "").strip():
            kp_str.append("")
            kp_n.append(0)
            kp_ok.append(0)
            continue
        try:
            # Берём больше кандидатов, затем чистим/дедупим и режем до top_k
            kp_raw = extract_keyphrases(
                t,
                lang=lg,
                top_k=keyphrases_candidate_k,
                max_ngram=keyphrases_max_ngram,
            )
            kp_clean = postprocess_keyphrases(kp_raw, top_k=keyphrases_top_k)

            s = ";".join(kp_clean)
            kp_str.append(s)
            kp_n.append(min(len(kp_clean), 65535))
            kp_ok.append(1)
        except Exception:
            kp_str.append("")
            kp_n.append(0)
            kp_ok.append(0)

    out["keyphrases"] = pd.Series(kp_str, index=out.index, dtype="string")
    out["keyphrases_n"] = pd.Series(kp_n, index=out.index, dtype="UInt16")
    out["keyphrases_ok"] = pd.Series(kp_ok, index=out.index, dtype="UInt8")

    # 3) sentiment (nullable score + флаг ok)
    s_labels: List[str] = []
    s_scores: List[Optional[float]] = []
    s_ok: List[int] = []
    s_src: List[str] = []
    s_len: List[int] = []

    for s, t in zip(summary_list, text_list):
        if (s or "").strip():
            base = s
            src = "summary"
        elif (t or "").strip():
            base = t
            src = "text"
        else:
            base = ""
            src = "empty"

        s_src.append(src)
        s_len.append(min(len(base), 65535))

        if not base.strip():
            s_labels.append("")
            s_scores.append(None)
            s_ok.append(0)
            continue

        try:
            sent = analyze_sentiment(base)
            if getattr(sent, "ok", False):
                s_labels.append(getattr(sent, "label", "") or "")
                s_scores.append(getattr(sent, "score", None))
                s_ok.append(1)
            else:
                s_labels.append("")
                s_scores.append(None)
                s_ok.append(0)
        except Exception:
            s_labels.append("")
            s_scores.append(None)
            s_ok.append(0)

    out["sentiment_label"] = pd.Series(s_labels, index=out.index, dtype="string")
    out["sentiment_score"] = pd.Series(s_scores, index=out.index, dtype="Float64")
    out["sentiment_ok"] = pd.Series(s_ok, index=out.index, dtype="UInt8")
    out["sentiment_source"] = pd.Series(s_src, index=out.index, dtype="string")
    out["sentiment_input_len"] = pd.Series(s_len, index=out.index, dtype="UInt16")

    return out