"""
DataFrame enrichment: lang + keyphrases + sentiment.

Adds columns:
- lang
- keyphrases (string "a;b;c")
- sentiment_label (pos|neu|neg)
- sentiment_score (float)
"""

from __future__ import annotations

import pandas as pd
from typing import Any

from src.processing.language import detect_lang
from src.processing.keyphrases import extract_keyphrases, keyphrases_to_str
from src.processing.sentiment import analyze_sentiment

def _safe_text(x: Any) -> str:
    # важно: NaN/None -> ""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    # str(nan) -> "nan" (плохой кейс) — считаем пустым
    if s.strip().lower() == "nan":
        return ""
    return s 

def add_lang_keyphrases_sentiment(
    df: pd.DataFrame,
    *,
    text_col: str = "nlp_text",
    summary_col: str = "summary",
) -> pd.DataFrame:
    out = df.copy()

    # Ensure columns exist even if df is empty
    if out.empty:
        if "lang" not in out.columns:
            out["lang"] = pd.Series(dtype="string")
        if "keyphrases" not in out.columns:
            out["keyphrases"] = pd.Series(dtype="string")
        if "sentiment_label" not in out.columns:
            out["sentiment_label"] = pd.Series(dtype="string")
        if "sentiment_score" not in out.columns:
            out["sentiment_score"] = pd.Series(dtype="float")
        return out

    # Safe text inputs
    text = out.get(text_col, pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    summary = out.get(summary_col, pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)

    # 1) lang (use title+clean_text already assembled as nlp_text by summarization.py)
    langs = []
    for t in text.tolist():
        lg, _score = detect_lang(t)
        langs.append(lg)
    out["lang"] = langs

    # 2) keyphrases (multilingual, by YAKE)
    kp_str = []
    for t, lg in zip(text.tolist(), langs):
        kp = extract_keyphrases(t, lang=lg, top_k=15, max_ngram=3)
        kp_str.append(keyphrases_to_str(kp))
    out["keyphrases"] = kp_str

    # 3) sentiment (faster/cleaner by summary; fallback to text if summary empty)
    s_labels = []
    s_scores = []

    for _, row in out.iterrows():
        s = _safe_text(row.get("summary"))
        t = _safe_text(row.get("nlp_text")) or _safe_text(row.get("clean_text"))

        sent = analyze_sentiment(s if s.strip() else t)
        s_labels.append(sent.label)
        s_scores.append(float(sent.score))

    out["sentiment_label"] = s_labels
    out["sentiment_score"] = s_scores

    return out