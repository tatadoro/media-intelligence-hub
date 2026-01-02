"""
Sentiment (pos/neu/neg) with a multilingual transformer model.

- dependencies: transformers + torch (heavy)
- IMPORTANT: module is designed to be OPTIONAL.
  If transformers/torch/model not available -> returns ("neu", 0.0) and never breaks pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

try:
    from transformers import pipeline  # type: ignore
    _AVAILABLE = True
except Exception:
    pipeline = None  # type: ignore
    _AVAILABLE = False


_SENT_PIPE = None  # lazy init


@dataclass(frozen=True)
class Sentiment:
    label: str  # "pos" | "neu" | "neg"
    score: float


def _get_pipeline():
    global _SENT_PIPE
    if _SENT_PIPE is not None:
        return _SENT_PIPE
    if not _AVAILABLE or pipeline is None:
        return None

    try:
        _SENT_PIPE = pipeline(
            "sentiment-analysis",
            model=_MODEL,
            tokenizer=_MODEL,
        )
        return _SENT_PIPE
    except Exception:
        return None


def analyze_sentiment(text: str) -> Sentiment:
    t = (text or "").strip()
    if not t:
        return Sentiment("neu", 0.0)

    pl = _get_pipeline()
    if pl is None:
        return Sentiment("neu", 0.0)

    try:
        res = pl(t[:2000], truncation=True)[0]  # dict(label=..., score=...)
        raw_label = (res.get("label") or "").lower()
        score = float(res.get("score") or 0.0)

        if "pos" in raw_label:
            return Sentiment("pos", score)
        if "neg" in raw_label:
            return Sentiment("neg", score)
        return Sentiment("neu", score)
    except Exception:
        return Sentiment("neu", 0.0)