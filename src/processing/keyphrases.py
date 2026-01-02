"""
Keyphrases extraction (multilingual) using YAKE.

- dependency: yake (moderate, but usually easy)
- output: list of normalized keyphrases
"""

from __future__ import annotations

import re
from typing import List

try:
    import yake  # type: ignore
    _AVAILABLE = True
except Exception:
    yake = None  # type: ignore
    _AVAILABLE = False

_RE_SPACES = re.compile(r"\s+")


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

    try:
        kw_extractor = yake.KeywordExtractor(lan=lan, n=max_ngram, top=top_k)
        kws = kw_extractor.extract_keywords(t)
    except Exception:
        # If YAKE rejects the lang code, fall back to English
        kw_extractor = yake.KeywordExtractor(lan="en", n=max_ngram, top=top_k)
        kws = kw_extractor.extract_keywords(t)

    out: List[str] = []
    seen = set()

    for phrase, _score in kws:
        p = _RE_SPACES.sub(" ", (phrase or "").strip().lower())
        if len(p) < 3:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    return out


def keyphrases_to_str(keyphrases: List[str]) -> str:
    """Compatibility with your CH schema style: 'a;b;c'."""
    return ";".join([p for p in keyphrases if p and p.strip()])