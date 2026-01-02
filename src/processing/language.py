"""
Language detection for news texts.

- dependency: langid (lightweight)
- robust: safe defaults, short/empty texts -> "unknown"
"""

from __future__ import annotations

from typing import Tuple

try:
    import langid  # type: ignore
    _AVAILABLE = True
except Exception:
    langid = None  # type: ignore
    _AVAILABLE = False


def detect_lang(text: str) -> tuple[str, float]:
    t = (text or "").strip()
    if not t or not _AVAILABLE or langid is None:
        return "unknown", float("-inf")

    code, score = langid.classify(t[:5000])
    return (code or "unknown"), float(score)