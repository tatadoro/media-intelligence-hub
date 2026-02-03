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


def _count_scripts(text: str) -> tuple[int, int]:
    cyr = 0
    lat = 0
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x052F:  # Cyrillic
            cyr += 1
        elif 0x0041 <= o <= 0x007A:  # Latin
            lat += 1
    return cyr, lat


def detect_lang(text: str) -> tuple[str, float]:
    t = (text or "").strip()
    if not t:
        return "unknown", float("-inf")

    # Heuristic for mixed scripts on longer texts
    cyr, lat = _count_scripts(t)
    total = cyr + lat
    if total >= 40:
        cyr_r = cyr / total if total else 0.0
        lat_r = lat / total if total else 0.0
        if 0.3 <= cyr_r <= 0.7 and 0.3 <= lat_r <= 0.7:
            return "mixed", 0.0

    # Short texts: avoid unstable detection
    if total < 15:
        return "unknown", float("-inf")

    if not _AVAILABLE or langid is None:
        # Fallback heuristic: pick dominant script
        if total > 0:
            if cyr >= lat * 2:
                return "ru", 0.0
            if lat >= cyr * 2:
                return "en", 0.0
        return "unknown", float("-inf")

    code, score = langid.classify(t[:5000])
    return (code or "unknown"), float(score)
