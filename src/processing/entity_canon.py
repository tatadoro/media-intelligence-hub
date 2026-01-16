from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, DefaultDict

try:
    import pymorphy2
    _MORPH = pymorphy2.MorphAnalyzer()
    _MORPH_AVAILABLE = True
except Exception:
    _MORPH_AVAILABLE = False
    _MORPH = None

from typing import List


def parse_semicolon_list(s: str | None) -> List[str]:
    """
    Разбирает строку вида "a; b; c" в список ["a","b","c"].
    - Убирает пробелы по краям
    - Выкидывает пустые элементы
    - Не меняет регистр и не делает нормализацию (это важно)
    """
    if not s:
        return []
    parts = [p.strip() for p in str(s).split(";")]
    return [p for p in parts if p]


def join_semicolon_list(items: List[str] | None) -> str:
    """
    Собирает список в "a; b; c".
    - Убирает пробелы по краям каждого элемента
    - Выкидывает пустые элементы
    """
    if not items:
        return ""
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return "; ".join(cleaned)

_RE_SPACES = re.compile(r"\s+")
_RE_BRACKETS = re.compile(r"\s*\([^)]*\)")

def _clean(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = _RE_BRACKETS.sub("", s)
    s = _RE_SPACES.sub(" ", s)
    return s.strip()

def _normal_form_token(tok: str) -> str:
    tok = _clean(tok)
    if not tok:
        return ""
    if not _MORPH_AVAILABLE or _MORPH is None:
        return tok
    p = _MORPH.parse(tok)
    if not p:
        return tok
    return p[0].normal_form or tok

def _canon_person_phrase(phrase: str) -> str:
    phrase = _clean(phrase)
    if not phrase:
        return ""

    # токены только буквенно-дефисные
    tokens = [t for t in re.split(r"[^\w-]+", phrase) if t]
    if not tokens:
        return ""

    # нормализуем каждый токен
    norm = [_normal_form_token(t) for t in tokens]
    norm = [t for t in norm if t]

    if not norm:
        return ""

    # для однословного — это почти всегда фамилия/имя: тоже normal_form
    if len(norm) == 1:
        return norm[0]

    # ключевой момент: фамилия = последний токен -> normal_form уже применён
    # оставляем: имя + (отчество если есть) + фамилия
    return " ".join(norm)

def canon_list_persons(_: str) -> str:
    raise RuntimeError(
        "canon_list_persons deprecated: use entities_persons_canon produced in silver_to_gold_local.py"
    )