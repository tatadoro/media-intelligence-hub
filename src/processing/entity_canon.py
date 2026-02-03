# src/processing/entity_canon.py
from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Set

# --- Morphology (pymorphy3) ---
_MORPH_ERROR: Optional[str] = None
try:
    import pymorphy3  # type: ignore

    _MORPH_AVAILABLE = True
    _MORPH = pymorphy3.MorphAnalyzer(lang="ru")
except Exception as e:
    _MORPH_AVAILABLE = False
    _MORPH = None
    _MORPH_ERROR = repr(e)

_RE_SPACES = re.compile(r"\s+")
_RE_BRACKETS = re.compile(r"\s*\([^)]*\)")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")

# --- GEO synonyms (минимальный каркас; расширяем постепенно) ---
GEO_SYNONYMS: Dict[str, str] = {
    # Россия / РФ
    "рф": "россия",
    "российская федерация": "россия",
    "российский федерация": "россия",
    "россии": "россия",  # на случай, если морфология недоступна
    "россией": "россия",
    # США
    "соединенные штаты": "сша",
    "соединенный штат": "сша",
    "соединенных штатов": "сша",
    "соединенные штаты америки": "сша",
    # ЕС / Евросоюз
    "евросоюз": "ес",
    "европейский союз": "ес",
    "ес": "ес",
    # Великобритания
    "великобритания": "великобритания",
    "соединенное королевство": "великобритания",
}
# --- ORG synonyms (минимальный каркас; расширяем постепенно) ---
ORG_SYNONYMS: Dict[str, str] = {
    "международный уголовный суд": "мус",
    "международный уголовный суд оон": "мус",
    "мус": "мус",
    "газеты.ru": "газета.ru",
    "минобороны": "минобороны",
    "оон": "оон",
    "объединенные нации": "оон",
    "нато": "нато",
}
# --- ORG cleanup patterns ---
_RE_QUOTES = re.compile(r"[\"“”„«»]")
_RE_ORG_LEGAL = re.compile(
    r"^(ооо|оао|зао|пао|pao|ао|ип|нко)\s+|"
    r"\s+(ооо|оао|зао|пао|pao|ао|ип|нко)$",
    re.IGNORECASE,
)
_RE_ORG_TAIL = re.compile(
    r"\s+(группа|холдинг|компания|корпорация|банк|университет|институт)$",
    re.IGNORECASE,
)

# Частицы/служебные элементы в ФИО/названиях, которые лучше не склонять
_PERSON_PARTICLES = {"де", "ван", "фон", "аль", "эль", "ибн", "бин", "да", "ди", "дю", "ле", "ла"}


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_semicolon_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in str(s).split(";") if p.strip()]


def join_semicolon_list(items: List[str] | None) -> str:
    if not items:
        return ""
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return ";".join(cleaned)


def _clean(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = _RE_BRACKETS.sub("", s)
    s = _RE_SPACES.sub(" ", s)
    return s.strip()


@lru_cache(maxsize=200_000)
def _parse_best(tok: str):
    """
    Кешируем морфоразбор токена, чтобы ускорить batch-каноникализацию.
    """
    if not _MORPH_AVAILABLE or _MORPH is None:
        return None
    parses = _MORPH.parse(tok)
    if not parses:
        return None
    return parses[0]


def _normal_form_token(tok: str) -> str:
    """
    Нормальная форма (лемма). Используется там, где nomn-инфлекция не нужна/опасна (например ORG).
    """
    t = _clean(tok)
    if not t:
        return ""
    best = _parse_best(t)
    if best is None:
        return t
    return (best.normal_form or t).strip()


def _nomn_token(tok: str) -> str:
    """
    Приводим токен к именительному падежу (nomn), если можем.
    Если не получилось — fallback на normal_form.
    """
    t = _clean(tok)
    if not t:
        return ""

    # не трогаем частицы/служебные элементы
    if t in _PERSON_PARTICLES:
        return t

    # дефисные токены обрабатываем по частям (Сан-Франциско, Жан-Поль)
    if "-" in t:
        parts = [p for p in t.split("-") if p]
        if len(parts) >= 2:
            norm_parts = [_nomn_token(p) for p in parts]
            norm_parts = [p for p in norm_parts if p]
            return "-".join(norm_parts) if norm_parts else t

    best = _parse_best(t)
    if best is None:
        return t

    inf = best.inflect({"nomn"})
    if inf is not None and inf.word:
        return inf.word.strip()

    return (best.normal_form or t).strip()


# -------------------------
# PERSONS
# -------------------------
def normalize_lastname_form(last: str, all_last_forms_raw: Optional[Set[str]] = None) -> str:
    """
    Эвристика X vs Xа/Xя:
    если среди raw-форм встречается X, то Xа/Xя приводим к X.
    """
    l = _clean(last)
    if not l:
        return ""
    if not all_last_forms_raw:
        return l
    if l.endswith("а") and l[:-1] in all_last_forms_raw:
        return l[:-1]
    if l.endswith("я") and l[:-1] in all_last_forms_raw:
        return l[:-1]
    return l


def canon_person_phrase(phrase: str, all_last_forms_raw: Optional[Set[str]] = None) -> str:
    """
    Канон ФИО:
    - чистим
    - выделяем токены
    - приводим токены к nomn
    - для фамилии применяем эвристику X vs Xа/Xя на основе raw-форм батча
    - фильтруем редкий мусор Natasha/NER для PERSON (например, мн. число: "кобылаши")
    """
    phrase = _clean(phrase)
    if not phrase:
        return ""

    tokens = [m.group(0) for m in _TOKEN_RE.finditer(phrase)]
    if not tokens:
        return ""

    norm = [_nomn_token(t) for t in tokens]
    norm = [t for t in norm if t]
    if not norm:
        return ""

    # однословная "персона" (часто фамилия): нормализуем и фильтруем шум
    if len(norm) == 1:
        last = normalize_lastname_form(norm[0], all_last_forms_raw)

        # мусорные PER во множественном числе ("кобылаши", "ивановы" и т.п.)
        if len(last) >= 4 and last.endswith(("ы", "и")):
            return ""

        # базовые ложные срабатывания NER: гео/аббревиатуры как PER
        if last in {"россия", "рф", "ес"}:
            return ""

        return last

    # многословная персона: фамилия — последний токен
    norm[-1] = normalize_lastname_form(norm[-1], all_last_forms_raw)
    return " ".join(norm).strip()


def canon_persons_batch_semicolon(rows: List[str | None]) -> List[str]:
    """
    Batch-level canonicalization для persons:
    1) собираем raw-формы фамилий (без удаления падежей), но расширяем множеством "подсказок":
       - raw last
       - raw last без -а/-я (эвристика)
       - nomn(last) (если морфология доступна)
    2) канонизируем фразы (nomn + normalize_lastname_form)
    3) строим мапу фамилия -> самое частотное ФИО (>=2 токена)
    4) заменяем однословные фамилии на ФИО
    """
    raw_lists: List[List[str]] = []
    all_last_raw: Set[str] = set()

    for s in rows:
        items_raw = [_clean(x) for x in parse_semicolon_list(s)]
        items_raw = [x for x in items_raw if x]
        raw_lists.append(items_raw)

        for ent in items_raw:
            parts = ent.split()
            if not parts:
                continue
            last_raw = _clean(parts[-1])
            if not last_raw:
                continue

            # 1) raw last
            all_last_raw.add(last_raw)

            # 2) эвристика: base без -а/-я
            if len(last_raw) >= 4 and last_raw.endswith(("а", "я")):
                all_last_raw.add(last_raw[:-1])

            # 3) nomn(last), если доступно
            if _MORPH_AVAILABLE and _MORPH is not None:
                all_last_raw.add(_nomn_token(last_raw))

    canon_lists: List[List[str]] = []
    for items_raw in raw_lists:
        canon = [canon_person_phrase(x, all_last_forms_raw=all_last_raw) for x in items_raw]
        canon = [x for x in canon if x]
        canon_lists.append(uniq_keep_order(canon))

    by_last = defaultdict(Counter)
    for plist in canon_lists:
        for p in plist:
            parts = p.split()
            if len(parts) >= 2:
                last = parts[-1]
                by_last[last][p] += 1

    best_full_by_last = {last: cnt.most_common(1)[0][0] for last, cnt in by_last.items()}

    out: List[str] = []
    for plist in canon_lists:
        mapped: List[str] = []
        for p in plist:
            parts = p.split()
            if len(parts) == 1:
                last = normalize_lastname_form(parts[0], all_last_raw)
                mapped.append(best_full_by_last.get(last, last))
            else:
                mapped.append(p)
        out.append(join_semicolon_list(uniq_keep_order(mapped)))

    return out


# -------------------------
# ORGS
# -------------------------
# --- ORG: quality gate ---
_ORG_SINGLE_STOP = {
    # слишком общие типы/роды сущностей, которые NER иногда метит как ORG
    "газета",
    "издание",
    "сми",
    "суд",
    "флот",
    "банк",
    "университет",
    "институт",
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
    "корпорация",
    "компания",
    "холдинг",
    "группа",
}
# однословные прилагательные как ORG — почти всегда шум
_RE_ADJ_SINGLE = re.compile(
    r".+(ский|ская|ское|ские|ской|ском|ских|скому|скую|"
    r"ный|ная|ное|ные|ного|ному|ным|ными|ной|ную)$"
)

# доменные зоны, которые часто приходят отдельным токеном
_TLDS = {"ru", "com", "org", "net", "io", "info", "tv", "ua", "by", "kz", "de", "fr", "uk", "us"}

def _maybe_join_domain(tokens: List[str]) -> List[str]:
    """
    Нормализуем доменные бренды вида "газета ru" / "газеты ru" -> "газета.ru".
    Склейка делается по паттерну [..., WORD, TLD], при этом WORD приводим к normal_form.
    """
    if len(tokens) < 2:
        return tokens

    last = _clean(tokens[-1])
    prev_raw = tokens[-2]

    if last in _TLDS:
        prev = _clean(prev_raw)
        if prev and prev.replace("-", "").isalpha():
            prev_norm = _normal_form_token(prev)  # ключевой момент: "газеты" -> "газета"
            return tokens[:-2] + [f"{prev_norm}.{last}"]

    return tokens

_RE_RU_MARKERS = re.compile(r"(?:^| )(?:рф|россия|российск\w*)(?: |$)")

def _maybe_canon_ru_mod(org: str) -> str:
    """
    Канонизируем 'Министерство обороны ... РФ/России' -> 'минобороны'.
    Важно: без маркеров РФ/России НЕ трогаем (чтобы не приписать всем странам).
    """
    o = _clean(org)
    if not o:
        return ""

    # уже "минобороны" — оставляем
    if o == "минобороны":
        return o

    # ловим варианты после лемматизации: "министерство оборона рф/россия/российский ..."
    if "министерство оборона" in o and _RE_RU_MARKERS.search(o):
        return "минобороны"

    # ещё один частый вариант: "министерство обороны ..." (если морфология/лемматизация дала "обороны")
    if "министерство обороны" in o and _RE_RU_MARKERS.search(o):
        return "минобороны"

    return org

def _org_quality_gate(org: str, tokens: List[str]) -> str:
    o = _clean(org)
    if not o:
        return ""

    if len(tokens) == 1:
        if o in _ORG_SINGLE_STOP:
            return ""
        if _RE_ADJ_SINGLE.match(o):
            return ""

    return org

def canon_org_phrase(phrase: str) -> str:
    """
    ORG-канон:
    - чистим кавычки/ООО/АО и т.п.
    - убираем типовые хвосты ("группа", "банк"...)
    - нормализуем домены: "газета ru" -> "газета.ru"
    - лемматизируем токены через normal_form (без inflect nomn)
    - применяем quality gate (режем только очевидный мусор)
    - применяем словарь синонимов ORG_SYNONYMS (например, "международный уголовный суд" -> "мус")
    """
    s = _clean(phrase)
    if not s:
        return ""

    s = _RE_QUOTES.sub("", s)
    s = _RE_ORG_LEGAL.sub("", s).strip()
    s = _RE_ORG_TAIL.sub("", s).strip()
    s = _RE_SPACES.sub(" ", s)

    tokens = [m.group(0) for m in _TOKEN_RE.finditer(s)]
    if not tokens:
        return ""

    # доменные бренды
    tokens = _maybe_join_domain(tokens)

    # мягкая лемматизация (без nomn)
    norm = [_normal_form_token(t) for t in tokens]
    norm = [t for t in norm if t]
    if not norm:
        return ""

    org = " ".join(norm).strip()

    # quality gate: режем только очевидный мусор
    org = _org_quality_gate(org, norm)
    if not org:
        return ""
    
    org = _maybe_canon_ru_mod(org)
    if not org:
        return ""
    # словарь синонимов/унификаций
    org = ORG_SYNONYMS.get(org, org)
    return org


def canon_orgs_semicolon(s: str | None) -> str:
    items = parse_semicolon_list(s)
    canon = [canon_org_phrase(x) for x in items]
    canon = [x for x in canon if x]
    canon = uniq_keep_order(canon)
    return join_semicolon_list(canon)


# -------------------------
# GEO
# -------------------------
def canon_geo_phrase(phrase: str) -> str:
    """
    GEO-канон:
    - чистим
    - токены приводим к nomn (как для persons)
    - применяем GEO_SYNONYMS (например, "евросоюз" -> "ес", "рф" -> "россия")
    """
    s = _clean(phrase)
    if not s:
        return ""

    tokens = [m.group(0) for m in _TOKEN_RE.finditer(s)]
    if not tokens:
        return ""

    norm = [_nomn_token(t) for t in tokens]
    norm = [t for t in norm if t]
    if not norm:
        return ""

    g = " ".join(norm).strip()
    return GEO_SYNONYMS.get(g, g)


def canon_geo_semicolon(s: str | None) -> str:
    items = parse_semicolon_list(s)
    canon = [canon_geo_phrase(x) for x in items]
    canon = [x for x in canon if x]
    canon = uniq_keep_order(canon)
    return join_semicolon_list(canon)
