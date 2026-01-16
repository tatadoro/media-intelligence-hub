from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import pandas as pd

from src.processing.nlp_extras import add_lang_keyphrases_sentiment
from src.processing.summarization import enrich_articles_with_summary_and_keywords

CH_TZ = "Europe/Moscow"
MIN_PUBLISHED_AT_UTC = pd.Timestamp("2000-01-01", tz="UTC")

# Фильтрация низкокачественного контента перед обогащением
EXCLUDE_BODY_STATUS = {"parsed_empty", "fetch_failed", "no_link", "too_short"}
MIN_CLEAN_TEXT_CHARS = 400

# --- NER (persons/geo) ---
try:
    from natasha import Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter

    _NER_AVAILABLE = True
    _segmenter = Segmenter()
    _emb = NewsEmbedding()
    _tagger = NewsNERTagger(_emb)
    _morph = MorphVocab()
except Exception:
    _NER_AVAILABLE = False

# --- Morphology for canonicalization + verbs ---
try:
    import pymorphy2

    _MORPH_AVAILABLE = True
    _morph_analyzer = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH_AVAILABLE = False
    _morph_analyzer = None  # type: ignore[assignment]

_RE_BRACKETS = re.compile(r"\s*\([^)]*\)")
_RE_SPACES = re.compile(r"\s+")

# --- Actions extraction (persons -> verbs) ---
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")

STOP_VERBS_DEFAULT = {"быть", "стать", "мочь", "иметь", "сказать", "говорить"}

# Синонимы/унификация для гео (минимально, расширишь по мере надобности)
GEO_SYNONYMS: Dict[str, str] = {
    "рф": "россия",
    "российский федерация": "россия",
    "российская федерация": "россия",
    "соединенный штат": "сша",
    "соединенные штат": "сша",
    "соединенные штаты": "сша",
}


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _clean_entity_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _RE_BRACKETS.sub("", s)  # убрать "(...)"
    s = s.replace("ё", "е")
    s = _RE_SPACES.sub(" ", s)
    return s.strip()


def _lemma_surname_nomn(token: str) -> str:
    """
    Приводим фамилию к именительному падежу (nomn), если можем.
    Если не получилось — fallback на normal_form.
    """
    t = (token or "").strip().lower().replace("ё", "е")
    if not t:
        return ""
    if not _MORPH_AVAILABLE or _morph_analyzer is None:
        return t

    parses = _morph_analyzer.parse(t)
    if not parses:
        return t

    best = parses[0]
    inf = best.inflect({"nomn"})
    if inf is not None and inf.word:
        return inf.word
    return best.normal_form


def _lemmatize_phrase(s: str) -> str:
    """
    Лемматизация фразы для схлопывания падежей.
    Для ФИО: последний токен (фамилия) стараемся привести к именительному падежу (nomn).
    """
    s = _clean_entity_text(s)
    if not s:
        return ""

    if not _MORPH_AVAILABLE or _morph_analyzer is None:
        return s

    tokens = [t for t in re.split(r"[^\w-]+", s) if t]
    if not tokens:
        return ""

    lemmas: List[str] = []
    for i, t in enumerate(tokens):
        if t in {"сша", "рф", "ес"}:
            lemmas.append(t)
            continue

        # если это последняя часть фразы из 2+ слов — считаем, что это фамилия
        if len(tokens) >= 2 and i == len(tokens) - 1:
            lemmas.append(_lemma_surname_nomn(t))
            continue

        p = _morph_analyzer.parse(t) if _morph_analyzer is not None else []
        lemmas.append(p[0].normal_form if p else t)

    return " ".join(lemmas).strip()


def _normalize_lastname_form(last: str, all_last_forms: set[str]) -> str:
    """
    Практичная эвристика:
    если встречаются формы X и Xа (или X и Xя), то canonical = X.
    Это чинит "кобылаша" -> "кобылаш" при наличии "кобылаш" в исходных сущностях.
    """
    l = (last or "").strip().lower().replace("ё", "е")
    if not l:
        return ""

    # генитив мужских фамилий часто добавляет "а/я"
    if l.endswith("а") and l[:-1] in all_last_forms:
        return l[:-1]
    if l.endswith("я") and l[:-1] in all_last_forms:
        return l[:-1]

    return l


def _canonicalize_geo(geo_mentions: List[str]) -> List[str]:
    out: List[str] = []
    for g in geo_mentions:
        g0 = _lemmatize_phrase(g)
        if not g0:
            continue
        g0 = GEO_SYNONYMS.get(g0, g0)
        out.append(g0)
    return _uniq_keep_order(out)


def _canonicalize_persons(person_mentions: List[str]) -> List[str]:
    """
    Локальная каноникализация (в пределах одного текста).
    Для итоговой склейки one-word -> full используем глобальный проход в add_persons_geo_columns().
    """
    canon = [_lemmatize_phrase(p) for p in person_mentions]
    canon = [p for p in canon if p]
    return _uniq_keep_order(canon)


def extract_persons_geo_lists(text: str) -> Tuple[List[str], List[str]]:
    """
    Возвращает (persons_list, geo_list) как списки в канонической форме:
    - clean + lemmatize
    - удаление "(...)" хвостов
    - geo дополнительно прогоняется через словарь GEO_SYNONYMS

    Важно: здесь НЕ делаем глобальную склейку фамилии -> полное имя.
    Это делается вторым проходом на уровне всего батча в add_persons_geo_columns().
    """
    if not _NER_AVAILABLE:
        return [], []

    if not text or not str(text).strip():
        return [], []

    doc = Doc(str(text))
    doc.segment(_segmenter)
    doc.tag_ner(_tagger)

    persons_raw: List[str] = []
    geo_raw: List[str] = []

    for span in doc.spans:
        span.normalize(_morph)
        raw = (span.normal or span.text or "").strip()
        if not raw:
            continue

        val = _lemmatize_phrase(raw)
        if not val:
            continue

        if span.type == "PER":
            persons_raw.append(val)
        elif span.type == "LOC":
            geo_raw.append(val)

    persons = _canonicalize_persons(persons_raw)
    geo = _canonicalize_geo(geo_raw)

    return persons, geo


def extract_persons_geo(text: str) -> Tuple[str, str]:
    """
    Совместимость: возвращает (persons, geo) как строки 'a;b;c' в нижнем регистре.
    """
    persons, geo = extract_persons_geo_lists(text)
    return ";".join(persons), ";".join(geo)


def add_persons_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонки persons и geo в gold.
    Берём текст в приоритете: nlp_text -> clean_text -> raw_text.

    Для persons: 2 прохода на уровне всего датафрейма (батча):
    1) извлечь списки персон по каждой строке (лемматизация, чистка)
    2) собрать глобальную мапу: фамилия -> самое частотное полное имя (>=2 токена)
    3) заменить однословные упоминания (трамп) на полные (дональд трамп), если есть
    """
    out = df.copy()

    if out.empty:
        if "persons" not in out.columns:
            out["persons"] = pd.Series(dtype="string")
        if "geo" not in out.columns:
            out["geo"] = pd.Series(dtype="string")
        return out

    if "nlp_text" in out.columns:
        text_s = out["nlp_text"]
    elif "clean_text" in out.columns:
        text_s = out["clean_text"]
    else:
        text_s = out.get("raw_text", pd.Series([""] * len(out), index=out.index))

    text_s = text_s.fillna("").astype(str)

    persons_lists: List[List[str]] = []
    geo_lists: List[List[str]] = []
    for t in text_s.tolist():
        p_list, g_list = extract_persons_geo_lists(t)
        persons_lists.append(p_list)
        geo_lists.append(g_list)

    # 2) Глобальная мапа фамилия -> самое частотное полное имя (>=2 токена)
    by_last: DefaultDict[str, Counter] = defaultdict(Counter)
    for plist in persons_lists:
        for p in plist:
            parts = p.split()
            if len(parts) >= 2:
                last = parts[-1].strip().lower()
                if last:
                    by_last[last][p] += 1

    best_full_by_last: Dict[str, str] = {last: cnt.most_common(1)[0][0] for last, cnt in by_last.items()}

    persons_out: List[str] = []
    geo_out: List[str] = []

    for plist, glist in zip(persons_lists, geo_lists):
        mapped: List[str] = []
        for p in plist:
            parts = p.split()
            if len(parts) == 1:
                mapped.append(best_full_by_last.get(parts[0], p))
            else:
                mapped.append(p)

        persons_out.append(";".join(_uniq_keep_order(mapped)))
        geo_out.append(";".join(_uniq_keep_order(glist)))

    out["persons"] = persons_out
    out["geo"] = geo_out
    return out


def add_entities_persons_canon_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    canonical для entities_persons:
    - для набора возможных фамилий используем raw-формы (без морфологии),
      чтобы не потерять "кобылаш" (если морфология превращает в "кобылаша")
    - затем: лемматизация/чистка + склейка фамилия->ФИО + эвристика X vs Xа/Xя
    """
    out = df.copy()

    if out.empty:
        if "entities_persons_canon" not in out.columns:
            out["entities_persons_canon"] = pd.Series(dtype="string")
        return out

    src_col = "entities_persons" if "entities_persons" in out.columns else ("persons" if "persons" in out.columns else None)
    if src_col is None:
        out["entities_persons_canon"] = ""
        return out

    src_s = out[src_col].fillna("").astype(str)

    # PASS 0: соберём raw формы фамилий БЕЗ морфологии (это важно)
    all_last_forms_raw: set[str] = set()
    raw_lists: List[List[str]] = []
    for s in src_s.tolist():
        items_raw = [x.strip() for x in s.split(";") if x.strip()]
        items_raw = [_clean_entity_text(x) for x in items_raw]
        items_raw = [x for x in items_raw if x]
        raw_lists.append(items_raw)
        for ent in items_raw:
            parts = ent.split()
            if parts:
                all_last_forms_raw.add(parts[-1])

    # PASS 1: лемматизация/чистка -> списки (для дальнейшей склейки)
    persons_lists: List[List[str]] = []
    for items_raw in raw_lists:
        items = [_lemmatize_phrase(x) for x in items_raw]
        items = [x for x in items if x]
        persons_lists.append(_uniq_keep_order(items))

    # строим глобальную мапу фамилия -> самое частотное полное имя (>=2 токена)
    by_last: DefaultDict[str, Counter] = defaultdict(Counter)
    for plist in persons_lists:
        for p in plist:
            parts = p.split()
            if len(parts) >= 2:
                last_norm = _normalize_lastname_form(parts[-1], all_last_forms_raw)
                full = " ".join(parts[:-1] + [last_norm]).strip()
                by_last[last_norm][full] += 1

    best_full_by_last: Dict[str, str] = {last: cnt.most_common(1)[0][0] for last, cnt in by_last.items()}

    # PASS 2: нормализуем фамилии и маппим однословные на ФИО
    canon_out: List[str] = []
    for plist in persons_lists:
        mapped: List[str] = []
        for p in plist:
            parts = p.split()
            if len(parts) == 1:
                last_norm = _normalize_lastname_form(parts[0], all_last_forms_raw)
                mapped.append(best_full_by_last.get(last_norm, last_norm))
            else:
                last_norm = _normalize_lastname_form(parts[-1], all_last_forms_raw)
                mapped.append(" ".join(parts[:-1] + [last_norm]).strip())

        canon_out.append(";".join(_uniq_keep_order(mapped)))

    out["entities_persons_canon"] = canon_out
    return out


def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)
    return [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]


@lru_cache(maxsize=50_000)
def _parse_tok(tok: str):
    if not _MORPH_AVAILABLE or _morph_analyzer is None:
        return None
    return _morph_analyzer.parse(tok)[0]


def _extract_verbs_from_tokens(tokens: List[str], stop_verbs: set[str]) -> List[str]:
    if not _MORPH_AVAILABLE or _morph_analyzer is None:
        return []

    out: List[str] = []
    for tok in tokens:
        p = _parse_tok(tok)
        if p is None:
            continue
        pos = p.tag.POS
        if pos in {"VERB", "INFN"}:
            lemma = p.normal_form
            if lemma and lemma not in stop_verbs:
                out.append(lemma)
    return out


def _surname_variants_for_matching(last: str) -> List[str]:
    """
    Для матчинга в тексте добавляем простые русские падежные варианты фамилии:
    X -> X, Xа, Xя (чтобы 'кобылаш' матчился на 'кобылаша').
    Это именно ДЛЯ ПОИСКА в тексте, ключ персоны остаётся каноническим.
    """
    l = (last or "").strip().lower().replace("ё", "е")
    if not l:
        return []
    variants = [l]

    # если уже оканчивается на 'а/я' — оставим как есть
    if not l.endswith(("а", "я")):
        variants.append(l + "а")
        variants.append(l + "я")

    return _uniq_keep_order([v for v in variants if v])


def _compile_person_patterns(persons: List[str]) -> Dict[str, List[re.Pattern[str]]]:
    """
    Компилирует паттерны для матчинга персон в тексте:
    - полное имя
    - фамилия (если есть >= 2 токена) + простые падежные варианты фамилии

    Важно: ключ в словаре — ВСЕГДА полное имя p0 (канон),
    фамилия/падежи — только варианты для поиска.
    """
    out: Dict[str, List[re.Pattern[str]]] = {}
    for p in persons:
        p0 = (p or "").strip().lower().replace("ё", "е")
        if not p0:
            continue

        parts = p0.split()
        variants: List[str] = [p0]

        if len(parts) >= 2:
            last = parts[-1]
            variants.extend(_surname_variants_for_matching(last))

        pats: List[re.Pattern[str]] = []
        for v in dict.fromkeys(variants):
            v = (v or "").strip()
            if not v:
                continue
            pats.append(re.compile(rf"(?<!\w){re.escape(v)}(?!\w)"))

        if pats:
            out[p0] = pats

    return out


def _extract_actions_for_persons(text: str, persons: List[str]) -> Dict[str, List[str]]:
    """
    person -> list of verb lemmas extracted from sentences where the person is mentioned.
    """
    t = (text or "").strip()
    if not t or not persons:
        return {}

    persons_norm: List[str] = []
    for p in persons:
        p0 = (p or "").strip().lower().replace("ё", "е")
        if p0:
            persons_norm.append(p0)

    persons_norm = _uniq_keep_order(persons_norm)
    if not persons_norm:
        return {}

    person_patterns = _compile_person_patterns(persons_norm)
    if not person_patterns:
        return {}

    res: Dict[str, List[str]] = {p: [] for p in person_patterns.keys()}
    stop_verbs = set(STOP_VERBS_DEFAULT)

    for sent in _split_sentences(t):
        s_low = sent.lower().replace("ё", "е")

        matched: List[str] = [p for p, pats in person_patterns.items() if any(pat.search(s_low) for pat in pats)]
        if not matched:
            continue

        tokens = [m.group(0).lower().replace("ё", "е") for m in _TOKEN_RE.finditer(sent)]
        verbs = _extract_verbs_from_tokens(tokens, stop_verbs)
        if not verbs:
            continue

        for p in matched:
            res[p].extend(verbs)

    return {p: v for p, v in res.items() if v}


def _drop_oneword_surnames_if_full_present(persons: List[str]) -> List[str]:
    """
    Удаляет однословные упоминания фамилии, если в списке уже есть полное имя (>=2 токена)
    с этой фамилией. Работает с простыми падежными вариантами типа "кобылаш/кобылаша".
    """
    persons = _uniq_keep_order([p.strip() for p in persons if p and p.strip()])
    full = [p for p in persons if len(p.split()) >= 2]
    if not full:
        return persons

    def _base_last(x: str) -> str:
        x = (x or "").strip().lower().replace("ё", "е")
        if x.endswith(("а", "я")) and len(x) > 2:
            return x[:-1]
        return x

    full_last_bases = set()
    for p in full:
        last = p.split()[-1]
        full_last_bases.add(_base_last(last))
        full_last_bases.add(last)

    out: List[str] = []
    for p in persons:
        parts = p.split()
        if len(parts) >= 2:
            out.append(p)
            continue

        one = parts[0]
        if _base_last(one) in full_last_bases or one in full_last_bases:
            continue

        out.append(p)

    return _uniq_keep_order(out)


def add_persons_actions_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет:
    - persons_actions: строка формата "person:verb1,verb2|person2:verb1,..."
    - actions_verbs: общий top глаголов по статье (через ;)

    Безопасно: если нет pymorphy2 или persons пуст — колонки будут пустыми строками.
    """
    out = df.copy()

    if out.empty:
        out["persons_actions"] = pd.Series(dtype="string")
        out["actions_verbs"] = pd.Series(dtype="string")
        return out

    # текст для анализа
    if "nlp_text" in out.columns:
        text_s = out["nlp_text"]
    elif "clean_text" in out.columns:
        text_s = out["clean_text"]
    else:
        text_s = out.get("raw_text", pd.Series([""] * len(out), index=out.index))

    text_s = text_s.fillna("").astype(str)

    # persons source priority:
    # 1) entities_persons_canon (лучше: уже склеено и меньше дублей)
    # 2) persons (эвристики пайплайна)
    # 3) entities_persons (сырые из NER)
    if "entities_persons_canon" in out.columns:
        persons_s = out["entities_persons_canon"]
    elif "persons" in out.columns:
        persons_s = out["persons"]
    elif "entities_persons" in out.columns:
        persons_s = out["entities_persons"]
    else:
        persons_s = pd.Series([""] * len(out), index=out.index)

    persons_s = persons_s.fillna("").astype(str)

    persons_actions_out: List[str] = []
    actions_verbs_out: List[str] = []

    for t, persons_str in zip(text_s.tolist(), persons_s.tolist()):
        persons = [p.strip() for p in persons_str.split(";") if p.strip()]
        persons = _drop_oneword_surnames_if_full_present(persons)

        m = _extract_actions_for_persons(t, persons)

        parts: List[str] = []
        flat: List[str] = []

        for person in sorted(m.keys()):
            cnt = Counter(m[person])
            top = [v for v, _ in cnt.most_common(10)]
            if top:
                parts.append(f"{person}:{','.join(top)}")
                flat.extend(m[person])

        persons_actions_out.append("|".join(parts) if parts else "")

        if flat:
            top_all = [v for v, _ in Counter(flat).most_common(15)]
            actions_verbs_out.append(";".join(top_all))
        else:
            actions_verbs_out.append("")

    out["persons_actions"] = persons_actions_out
    out["actions_verbs"] = actions_verbs_out
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Преобразование silver -> gold: summary + keywords (TF-IDF) и выгрузка в MinIO (опционально)."
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Путь к silver-файлу (json/jsonl/parquet/csv).")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Куда сохранить gold Parquet. Если не указан — путь будет выведен из input (data/gold/..._processed.parquet).",
    )
    parser.add_argument(
        "--upload-minio",
        action="store_true",
        help="Если указан, после сохранения gold-файла загрузить его в MinIO (bucket media-intel, prefix gold/).",
    )
    parser.add_argument(
        "--with-actions",
        action="store_true",
        help="Если указан, извлекаем глаголы-действия для персон (persons_actions/actions_verbs).",
    )
    return parser.parse_args()


def infer_output_path(input_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    gold_dir = project_root / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.name
    for suf in (".jsonl", ".json", ".parquet", ".csv"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    return gold_dir / f"{stem}_processed.parquet"


def load_silver_df(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(input_path, lines=(suffix == ".jsonl"))
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


def normalize_published_at(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "published_at" not in out.columns:
        out["published_at"] = None

    published_at_utc = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
    published_at_utc = published_at_utc.where(published_at_utc >= MIN_PUBLISHED_AT_UTC)

    published_at_msk = published_at_utc.dt.tz_convert(CH_TZ).dt.tz_localize(None)
    out["published_at"] = published_at_msk
    return out


def _safe_col_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df), index=df.index)


def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "id" not in out.columns:
        out["id"] = ""

    id_s = out["id"].fillna("").astype(str).str.strip()
    if (id_s != "").all():
        return out

    uid_s = (
        out["uid"].fillna("").astype(str).str.strip()
        if "uid" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    link_s = (
        out["link"].fillna("").astype(str).str.strip()
        if "link" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    src_s = (
        out["source"].fillna("").astype(str).str.strip()
        if "source" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    ttl_s = (
        out["title"].fillna("").astype(str).str.strip()
        if "title" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    pub_s = out["published_at"].astype(str) if "published_at" in out.columns else pd.Series([""] * len(out), index=out.index)

    def md5_hex(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    mask = id_s == ""

    use_uid = mask & (uid_s != "")
    if use_uid.any():
        out.loc[use_uid, "id"] = uid_s[use_uid]

    id_s2 = out["id"].fillna("").astype(str).str.strip()
    mask2 = id_s2 == ""

    use_link = mask2 & (link_s != "")
    if use_link.any():
        out.loc[use_link, "id"] = link_s[use_link].map(md5_hex)

    id_s3 = out["id"].fillna("").astype(str).str.strip()
    mask3 = id_s3 == ""
    if mask3.any():
        key = (src_s + "|" + ttl_s + "|" + pub_s).fillna("").astype(str)
        out.loc[mask3, "id"] = key[mask3].map(md5_hex)

    return out


def dedup_batch_silver(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _dedup_key(row: pd.Series) -> str:
        src = str(row.get("source") or "")
        link = str(row.get("link") or "")
        if link:
            return f"{src}|{link}"
        title = str(row.get("title") or "")
        published = str(row.get("published_at") or "")
        return f"{src}|{title}|{published}"

    out["dedup_key"] = out.apply(_dedup_key, axis=1)

    clean_text_s = _safe_col_series(out, "clean_text").fillna("").astype(str)
    raw_text_s = _safe_col_series(out, "raw_text").fillna("").astype(str)
    body_s = _safe_col_series(out, "body").fillna("").astype(str)

    out["clean_len"] = clean_text_s.str.len()
    out["has_body"] = (raw_text_s.str.len() > 0) | (body_s.str.len() > 0)
    out["published_at_dt"] = pd.to_datetime(out.get("published_at"), errors="coerce")

    before = len(out)
    out = out.sort_values(
        by=["dedup_key", "has_body", "clean_len", "published_at_dt"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["dedup_key"], keep="first")
    after = len(out)

    print(f"[INFO] Dedup inside batch (silver): {before} -> {after} (dropped {before - after})")

    return out.drop(columns=["dedup_key", "clean_len", "has_body", "published_at_dt"], errors="ignore")


def filter_low_quality_for_gold(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    before = len(out)

    if "body_status" in out.columns:
        mask_bad = out["body_status"].astype(str).str.strip().isin(EXCLUDE_BODY_STATUS)
        dropped = int(mask_bad.sum())
        if dropped:
            out = out.loc[~mask_bad].copy()
            print(f"[INFO] Drop by body_status: {dropped} rows (excluded={sorted(EXCLUDE_BODY_STATUS)})")

    clean_text_s = _safe_col_series(out, "clean_text").fillna("").astype(str)
    clean_len = clean_text_s.str.len()
    mask_short = clean_len < MIN_CLEAN_TEXT_CHARS
    dropped = int(mask_short.sum())
    if dropped:
        out = out.loc[~mask_short].copy()
        print(f"[INFO] Drop by clean_text length: {dropped} rows (min_len={MIN_CLEAN_TEXT_CHARS})")

    after = len(out)
    print(f"[INFO] Quality filter before gold: {before} -> {after} (dropped {before - after})")
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")

    print("[INFO] Преобразуем silver -> gold (summary + keywords)")
    print(f"silver: {input_path}")

    df_silver = load_silver_df(input_path)

    # 1) нормализуем время
    df_silver = normalize_published_at(df_silver)

    # 1.1) гарантируем id
    df_silver = ensure_id_column(df_silver)

    # 2) dedup внутри батча ДО обогащения
    df_silver = dedup_batch_silver(df_silver)

    # 3) фильтрация низкокачественных записей
    df_silver = filter_low_quality_for_gold(df_silver)

    # output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = infer_output_path(input_path)

    if df_silver.empty:
        print("[WARN] После фильтрации quality не осталось строк. Сохраняем пустой gold.")
        df_gold = df_silver.copy()

        for col, dtype in [
            ("summary", "string"),
            ("keywords", "object"),
            ("persons", "string"),
            ("geo", "string"),
            ("entities_persons_canon", "string"),
            ("lang", "string"),
            ("keyphrases", "string"),
            ("sentiment_label", "string"),
            ("sentiment_score", "float"),
        ]:
            if col not in df_gold.columns:
                df_gold[col] = pd.Series(dtype=dtype)

        if args.with_actions:
            if "persons_actions" not in df_gold.columns:
                df_gold["persons_actions"] = pd.Series(dtype="string")
            if "actions_verbs" not in df_gold.columns:
                df_gold["actions_verbs"] = pd.Series(dtype="string")
    else:
        df_gold = enrich_articles_with_summary_and_keywords(df_silver)

        # NER persons/geo (внутренние колонки)
        df_gold = add_persons_geo_columns(df_gold)

        # canonical именно для entities_persons (для BI/ClickHouse)
        df_gold = add_entities_persons_canon_column(df_gold)

        if args.with_actions:
            df_gold = add_persons_actions_columns(df_gold)

        df_gold = add_lang_keyphrases_sentiment(df_gold)

        # гарантируем id в gold перед валидацией/загрузкой
        df_gold = ensure_id_column(df_gold)

    print(f"gold:  {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_gold.to_parquet(output_path, index=False)

    if args.upload_minio:
        import subprocess

        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        bucket = os.getenv("MINIO_BUCKET", "media-intel")
        alias = os.getenv("MINIO_ALIAS", "local")

        if not minio_access_key or not minio_secret_key:
            raise ValueError(
                "Не заданы MINIO_ACCESS_KEY/MINIO_SECRET_KEY. Заполни .env (см. .env.example) и повтори."
            )

        try:
            subprocess.run(["mc", "--version"], check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise FileNotFoundError("Команда 'mc' не найдена. Установи MinIO Client (mc) и повтори.") from e

        out_aliases = subprocess.run(["mc", "alias", "list"], check=True, capture_output=True, text=True).stdout
        if f"{alias} " not in out_aliases and f"{alias}\t" not in out_aliases:
            subprocess.run(
                ["mc", "alias", "set", alias, minio_endpoint, minio_access_key, minio_secret_key],
                check=True,
                capture_output=True,
                text=True,
            )

        dst = f"{alias}/{bucket}/gold/{output_path.name}"
        print(f"[INFO] Загружаем gold в MinIO: {dst}")
        subprocess.run(["mc", "cp", str(output_path), dst], check=True)
        print("[OK] Gold загружен в MinIO")


if __name__ == "__main__":
    main()