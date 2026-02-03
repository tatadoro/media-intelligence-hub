from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Callable, Optional, Iterable

import pandas as pd

from src.processing.entity_canon import (
    canon_geo_semicolon,
    canon_orgs_semicolon,
    canon_persons_batch_semicolon,
)
from src.processing.nlp_extras import add_lang_keyphrases_sentiment
from src.processing.ner import extract_persons_orgs_geo as ner_extract_persons_orgs_geo
from src.processing.summarization import enrich_articles_with_summary_and_keywords

CH_TZ = "Europe/Moscow"
MIN_PUBLISHED_AT_UTC = pd.Timestamp("2000-01-01", tz="UTC")

# Фильтрация низкокачественного контента перед обогащением
EXCLUDE_BODY_STATUS = {"parsed_empty", "fetch_failed", "no_link", "too_short"}
MIN_CLEAN_TEXT_CHARS = 400

# -----------------------------
# Actions extraction (persons -> verbs)
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")

# Расширенный стоп-лист (минимальный MVP-набор; расширяй по мере надобности)
STOP_VERBS_DEFAULT = {
    "быть",
    "стать",
    "мочь",
    "иметь",
    "являться",
    "находиться",
    "считаться",
    "сказать",
    "говорить",
    "сообщить",
    "заявить",
    "отметить",
    "подчеркнуть",
    "написать",
    "рассказать",
    "пояснить",
    "уточнить",
    "добавить",
    "признать",
    "объявить",
}

# Нормализация “речевых” глаголов в один тип (чтобы уменьшить шум в BI)
VERB_CANON = {
    "сказать": "заявить",
    "говорить": "заявить",
    "сообщить": "заявить",
    "отметить": "заявить",
    "подчеркнуть": "заявить",
    "заявить": "заявить",
}

# --- Morphology for verbs (опционально) ---
try:
    import pymorphy3  # type: ignore

    _MORPH_AVAILABLE = True
    _morph_analyzer = pymorphy3.MorphAnalyzer(lang="ru")
except Exception:
    _MORPH_AVAILABLE = False
    _morph_analyzer = None  # type: ignore[assignment]


def _safe_apply(
    df: pd.DataFrame,
    fn: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
    step_name: str,
) -> pd.DataFrame:
    """
    Защита от случаев, когда функция по ошибке возвращает None.
    Это предотвращает падение на следующих шагах (например, ensure_id_column(None)).
    """
    res = fn(df)
    if res is None:
        print(f"[WARN] {step_name} вернула None. Оставляем DataFrame без изменений.")
        return df
    if not isinstance(res, pd.DataFrame):
        raise TypeError(f"[ERROR] {step_name} вернула {type(res)} вместо pd.DataFrame")
    return res


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _print_nlp_health(df: pd.DataFrame) -> None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return

    total = len(df)

    def _ratio_ok(col: str) -> str:
        if col not in df.columns:
            return "n/a"
        s = df[col].fillna(0)
        try:
            ok = (s.astype(int) > 0).sum()
        except Exception:
            ok = (s == 1).sum()
        return f"{ok}/{total} ({ok / total:.0%})"

    def _ratio_lang() -> str:
        if "lang" not in df.columns:
            return "n/a"
        s = df["lang"].fillna("").astype(str)
        ok = ((s != "") & (s != "unknown")).sum()
        return f"{ok}/{total} ({ok / total:.0%})"

    print(
        "[NLP] health: "
        f"lang_ok={_ratio_lang()} "
        f"keyphrases_ok={_ratio_ok('keyphrases_ok')} "
        f"sentiment_ok={_ratio_ok('sentiment_ok')}"
    )


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
    """
    Извлекает леммы глаголов (VERB/INFN) из набора токенов.
    """
    if not _MORPH_AVAILABLE or _morph_analyzer is None:
        return []

    out: List[str] = []
    for tok in tokens:
        p = _parse_tok(tok)
        if p is None:
            continue
        pos = p.tag.POS
        if pos in {"VERB", "INFN"}:
            lemma = (p.normal_form or "").strip()
            if not lemma:
                continue
            lemma = VERB_CANON.get(lemma, lemma)
            if lemma in stop_verbs:
                continue
            out.append(lemma)
    return out


def _surname_variants_for_matching(last: str) -> List[str]:
    """
    Для матчинга в тексте добавляем простые падежные варианты фамилии:
    X -> X, Xа, Xя (чтобы 'кобылаш' матчился на 'кобылаша').
    Это именно ДЛЯ ПОИСКА в тексте, ключ персоны остаётся каноническим.
    """
    l = (last or "").strip().lower().replace("ё", "е")
    if not l:
        return []
    variants = [l]
    if not l.endswith(("а", "я")):
        variants.append(l + "а")
        variants.append(l + "я")
    return _uniq_keep_order([v for v in variants if v])


def _compile_person_patterns(persons: List[str]) -> Dict[str, List[re.Pattern[str]]]:
    """
    Компилирует паттерны для матчинга персон в тексте:
    - полное имя
    - фамилия (если есть >= 2 токена) + простые падежные варианты фамилии

    Ключ в словаре — каноническая персона (p0),
    варианты — только для поиска.
    """
    out: Dict[str, List[re.Pattern[str]]] = {}
    for p in persons:
        p0 = (p or "").strip().lower().replace("ё", "е")
        if not p0:
            continue

        parts = p0.split()
        variants: List[str] = [p0]
        if len(parts) >= 2:
            variants.extend(_surname_variants_for_matching(parts[-1]))

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

    MVP-логика:
    - ищем упоминания (полное имя/фамилия/простейшие падежные варианты фамилии)
    - берём глаголы из предложения
    """
    t = (text or "").strip()
    if not t or not persons:
        return {}

    persons_norm = _uniq_keep_order(
        [(p or "").strip().lower().replace("ё", "е") for p in persons if (p or "").strip()]
    )
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

    Безопасно: если нет pymorphy3 или persons пуст — колонки будут пустыми строками.
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
    # 1) entities_persons_canon
    # 2) persons
    # 3) entities_persons
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


# -----------------------------
# Entities extraction + canon columns
# -----------------------------
def add_entities_persons_geo_from_text_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Гарантирует наличие entities_persons/entities_orgs/entities_geo.
    Если колонок нет — извлекаем NER из текста (nlp_text -> clean_text -> raw_text).
    """
    out = df.copy()

    # Если все колонки есть — ничего не делаем
    if (
        "entities_persons" in out.columns
        and "entities_geo" in out.columns
        and "entities_orgs" in out.columns
    ):
        return out

    if out.empty:
        if "entities_persons" not in out.columns:
            out["entities_persons"] = pd.Series(dtype="string")
        if "entities_orgs" not in out.columns:
            out["entities_orgs"] = pd.Series(dtype="string")
        if "entities_geo" not in out.columns:
            out["entities_geo"] = pd.Series(dtype="string")
        return out

    if "nlp_text" in out.columns:
        text_s = out["nlp_text"]
    elif "clean_text" in out.columns:
        text_s = out["clean_text"]
    else:
        text_s = out.get("raw_text", pd.Series([""] * len(out), index=out.index))

    text_s = text_s.fillna("").astype(str)

    persons_out: List[str] = []
    orgs_out: List[str] = []
    geo_out: List[str] = []

    for t in text_s.tolist():
        p, o, g = ner_extract_persons_orgs_geo(t)
        persons_out.append(p or "")
        orgs_out.append(o or "")
        geo_out.append(g or "")

    # не затираем существующее, добавляем только отсутствующее
    if "entities_persons" not in out.columns:
        out["entities_persons"] = persons_out
    if "entities_orgs" not in out.columns:
        out["entities_orgs"] = orgs_out
    if "entities_geo" not in out.columns:
        out["entities_geo"] = geo_out

    return out


def add_entities_persons_canon_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает батчевую склейку персон:
    - лемматизация/нормализация
    - фамилия -> самое частотное полное имя в батче
    """
    out = df.copy()
    if "entities_persons" not in out.columns:
        out["entities_persons_canon"] = ""
        return out

    out["entities_persons_canon"] = canon_persons_batch_semicolon(
        out["entities_persons"].fillna("").astype(str).tolist()
    )
    return out


def add_entities_orgs_canon_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Каноникализация организаций (если entities_orgs присутствует).
    """
    out = df.copy()
    if "entities_orgs" not in out.columns:
        out["entities_orgs"] = ""
        out["entities_orgs_canon"] = ""
        return out
    out["entities_orgs_canon"] = out["entities_orgs"].fillna("").astype(str).map(canon_orgs_semicolon)
    return out


def add_entities_geo_canon_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Каноникализация гео (если entities_geo присутствует).
    """
    out = df.copy()
    if "entities_geo" not in out.columns:
        out["entities_geo"] = ""
        out["entities_geo_canon"] = ""
        return out
    out["entities_geo_canon"] = out["entities_geo"].fillna("").astype(str).map(canon_geo_semicolon)
    return out


# -----------------------------
# CLI + IO
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Преобразование silver -> gold: summary + keywords и доп. NLP колонки."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Файл/директория/glob для silver (json/jsonl/parquet/csv). Пример: data/silver/articles_*_clean.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help=(
            "Куда сохранить gold. "
            "Если вход один — можно указать файл .parquet. "
            "Если входов несколько — укажи директорию, и файлы будут сохранены как <stem>_processed.parquet."
        ),
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
    """
    Гарантирует колонку id.
    Приоритет:
    1) uid (если есть и непустой)
    2) md5(link) (если link есть)
    3) md5(source|title|published_at)
    """
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
    pub_s = (
        out["published_at"].astype(str)
        if "published_at" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )

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


def _expand_input_to_files(input_arg: str, project_root: Path) -> List[Path]:
    """
    Разворачивает --input в список существующих файлов.
    Поддерживает:
      - конкретный файл
      - директорию (берём *.json/*.jsonl/*.parquet/*.csv)
      - glob (например data/silver/articles_*_clean.json)
    Все относительные пути считаем относительно project_root.
    """
    s = str(input_arg).strip()
    if not s:
        return []

    # Если есть wildcard — обрабатываем как glob
    has_glob = any(ch in s for ch in ["*", "?", "["])

    if has_glob:
        pat = s
        if not Path(pat).is_absolute():
            pat = str(project_root / pat)
        files = sorted([Path(x) for x in glob.glob(pat) if Path(x).is_file()])
        return files

    p = Path(s)
    if not p.is_absolute():
        p = project_root / p

    if p.exists() and p.is_dir():
        exts = (".json", ".jsonl", ".parquet", ".csv")
        files: List[Path] = []
        for ext in exts:
            files.extend([x for x in p.glob(f"*{ext}") if x.is_file()])
        return sorted(files)

    if p.exists() and p.is_file():
        return [p]

    # На всякий случай пробуем как glob даже без wildcard (иногда передают "path/{a,b}.json" и т.п.)
    files2 = sorted([Path(x) for x in glob.glob(str(p)) if Path(x).is_file()])
    return files2


def _resolve_output_paths(
    input_files: List[Path],
    output_arg: Optional[str],
    project_root: Path,
) -> List[Path]:
    """
    Возвращает список output_path той же длины, что и input_files.
    Правила:
      - output_arg is None: infer_output_path(input_file) для каждого
      - один input + output_arg: это путь к файлу (если относительный — от project_root)
      - много input + output_arg: трактуем output_arg как директорию (создаём при необходимости)
        и пишем файлы как <stem>_processed.parquet
    """
    if not input_files:
        return []

    if not output_arg:
        return [infer_output_path(p) for p in input_files]

    outp = Path(output_arg)
    if not outp.is_absolute():
        outp = project_root / outp

    if len(input_files) == 1:
        outp.parent.mkdir(parents=True, exist_ok=True)
        return [outp]

    # multi-input: output must be directory (или путь, который считаем директорией)
    # Если указали файл с суффиксом .parquet — это неоднозначно и лучше упасть.
    if outp.suffix.lower() == ".parquet":
        raise ValueError(
            f"--output={outp} выглядит как файл .parquet, но входов несколько ({len(input_files)}). "
            "Укажи директорию для --output."
        )

    out_dir = outp
    out_dir.mkdir(parents=True, exist_ok=True)

    outs: List[Path] = []
    for ip in input_files:
        outs.append(out_dir / f"{ip.stem}_processed.parquet")
    return outs


def _upload_to_minio(output_path: Path) -> None:
    import subprocess

    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket = os.getenv("MINIO_BUCKET", "media-intel")
    alias = os.getenv("MINIO_ALIAS", "local")

    if not minio_access_key or not minio_secret_key:
        raise ValueError("Не заданы MINIO_ACCESS_KEY/MINIO_SECRET_KEY. Заполни .env (см. .env.example) и повтори.")

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


def _process_one_file(input_path: Path, output_path: Path, with_actions: bool) -> None:
    print("[INFO] Преобразуем silver -> gold (summary + keywords)")
    print(f"silver: {input_path}")

    df_silver = load_silver_df(input_path)
    _validate_required_columns(
        df_silver,
        ["id", "title", "link", "source", "published_at", "clean_text"],
        context="silver",
    )

    # 1) нормализуем время
    df_silver = normalize_published_at(df_silver)

    # 1.1) гарантируем id
    df_silver = ensure_id_column(df_silver)

    # 2) dedup внутри батча ДО обогащения
    df_silver = dedup_batch_silver(df_silver)

    # 3) фильтрация низкокачественных записей
    df_silver = filter_low_quality_for_gold(df_silver)

    if df_silver.empty:
        print("[WARN] После фильтрации quality не осталось строк. Сохраняем пустой gold.")
        df_gold = df_silver.copy()

        # Стабильный контракт колонок для downstream (ClickHouse/BI)
        for col, dtype in [
            ("summary", "string"),
            ("keywords", "object"),
            ("nlp_text", "string"),
            ("entities_persons", "string"),
            ("entities_geo", "string"),
            ("entities_orgs", "string"),
            ("entities_persons_canon", "string"),
            ("entities_geo_canon", "string"),
            ("entities_orgs_canon", "string"),
            ("persons", "string"),
            ("geo", "string"),
            ("lang", "string"),
            ("keyphrases", "string"),
            ("sentiment_label", "string"),
            ("sentiment_score", "float"),
        ]:
            if col not in df_gold.columns:
                df_gold[col] = pd.Series(dtype=dtype)

        if with_actions:
            if "persons_actions" not in df_gold.columns:
                df_gold["persons_actions"] = pd.Series(dtype="string")
            if "actions_verbs" not in df_gold.columns:
                df_gold["actions_verbs"] = pd.Series(dtype="string")
    else:
        # 4) summary + keywords
        df_gold = enrich_articles_with_summary_and_keywords(df_silver)
        if df_gold is None or not isinstance(df_gold, pd.DataFrame):
            raise RuntimeError(
                "[ERROR] enrich_articles_with_summary_and_keywords() не вернула DataFrame. "
                "Проверь return в src/processing/summarization.py"
            )

        # 5) извлечение сущностей (только если колонок нет)
        df_gold = _safe_apply(
            df_gold,
            add_entities_persons_geo_from_text_if_missing,
            "add_entities_persons_geo_from_text_if_missing",
        )

        # 6) canonical columns
        df_gold = _safe_apply(df_gold, add_entities_persons_canon_column, "add_entities_persons_canon_column")
        df_gold = _safe_apply(df_gold, add_entities_geo_canon_column, "add_entities_geo_canon_column")
        df_gold = _safe_apply(df_gold, add_entities_orgs_canon_column, "add_entities_orgs_canon_column")

        # 7) совместимые “presentation” колонки persons/geo (строго из канона)
        df_gold["persons"] = df_gold.get("entities_persons_canon", "").fillna("").astype(str)
        df_gold["geo"] = df_gold.get("entities_geo_canon", df_gold.get("geo", "")).fillna("").astype(str)

        # 8) actions (опционально)
        if with_actions:
            df_gold = _safe_apply(df_gold, add_persons_actions_columns, "add_persons_actions_columns")

        # 9) lang + keyphrases + sentiment
        df_gold = _safe_apply(df_gold, add_lang_keyphrases_sentiment, "add_lang_keyphrases_sentiment")
        _print_nlp_health(df_gold)

        # гарантируем id в gold
        df_gold = ensure_id_column(df_gold)

    print(f"gold:  {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_gold.to_parquet(output_path, index=False)


def _validate_required_columns(df: pd.DataFrame, required: List[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] {context} data is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    empty_cols = [c for c in required if df[c].isna().all()]
    if empty_cols:
        raise ValueError(
            f"[ERROR] {context} data has empty required columns: {empty_cols}. "
            "Ensure the upstream step populates these fields."
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    input_files = _expand_input_to_files(args.input, project_root)
    if not input_files:
        raise FileNotFoundError(f"Не найден входной файл/директория/шаблон: {args.input}")

    output_paths = _resolve_output_paths(input_files, args.output, project_root)

    print(f"[INFO] Input silver files: {len(input_files)}")
    for p in input_files[:10]:
        print(f"  - {p}")
    if len(input_files) > 10:
        print("  ...")

    # Прогоняем все входы
    for ip, op in zip(input_files, output_paths):
        _process_one_file(ip, op, with_actions=args.with_actions)

        # upload per-file
        if args.upload_minio:
            _upload_to_minio(op)


if __name__ == "__main__":
    main()
