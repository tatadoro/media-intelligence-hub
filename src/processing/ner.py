from __future__ import annotations

from typing import Tuple, List

try:
    from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc, MorphVocab

    _NATASHA_OK = True
except Exception:
    Segmenter = NewsEmbedding = NewsNERTagger = Doc = MorphVocab = None  # type: ignore
    _NATASHA_OK = False

if _NATASHA_OK:
    _segmenter = Segmenter()
    _emb = NewsEmbedding()
    _tagger = NewsNERTagger(_emb)
    _morph = MorphVocab()
else:
    _segmenter = _emb = _tagger = _morph = None


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_persons_orgs_geo(text: str) -> Tuple[str, str, str]:
    """
    Возвращает (persons, orgs, geo) как строки 'a;b;c' в нижнем регистре.
    persons: PER
    orgs: ORG
    geo: LOC
    """
    if not text or not text.strip() or not _NATASHA_OK:
        return "", "", ""

    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_ner(_tagger)

    persons: List[str] = []
    orgs: List[str] = []
    geo: List[str] = []

    for span in doc.spans:
        span.normalize(_morph)
        val = (span.normal or span.text or "").strip().lower()
        if not val:
            continue

        if span.type == "PER":
            persons.append(val)
        elif span.type == "ORG":
            orgs.append(val)
        elif span.type == "LOC":
            geo.append(val)

    persons = _uniq_keep_order(persons)
    orgs = _uniq_keep_order(orgs)
    geo = _uniq_keep_order(geo)

    return ";".join(persons), ";".join(orgs), ";".join(geo)


def extract_persons_geo(text: str) -> Tuple[str, str]:
    """
    Возвращает (persons, geo) как строки 'a;b;c' в нижнем регистре.
    persons: PER
    geo: LOC
    """
    persons, _orgs, geo = extract_persons_orgs_geo(text)
    return persons, geo
