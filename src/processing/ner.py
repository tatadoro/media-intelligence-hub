from __future__ import annotations

from typing import Tuple, List

from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc, MorphVocab


_segmenter = Segmenter()
_emb = NewsEmbedding()
_tagger = NewsNERTagger(_emb)
_morph = MorphVocab()


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_persons_geo(text: str) -> Tuple[str, str]:
    """
    Возвращает (persons, geo) как строки 'a;b;c' в нижнем регистре.
    persons: PER
    geo: LOC
    """
    if not text or not text.strip():
        return "", ""

    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_ner(_tagger)

    persons: List[str] = []
    geo: List[str] = []

    for span in doc.spans:
        span.normalize(_morph)
        val = (span.normal or span.text or "").strip().lower()
        if not val:
            continue

        if span.type == "PER":
            persons.append(val)
        elif span.type == "LOC":
            geo.append(val)

    persons = _uniq_keep_order(persons)
    geo = _uniq_keep_order(geo)

    return ";".join(persons), ";".join(geo)