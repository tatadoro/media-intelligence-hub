from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipeline.clean_raw_to_silver_local import transform_raw_to_silver
from src.pipeline.silver_to_gold_local import (
    dedup_batch_silver,
    ensure_id_column,
    filter_low_quality_for_gold,
    load_silver_df,
    normalize_published_at,
)
from src.processing.nlp_extras import add_lang_keyphrases_sentiment
from src.processing.summarization import KeywordsConfig, enrich_articles_with_summary_and_keywords


def test_smoke_etl_raw_to_gold(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    silver_path = tmp_path / "silver_clean.json"
    gold_path = tmp_path / "gold.parquet"

    long_text_1 = (
        "Длинный тестовый текст про искусственный интеллект, анализ данных, медиа и технологии. "
        "Здесь есть содержательные слова для ключевых фраз и саммари. "
    ) * 10
    long_text_2 = (
        "Большой материал про экономику, финансы, рынки и инвестиции, с упоминанием компаний и отраслей. "
        "Текст содержит разные термины и темы для выделения ключевых слов. "
    ) * 10
    raw_items = [
        {
            "id": "a1",
            "title": "Test Title One",
            "link": "https://example.com/one",
            "source": "example.com",
            "published_at": "2025-12-01T10:00:00Z",
            "raw_text": long_text_1,
        },
        {
            "id": "a2",
            "title": "Test Title Two",
            "link": "https://example.com/two",
            "source": "example.com",
            "published_at": "2025-12-02T12:30:00Z",
            "raw_text": long_text_2,
        },
    ]

    raw_path.write_text(json.dumps(raw_items, ensure_ascii=False), encoding="utf-8")

    transform_raw_to_silver(raw_path, silver_path)
    assert silver_path.exists()

    df = load_silver_df(silver_path)
    df = normalize_published_at(df)
    df = ensure_id_column(df)
    df = dedup_batch_silver(df)
    df = filter_low_quality_for_gold(df)
    assert not df.empty

    df = enrich_articles_with_summary_and_keywords(
        df,
        keywords_cfg=KeywordsConfig(min_df=1, max_df=1.0),
    )
    df = add_lang_keyphrases_sentiment(df)
    df.to_parquet(gold_path, index=False)

    assert gold_path.exists()

    df = pd.read_parquet(gold_path)
    assert len(df) == len(raw_items)

    expected_cols = {
        "summary",
        "keywords",
        "lang",
        "keyphrases",
        "sentiment_label",
        "sentiment_score",
        "published_at",
    }
    assert expected_cols.issubset(set(df.columns))


def test_keyphrases_ok_on_simple_text() -> None:
    df = pd.DataFrame(
        {
            "nlp_text": [
                "Artificial intelligence drives media analytics and innovation in modern newsrooms."
            ],
            "summary": [""],
            "keywords": [""],
        }
    )
    out = add_lang_keyphrases_sentiment(df)
    assert int(out.loc[0, "keyphrases_ok"]) == 1
    assert int(out.loc[0, "keyphrases_n"]) >= 1
    assert isinstance(out.loc[0, "keyphrases"], str)
