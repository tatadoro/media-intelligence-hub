-- ============================================
-- BI materialized aggregates for Superset
-- Source: media_intel.articles_dedup
-- ============================================

CREATE DATABASE IF NOT EXISTS media_intel;

-- 1) Articles by hour
CREATE TABLE IF NOT EXISTS media_intel.bi_articles_by_hour
(
    ts_hour DateTime('Europe/Moscow'),
    source  LowCardinality(String),
    lang    LowCardinality(String),
    cnt     UInt64
)
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(ts_hour)
ORDER BY (ts_hour, source, lang);

-- 2) Keywords by hour
CREATE TABLE IF NOT EXISTS media_intel.bi_keywords_by_hour
(
    ts_hour DateTime('Europe/Moscow'),
    source  LowCardinality(String),
    lang    LowCardinality(String),
    keyword LowCardinality(String),
    cnt     UInt64
)
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(ts_hour)
ORDER BY (ts_hour, source, lang, keyword);

-- 3) Entities by hour
CREATE TABLE IF NOT EXISTS media_intel.bi_entities_by_hour
(
    ts_hour      DateTime('Europe/Moscow'),
    source       LowCardinality(String),
    lang         LowCardinality(String),
    entity_type  LowCardinality(String),
    entity       LowCardinality(String),
    cnt          UInt64
)
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(ts_hour)
ORDER BY (ts_hour, source, lang, entity_type, entity);

-- 4) Sentiment states by hour (mergeable)
CREATE TABLE IF NOT EXISTS media_intel.bi_sentiment_by_hour_state
(
    ts_hour    DateTime('Europe/Moscow'),
    source     LowCardinality(String),
    lang       LowCardinality(String),

    cnt_state  AggregateFunction(count, UInt64),
    avg_state  AggregateFunction(avg, Float32),
    p50_state  AggregateFunction(quantileTDigest(0.5), Float32)
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(ts_hour)
ORDER BY (ts_hour, source, lang);

-- ============================================
-- ONE-TIME BACKFILL (history)
-- ============================================

INSERT INTO media_intel.bi_articles_by_hour
SELECT
    toStartOfHour(published_at) AS ts_hour,
    source,
    lang,
    count() AS cnt
FROM media_intel.articles_dedup
GROUP BY ts_hour, source, lang;

INSERT INTO media_intel.bi_keywords_by_hour
SELECT
    toStartOfHour(published_at) AS ts_hour,
    source,
    lang,
    lowerUTF8(trim(k)) AS keyword,
    count() AS cnt
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar(';', keywords) AS k
WHERE trim(k) != ''
GROUP BY ts_hour, source, lang, keyword;

INSERT INTO media_intel.bi_entities_by_hour
WITH toStartOfHour(published_at) AS ts_hour
SELECT
    ts_hour,
    source,
    lang,
    entity_type,
    lowerUTF8(trim(e)) AS entity,
    count() AS cnt
FROM
(
    SELECT published_at, source, lang, 'persons' AS entity_type, e
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_persons) AS e
    WHERE trim(e) != ''

    UNION ALL
    SELECT published_at, source, lang, 'orgs' AS entity_type, e
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_orgs) AS e
    WHERE trim(e) != ''

    UNION ALL
    SELECT published_at, source, lang, 'geo' AS entity_type, e
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_geo) AS e
    WHERE trim(e) != ''
)
GROUP BY ts_hour, source, lang, entity_type, entity;

INSERT INTO media_intel.bi_sentiment_by_hour_state
SELECT
    toStartOfHour(published_at) AS ts_hour,
    source,
    lang,
    countState(toUInt64(1)) AS cnt_state,
    avgState(sentiment_score) AS avg_state,
    quantileTDigestState(0.5)(sentiment_score) AS p50_state
FROM media_intel.articles_dedup
GROUP BY ts_hour, source, lang;

-- ============================================
-- MATERIALIZED VIEWS (incremental updates)
-- ============================================

CREATE MATERIALIZED VIEW IF NOT EXISTS media_intel.mv_bi_articles_by_hour
TO media_intel.bi_articles_by_hour
AS
SELECT
    toStartOfHour(published_at) AS ts_hour,
    source,
    lang,
    count() AS cnt
FROM media_intel.articles_dedup
GROUP BY ts_hour, source, lang;

CREATE MATERIALIZED VIEW IF NOT EXISTS media_intel.mv_bi_keywords_by_hour
TO media_intel.bi_keywords_by_hour
AS
SELECT
    toStartOfHour(published_at) AS ts_hour,
    source,
    lang,
    lowerUTF8(trim(k)) AS keyword,
    count() AS cnt
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar(';', keywords) AS k
WHERE trim(k) != ''
GROUP BY ts_hour, source, lang, keyword;

CREATE MATERIALIZED VIEW IF NOT EXISTS media_intel.mv_bi_entities_by_hour
TO media_intel.bi_entities_by_hour
AS
WITH
    toStartOfHour(published_at) AS ts_hour
SELECT
    ts_hour,
    source,
    lang,
    entity_type,
    lowerUTF8(trim(e)) AS entity,
    count() AS cnt
FROM
(
    SELECT published_at, source, lang, 'persons' AS entity_type, e
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_persons) AS e
    WHERE trim(e) != ''

    UNION ALL
    SELECT published_at, source, lang, 'orgs' AS entity_type, e
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_orgs) AS e
    WHERE trim(e) != ''

    UNION ALL
    SELECT published_at, source, lang, 'geo' AS entity_type, e
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_geo) AS e
    WHERE trim(e) != ''
)
GROUP BY ts_hour, source, lang, entity_type, entity;

CREATE MATERIALIZED VIEW IF NOT EXISTS media_intel.mv_bi_sentiment_by_hour_state
TO media_intel.bi_sentiment_by_hour_state
AS
SELECT
    toStartOfHour(published_at) AS ts_hour,
    source,
    lang,
    countState(toUInt64(1)) AS cnt_state,
    avgState(sentiment_score) AS avg_state,
    quantileTDigestState(0.5)(sentiment_score) AS p50_state
FROM media_intel.articles_dedup
GROUP BY ts_hour, source, lang;
