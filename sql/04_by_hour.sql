/* sql/04_by_hour.sql
   Hourly rollups for entities + sentiment
*/

/* 1) ENTITIES */

DROP TABLE IF EXISTS media_intel.bi_entities_by_hour;

CREATE TABLE media_intel.bi_entities_by_hour
(
    ts_hour   DateTime('Europe/Moscow'),
    source    LowCardinality(String),
    lang      LowCardinality(String),
    entity_type LowCardinality(String),
    entity    String,
    cnt       UInt64
)
ENGINE = MergeTree
ORDER BY (ts_hour, source, lang, entity_type, entity);

INSERT INTO media_intel.bi_entities_by_hour
WITH
    toStartOfHour(published_at) AS ts_hour
SELECT
    ts_hour,
    source,
    lang,
    entity_type,
    entity,
    count() AS cnt
FROM
(
    /* persons: ВАЖНО — берём canon */
    SELECT
        published_at,
        source,
        lang,
        'persons' AS entity_type,
        lowerUTF8(trim(e)) AS entity
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_persons_canon) AS e
    WHERE trim(e) != ''

    UNION ALL

    /* orgs */
    SELECT
        published_at,
        source,
        lang,
        'orgs' AS entity_type,
        lowerUTF8(trim(e)) AS entity
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_orgs) AS e
    WHERE trim(e) != ''

    UNION ALL

    /* geo */
    SELECT
        published_at,
        source,
        lang,
        'geo' AS entity_type,
        lowerUTF8(trim(e)) AS entity
    FROM media_intel.articles_dedup
    ARRAY JOIN splitByChar(';', entities_geo) AS e
    WHERE trim(e) != ''
) t
GROUP BY ts_hour, source, lang, entity_type, entity;


/* 2) SENTIMENT */

DROP TABLE IF EXISTS media_intel.bi_sentiment_by_hour;

CREATE TABLE media_intel.bi_sentiment_by_hour
(
    ts_hour        DateTime('Europe/Moscow'),
    source         LowCardinality(String),
    lang           LowCardinality(String),
    cnt            UInt64,
    sentiment_avg  Float32,
    sentiment_p50  Float32
)
ENGINE = MergeTree
ORDER BY (ts_hour, source, lang);

INSERT INTO media_intel.bi_sentiment_by_hour
SELECT
    toStartOfHour(published_at) AS ts_hour,
    source,
    lang,
    count() AS cnt,
    avg(toFloat32(sentiment_score)) AS sentiment_avg,
    quantileTDigest(0.5)(toFloat32(sentiment_score)) AS sentiment_p50
FROM media_intel.articles_dedup
GROUP BY ts_hour, source, lang;