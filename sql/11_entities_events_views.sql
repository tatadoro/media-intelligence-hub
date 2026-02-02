-- ============================================================
-- Events + hourly views for GEO and KEYPHRASES
-- Цель: воспроизводимые и устойчивые вьюхи, завязанные на
-- articles_dedup.entities_geo(_canon) и articles_dedup.keyphrases
-- ============================================================

-- GEO: “события” (1 гео-упоминание = 1 строка)
CREATE OR REPLACE VIEW media_intel.v_geo_events AS
SELECT
    published_at,
    source,
    link,
    title,
    lowerUTF8(
        trimBoth(
            arrayJoin(
                arrayFilter(
                    x -> trimBoth(x) != '',
                    splitByChar(
                        ';',
                        replaceRegexpAll(coalesce(entities_geo, ''), '[\\|]', ';')
                    )
                )
            )
        )
    ) AS geo
FROM media_intel.articles_dedup
WHERE coalesce(entities_geo, '') != '';

-- GEO: агрегация по часам
CREATE OR REPLACE VIEW media_intel.v_geo_hourly AS
SELECT
    toStartOfHour(published_at) AS dt,
    source,
    geo,
    count() AS mentions
FROM media_intel.v_geo_events
GROUP BY
    dt,
    source,
    geo;

-- KEYPHRASES: “события” (1 ключевая фраза = 1 строка)
-- ВАЖНО: в проекте ключевые фразы обычно лежат в keyphrases (а не keywords).
CREATE OR REPLACE VIEW media_intel.v_keywords_events AS
SELECT
    published_at,
    source,
    link,
    title,
    lowerUTF8(
        trimBoth(
            arrayJoin(
                arrayFilter(
                    x -> trimBoth(x) != '',
                    splitByChar(
                        ';',
                        replaceRegexpAll(coalesce(keyphrases, ''), '[\\|]', ';')
                    )
                )
            )
        )
    ) AS keyword
FROM media_intel.articles_dedup
WHERE coalesce(keyphrases, '') != '';

-- KEYPHRASES: агрегация по часам
CREATE OR REPLACE VIEW media_intel.v_keywords_hourly AS
SELECT
    toStartOfHour(published_at) AS dt,
    source,
    keyword,
    count() AS mentions
FROM media_intel.v_keywords_events
GROUP BY
    dt,
    source,
    keyword;