-- 09_actions_views.sql
-- Views for actions extraction: persons_actions + actions_verbs
-- Requires columns in articles(_dedup): persons_actions, actions_verbs

CREATE DATABASE IF NOT EXISTS media_intel;

USE media_intel;

-- 1) Flatten: person -> verb (one row per (article, person, verb))
CREATE OR REPLACE VIEW v_persons_actions_flat AS
WITH
    lowerUTF8(trim(arrayElement(splitByChar(':', pair_raw), 1))) AS person,
    lowerUTF8(trim(verb_raw)) AS verb
SELECT
    published_at,
    source,
    id,
    title,
    link,
    ingest_object_name,
    batch_id,
    person,
    verb
FROM articles_dedup
ARRAY JOIN splitByChar('|', persons_actions) AS pair_raw
ARRAY JOIN splitByChar(',', arrayElement(splitByChar(':', pair_raw), 2)) AS verb_raw
WHERE
    persons_actions != ''
    AND person != ''
    AND verb != '';

-- 2) Flatten: just verbs (one row per (article, verb))
CREATE OR REPLACE VIEW v_actions_verbs_flat AS
WITH lowerUTF8(trim(verb_raw)) AS verb
SELECT
    published_at,
    source,
    id,
    title,
    link,
    ingest_object_name,
    batch_id,
    verb
FROM articles_dedup
ARRAY JOIN splitByChar(';', actions_verbs) AS verb_raw
WHERE
    actions_verbs != ''
    AND verb != '';

-- 3) Hourly: persons (any verb)
CREATE OR REPLACE VIEW v_persons_actions_hourly AS
SELECT
    toStartOfHour(published_at) AS ts_hour,
    person,
    count() AS actions_rows,
    uniqExact(id) AS articles
FROM v_persons_actions_flat
GROUP BY
    ts_hour,
    person;

-- 4) Hourly: verbs
CREATE OR REPLACE VIEW v_actions_verbs_hourly AS
SELECT
    toStartOfHour(published_at) AS ts_hour,
    verb,
    count() AS verb_rows,
    uniqExact(id) AS articles
FROM v_actions_verbs_flat
GROUP BY
    ts_hour,
    verb;

-- 5) Daily/events: persons (any verb)
CREATE OR REPLACE VIEW v_persons_actions_events AS
SELECT
    toDate(published_at) AS dt,
    person,
    count() AS actions_rows,
    uniqExact(id) AS articles
FROM v_persons_actions_flat
GROUP BY
    dt,
    person;

-- 6) Daily/events: verbs
CREATE OR REPLACE VIEW v_actions_verbs_events AS
SELECT
    toDate(published_at) AS dt,
    verb,
    count() AS verb_rows,
    uniqExact(id) AS articles
FROM v_actions_verbs_flat
GROUP BY
    dt,
    verb;