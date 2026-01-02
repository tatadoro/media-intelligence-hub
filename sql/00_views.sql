USE media_intel;

CREATE OR REPLACE VIEW articles_dedup AS
SELECT
  dedup_key,

  argMax(id, score)                 AS id,
  argMax(title, score)              AS title,
  argMax(link, score)               AS link,
  argMax(published_at, score)       AS published_at,
  argMax(source, score)             AS source,

  argMax(raw_text, score)           AS raw_text,
  argMax(clean_text, score)         AS clean_text,
  argMax(nlp_text, score)           AS nlp_text,
  argMax(summary, score)            AS summary,
  argMax(keywords, score)           AS keywords,

  argMax(text_length_chars, score)  AS text_length_chars,
  argMax(num_sentences, score)      AS num_sentences,
  argMax(num_keywords, score)       AS num_keywords,

  argMax(batch_id, score)           AS batch_id,
  argMax(is_digest, score)          AS is_digest,

  argMax(entities_persons, score)   AS entities_persons,
  argMax(entities_orgs, score)      AS entities_orgs,
  argMax(entities_geo, score)       AS entities_geo,
  argMax(num_persons, score)        AS num_persons,
  argMax(num_orgs, score)           AS num_orgs,
  argMax(num_geo, score)            AS num_geo,

  argMax(ingest_object_name, score) AS ingest_object_name,
  argMax(clean_len, score)          AS best_clean_len
FROM
(
  WITH
    (SELECT max(loaded_at) FROM media_intel.load_log WHERE layer = 'gold') AS max_loaded
  SELECT
    a.*,
    if(empty(a.link), a.id, a.link) AS dedup_key,
    lengthUTF8(coalesce(a.clean_text, '')) AS clean_len,
    toUInt8(clean_len >= 40) AS has_body,

    coalesce(
      ll.loaded_at,
      toDateTime64('1970-01-01 00:00:00', 9, 'Europe/Moscow')
    ) AS loaded_at,

    toUInt8(coalesce(ll.loaded_at, toDateTime('1970-01-01 00:00:00')) = max_loaded) AS batch_priority,

    (has_body, clean_len, batch_priority, a.published_at, loaded_at) AS score
  FROM articles AS a
  LEFT JOIN
  (
    SELECT
      object_name,
      max(loaded_at) AS loaded_at
    FROM media_intel.load_log
    WHERE layer = 'gold'
    GROUP BY object_name
  ) AS ll
    ON a.ingest_object_name = ll.object_name
) AS t
GROUP BY dedup_key;

-- ------------------------------------------------------------------
-- Latest ingestion status per source (for Superset lag/status widgets)
-- ------------------------------------------------------------------

CREATE OR REPLACE VIEW ingestion_coverage_latest AS
SELECT
  source_type,
  source,
  batch_id,
  run_started_at,
  run_finished_at,
  duration_ms,
  items_found,
  items_saved,
  min_published_at,
  max_published_at,
  status,
  error_message,
  raw_object_name,

  -- флаг "источник отдаёт слишком старые публикации"
  if(
    isNull(max_published_at),
    1,
    max_published_at < now('Europe/Moscow') - INTERVAL 14 DAY
  ) AS is_stale,

  -- lag считаем только для НЕ-stale
  if(
    isNull(max_published_at) OR max_published_at < now('Europe/Moscow') - INTERVAL 14 DAY,
    NULL,
    dateDiff('minute', max_published_at, now('Europe/Moscow'))
  ) AS lag_minutes
FROM
(
  SELECT
    ic.source_type AS source_type,
    ic.source      AS source,

    argMax(ic.batch_id, ic.run_finished_at)        AS batch_id,
    argMax(ic.run_started_at, ic.run_finished_at)  AS run_started_at,
    max(ic.run_finished_at)                        AS run_finished_at,
    argMax(ic.duration_ms, ic.run_finished_at)     AS duration_ms,

    argMax(ic.items_found, ic.run_finished_at)     AS items_found,
    argMax(ic.items_saved, ic.run_finished_at)     AS items_saved,

    argMax(ic.min_published_at, ic.run_finished_at) AS min_published_at,
    argMax(ic.max_published_at, ic.run_finished_at) AS max_published_at,

    argMax(ic.status, ic.run_finished_at)          AS status,
    argMax(ic.error_message, ic.run_finished_at)   AS error_message,
    argMax(ic.raw_object_name, ic.run_finished_at) AS raw_object_name
  FROM media_intel.ingestion_coverage AS ic
  GROUP BY ic.source_type, ic.source
) AS t;