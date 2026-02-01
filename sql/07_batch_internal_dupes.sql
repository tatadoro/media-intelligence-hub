USE media_intel;

-- 07_batch_internal_dupes.sql
-- Дубли внутри батча по dedup_key = coalesce(nullIf(link, ''), nullIf(id, ''), '')

-- A) Сводка по каждому батчу
SELECT
  batch_key,
  sum(cnt) AS rows,
  count() AS uniq_keys,
  (sum(cnt) - count()) AS duplicate_rows,
  countIf(cnt > 1) AS duplicate_keys
FROM
(
  SELECT
    if(empty(batch_id), ingest_object_name, batch_id) AS batch_key,
    coalesce(nullIf(link, ''), nullIf(id, ''), '') AS dedup_key,
    count() AS cnt
  FROM articles
  GROUP BY
    batch_key,
    coalesce(nullIf(link, ''), nullIf(id, ''), '')
)
GROUP BY batch_key
ORDER BY duplicate_rows DESC, rows DESC
FORMAT Pretty;

-- B) ТОП дублей (какие ключи задублились и сколько раз)
SELECT
  if(empty(batch_id), ingest_object_name, batch_id) AS batch_key,
  coalesce(nullIf(link, ''), nullIf(id, ''), '') AS dedup_key,
  count() AS c
FROM articles
GROUP BY
  batch_key,
  coalesce(nullIf(link, ''), nullIf(id, ''), '')
HAVING c > 1
ORDER BY c DESC, batch_key
LIMIT 50
FORMAT Pretty;
