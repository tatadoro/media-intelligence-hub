USE media_intel;

-- 07_batch_internal_dupes.sql
-- Дубли внутри батча по dedup_key = if(empty(link), id, link)

-- A) Сводка по каждому батчу
SELECT
  ingest_object_name,
  sum(cnt) AS rows,
  count() AS uniq_keys,
  (sum(cnt) - count()) AS duplicate_rows,
  countIf(cnt > 1) AS duplicate_keys
FROM
(
  SELECT
    ingest_object_name,
    if(empty(link), id, link) AS dedup_key,
    count() AS cnt
  FROM articles
  GROUP BY
    ingest_object_name,
    if(empty(link), id, link)
)
GROUP BY ingest_object_name
ORDER BY duplicate_rows DESC, rows DESC
FORMAT Pretty;

-- B) ТОП дублей (какие ключи задублились и сколько раз)
SELECT
  ingest_object_name,
  if(empty(link), id, link) AS dedup_key,
  count() AS c
FROM articles
GROUP BY
  ingest_object_name,
  if(empty(link), id, link)
HAVING c > 1
ORDER BY c DESC, ingest_object_name
LIMIT 50
FORMAT Pretty;
