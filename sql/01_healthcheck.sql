USE media_intel;

-- 1) Общие объёмы и диапазон дат
SELECT
  count() AS rows,
  min(published_at) AS min_dt,
  max(published_at) AS max_dt,
  countIf(published_at < toDateTime64('2000-01-01 00:00:00', 9, 'Europe/Moscow')) AS bad_dates
FROM articles;

-- 2) Разрез по источникам
SELECT
  source,
  count() AS cnt
FROM articles
GROUP BY source
ORDER BY cnt DESC;

-- 3) Контроль batch-id (должен быть заполнен)
SELECT
  count() AS total,
  countIf(ingest_object_name = '') AS empty_ingest,
  if(total = 0, 0, round(empty_ingest / total, 3)) AS share_empty
FROM articles;