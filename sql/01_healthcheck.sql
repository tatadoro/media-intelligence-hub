USE media_intel;

SELECT
  count() AS rows,
  min(published_at) AS min_dt,
  max(published_at) AS max_dt
FROM articles;

SELECT
  source,
  count() AS cnt
FROM articles
GROUP BY source
ORDER BY cnt DESC;