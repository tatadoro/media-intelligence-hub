USE media_intel;

SELECT
  ingest_object_name,
  count() AS total,
  countIf(best_clean_len >= 40) AS has_body,
  countIf(best_clean_len < 40) AS title_only,
  round(has_body / total, 3) AS share_has_body
FROM articles_dedup
GROUP BY ingest_object_name
ORDER BY total DESC
FORMAT Pretty;