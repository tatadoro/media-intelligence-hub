USE media_intel;

SELECT
  if(empty(batch_id), ingest_object_name, batch_id) AS batch_key,
  count() AS total,
  countIf(best_clean_len >= 40) AS has_body,
  countIf(best_clean_len < 40) AS title_only,
  round(has_body / total, 3) AS share_has_body
FROM articles_dedup
GROUP BY batch_key
ORDER BY total DESC
FORMAT Pretty;