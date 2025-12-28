USE media_intel;

SELECT
  if(empty(batch_id), ingest_object_name, batch_id) AS batch_key,
  count() AS rows_loaded,
  countIf(lengthUTF8(coalesce(clean_text, '')) >= 40) AS has_body,
  countIf(lengthUTF8(coalesce(clean_text, '')) < 40) AS title_only,
  round(has_body / rows_loaded, 3) AS share_has_body,
  min(published_at) AS min_dt,
  max(published_at) AS max_dt
FROM articles
GROUP BY batch_key
ORDER BY rows_loaded DESC
FORMAT Pretty;
