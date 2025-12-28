USE media_intel;

WITH
  a AS
  (
    SELECT
      if(empty(batch_id), ingest_object_name, batch_id) AS batch_key,
      count() AS rows_loaded,
      countIf(lengthUTF8(coalesce(clean_text, '')) >= 40) AS has_body_rows,
      countIf(lengthUTF8(coalesce(clean_text, '')) < 40)  AS title_only_rows,
      min(published_at) AS min_dt,
      max(published_at) AS max_dt
    FROM articles
    GROUP BY batch_key
  ),
  w AS
  (
    SELECT
      if(empty(batch_id), ingest_object_name, batch_id) AS batch_key,
      count() AS winners
    FROM articles_dedup
    GROUP BY batch_key
  )
SELECT
  a.batch_key,
  a.rows_loaded,
  a.has_body_rows,
  a.title_only_rows,
  ifNull(w.winners, 0) AS winners,
  round(winners / a.rows_loaded, 3) AS share_winners,
  a.min_dt,
  a.max_dt
FROM a
LEFT JOIN w USING (batch_key)
ORDER BY a.rows_loaded DESC
FORMAT Pretty;