USE media_intel;

WITH
  40 AS min_body_len
SELECT
  count() AS total,
  countIf(lengthUTF8(coalesce(clean_text, '')) >= min_body_len) AS has_body,
  countIf(lengthUTF8(coalesce(clean_text, '')) <  min_body_len) AS title_only,
  round(has_body / total, 3) AS share_has_body
FROM articles_dedup;

-- Примеры "title_only"
SELECT
  published_at, title, source,
  lengthUTF8(coalesce(clean_text, '')) AS clean_len,
  summary, keywords
FROM articles_dedup
WHERE lengthUTF8(coalesce(clean_text, '')) < min_body_len
ORDER BY published_at DESC
LIMIT 10;