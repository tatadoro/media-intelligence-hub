USE media_intel;

-- 1) Сводка качества: сколько материалов с body (clean_text >= min_body_len)
WITH 40 AS min_body_len
SELECT
  count() AS total,
  countIf(lengthUTF8(coalesce(clean_text, '')) >= min_body_len) AS has_body,
  countIf(lengthUTF8(coalesce(clean_text, '')) <  min_body_len) AS title_only,
  if(total = 0, 0, round(has_body / total, 3)) AS share_has_body
FROM articles_dedup;

-- 2) Примеры "title_only"
WITH 40 AS min_body_len
SELECT
  published_at,
  title,
  source,
  lengthUTF8(coalesce(clean_text, '')) AS clean_len,
  summary,
  keywords
FROM articles_dedup
WHERE lengthUTF8(coalesce(clean_text, '')) < min_body_len
ORDER BY published_at DESC
LIMIT 10;