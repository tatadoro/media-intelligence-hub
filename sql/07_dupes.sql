/* sql/07_dupes.sql
   Поиск дублей в raw-таблице articles.
   Важно: включаем лимиты/спилл для GROUP BY, чтобы запросы не падали по памяти.
*/

USE media_intel;

SET max_bytes_before_external_group_by = 100000000;
SET max_memory_usage = 3000000000;

-- Дубли по id (самый важный ключ)
SELECT
  'dupe_id' AS kind,
  id,
  count() AS cnt,
  any(source) AS source,
  any(title)  AS title,
  max(published_at) AS max_published_at
FROM media_intel.articles
WHERE id != ''
GROUP BY id
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 50;

-- Дубли по (source, link) — полезно для RSS
-- Важно: НЕ берём any(title), чтобы не раздувать память на большом числе групп.
SELECT
  'dupe_source_link' AS kind,
  source,
  link,
  count() AS cnt,
  max(published_at) AS max_published_at
FROM media_intel.articles
WHERE link != ''
GROUP BY source, link
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 50;

-- Дубли по (source, title, published_at) — если link пустой
SELECT
  'dupe_source_title_time' AS kind,
  source,
  title,
  published_at,
  count() AS cnt
FROM media_intel.articles
GROUP BY source, title, published_at
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 50;