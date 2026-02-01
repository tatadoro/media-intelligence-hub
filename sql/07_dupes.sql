/* sql/07_dupes.sql
   Проверки дублей:
   1) RAW слой (media_intel.articles): дубли ожидаемы, но мониторим масштабы
   2) DEDUP слой (media_intel.articles_dedup): дублей по id/dedup_key быть не должно

   ВАЖНО:
   - clickhouse-client НЕ печатает ничего для SELECT, который вернул 0 строк.
     Поэтому в конце добавлены счётчики, которые всегда печатаются (1 строка).
   - Для согласованности используем dedup_key = coalesce(nullIf(link,''), nullIf(id,''), '').

   ПРО ВЫВОД:
   - В текстовых полях могут быть \t/\r/\n. В TSV-выводе это ломает колонки.
     Поэтому ниже печатаемые текстовые поля "санитизируются":
     replaceRegexpAll(x, '[\\t\\r\\n]', ' ')
*/

SET max_bytes_before_external_group_by = 100000000;
SET max_memory_usage = 3000000000;

-- =========================================================
-- 1) RAW (media_intel.articles)
-- =========================================================

-- 1.1 Дубли по id (в raw слое допустимо, но мониторим)
SELECT
  'raw_dupe_id' AS kind,
  replaceRegexpAll(id, '[\\t\\r\\n]', ' ') AS id,
  count() AS cnt,
  replaceRegexpAll(any(source), '[\\t\\r\\n]', ' ') AS source,
  replaceRegexpAll(any(title),  '[\\t\\r\\n]', ' ') AS title,
  max(published_at) AS max_published_at
FROM media_intel.articles
WHERE id != ''
GROUP BY id
HAVING cnt > 1
ORDER BY cnt DESC, max_published_at DESC
LIMIT 50;

-- 1.2 Дубли по (source, link) — полезно для RSS (raw слой)
SELECT
  'raw_dupe_source_link' AS kind,
  replaceRegexpAll(source, '[\\t\\r\\n]', ' ') AS source,
  replaceRegexpAll(link,   '[\\t\\r\\n]', ' ') AS link,
  count() AS cnt,
  replaceRegexpAll(any(title), '[\\t\\r\\n]', ' ') AS title,
  max(published_at) AS max_published_at
FROM media_intel.articles
WHERE link != ''
GROUP BY source, link
HAVING cnt > 1
ORDER BY cnt DESC, max_published_at DESC
LIMIT 50;

-- 1.3 Дубли по (source, title, published_at) — если link пустой (raw слой)
SELECT
  'raw_dupe_source_title_time' AS kind,
  replaceRegexpAll(source, '[\\t\\r\\n]', ' ') AS source,
  replaceRegexpAll(title,  '[\\t\\r\\n]', ' ') AS title,
  published_at,
  count() AS cnt
FROM media_intel.articles
GROUP BY source, title, published_at
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 50;

-- 1.4 RAW: дубли по dedup_key (link если есть, иначе id)
WITH
  coalesce(nullIf(link, ''), nullIf(id, ''), '') AS k
SELECT
  'raw_dupe_dedup_key' AS kind,
  replaceRegexpAll(k, '[\\t\\r\\n]', ' ') AS dedup_key,
  count() AS cnt,
  replaceRegexpAll(any(source), '[\\t\\r\\n]', ' ') AS source,
  replaceRegexpAll(any(title),  '[\\t\\r\\n]', ' ') AS title,
  max(published_at) AS max_published_at
FROM media_intel.articles
WHERE k != ''
GROUP BY k
HAVING cnt > 1
ORDER BY cnt DESC, max_published_at DESC
LIMIT 50;

-- =========================================================
-- 2) DEDUP (media_intel.articles_dedup)
-- =========================================================

-- 2.1 В dedup слое дубли по id быть НЕ должны
SELECT
  'dedup_dupe_id' AS kind,
  replaceRegexpAll(id, '[\\t\\r\\n]', ' ') AS id,
  count() AS cnt,
  replaceRegexpAll(any(source), '[\\t\\r\\n]', ' ') AS source,
  replaceRegexpAll(any(title),  '[\\t\\r\\n]', ' ') AS title,
  max(published_at) AS max_published_at
FROM media_intel.articles_dedup
WHERE id != ''
GROUP BY id
HAVING cnt > 1
ORDER BY cnt DESC, max_published_at DESC
LIMIT 50;

-- 2.2 В dedup слое дубли по (source, link) тоже не ожидаем
SELECT
  'dedup_dupe_source_link' AS kind,
  replaceRegexpAll(source, '[\\t\\r\\n]', ' ') AS source,
  replaceRegexpAll(link,   '[\\t\\r\\n]', ' ') AS link,
  count() AS cnt,
  replaceRegexpAll(any(title), '[\\t\\r\\n]', ' ') AS title,
  max(published_at) AS max_published_at
FROM media_intel.articles_dedup
WHERE link != ''
GROUP BY source, link
HAVING cnt > 1
ORDER BY cnt DESC, max_published_at DESC
LIMIT 50;

-- 2.3 В dedup слое дубли по dedup_key (как в view) тоже не ожидаем
SELECT
  'dedup_dupe_dedup_key' AS kind,
  replaceRegexpAll(dedup_key, '[\\t\\r\\n]', ' ') AS dedup_key,
  count() AS cnt,
  replaceRegexpAll(any(source), '[\\t\\r\\n]', ' ') AS source,
  replaceRegexpAll(any(title),  '[\\t\\r\\n]', ' ') AS title,
  max(published_at) AS max_published_at
FROM media_intel.articles_dedup
WHERE dedup_key != ''
GROUP BY dedup_key
HAVING cnt > 1
ORDER BY cnt DESC, max_published_at DESC
LIMIT 50;

-- =========================================================
-- 3) СЧЁТЧИКИ (всегда печатаются одной строкой)
-- =========================================================

-- 3.1 Сколько id имеют >1 строки в RAW
SELECT
  'raw_dupe_id_total' AS kind,
  count() AS value
FROM
(
  SELECT id
  FROM media_intel.articles
  WHERE id != ''
  GROUP BY id
  HAVING count() > 1
);

-- 3.2 Сколько (source, link) имеют >1 строки в RAW
SELECT
  'raw_dupe_source_link_total' AS kind,
  count() AS value
FROM
(
  SELECT source, link
  FROM media_intel.articles
  WHERE link != ''
  GROUP BY source, link
  HAVING count() > 1
);

-- 3.3 Сколько (source, title, published_at) имеют >1 строки в RAW
SELECT
  'raw_dupe_source_title_time_total' AS kind,
  count() AS value
FROM
(
  SELECT source, title, published_at
  FROM media_intel.articles
  GROUP BY source, title, published_at
  HAVING count() > 1
);

-- 3.4 Сколько dedup_key имеют >1 строки в RAW
SELECT
  'raw_dupe_dedup_key_total' AS kind,
  count() AS value
FROM
(
  SELECT coalesce(nullIf(link, ''), nullIf(id, ''), '') AS k
  FROM media_intel.articles
  WHERE coalesce(nullIf(link, ''), nullIf(id, ''), '') != ''
  GROUP BY k
  HAVING count() > 1
);

-- Memory-safe настройки: разрешаем ClickHouse “проливать” GROUP BY на диск,
-- чтобы отчёт не падал по памяти на больших таблицах.
SET max_bytes_before_external_group_by = 200000000;  -- ~200MB
SET max_bytes_before_external_sort     = 200000000;  -- на случай ORDER BY/uniq в файле

-- 3.5 Сколько id имеют >1 строки в DEDUP
SELECT
  'dedup_dupe_id_total' AS kind,
  count() AS value
FROM
(
  SELECT id
  FROM media_intel.articles_dedup
  WHERE id != ''
  GROUP BY id
  HAVING count() > 1
);

-- 3.6 Сколько (source, link) имеют >1 строки в DEDUP
SELECT
  'dedup_dupe_source_link_total' AS kind,
  count() AS value
FROM
(
  SELECT source, link
  FROM media_intel.articles_dedup
  WHERE link != ''
  GROUP BY source, link
  HAVING count() > 1
);

-- 3.7 Сколько dedup_key имеют >1 строки в DEDUP
SELECT
  'dedup_dupe_dedup_key_total' AS kind,
  count() AS value
FROM
(
  SELECT dedup_key
  FROM media_intel.articles_dedup
  WHERE dedup_key != ''
  GROUP BY dedup_key
  HAVING count() > 1
);