-- ============================================================
-- MIH: сводный отчёт (лёгкий, безопасный по памяти)
-- Формат вывода управляется CH_FORMAT через scripts/ch_run_sql.sh
-- ============================================================
SELECT 'ch_now' AS kind, toString(now()) AS value;
SELECT 'ch_timezone' AS kind, timezone() AS value;

-- 1) Сколько строк в основных таблицах
SELECT 'articles_total' AS kind, count() AS value
FROM media_intel.articles;

SELECT 'articles_dedup_total' AS kind, count() AS value
FROM media_intel.articles_dedup;

-- 2) Топ источников в dedup (ограничиваем, чтобы не раздувать вывод)
SELECT
  'top_sources_dedup' AS kind,
  source,
  count() AS cnt
FROM media_intel.articles_dedup
GROUP BY source
ORDER BY cnt DESC
LIMIT 30;

-- 3) Распределение языков (если колонки нет — этот запрос упадёт; у тебя она есть)
SELECT
  'lang_dedup' AS kind,
  lang,
  count() AS cnt
FROM media_intel.articles_dedup
GROUP BY lang
ORDER BY cnt DESC
LIMIT 30;

-- 4) Интроспекция: какие "status"-поля вообще есть в RAW таблице
-- (чтобы дальше можно было выбрать корректное поле вместо body_status)
SELECT
  'raw_status_columns' AS kind,
  name AS column_name,
  type AS column_type
FROM system.columns
WHERE database = 'media_intel'
  AND table = 'articles'
  AND positionCaseInsensitive(name, 'status') > 0
ORDER BY name;

-- 5) Сколько материалов за последние 24 часа (по published_at)
SELECT 'dedup_last_24h' AS kind, count() AS value
FROM media_intel.articles_dedup
WHERE published_at >= now() - INTERVAL 24 HOUR;

SELECT 'dedup_last_7d' AS kind, count() AS value
FROM media_intel.articles_dedup
WHERE published_at >= now() - INTERVAL 7 DAY;

SELECT 'dedup_last_30d' AS kind, count() AS value
FROM media_intel.articles_dedup
WHERE published_at >= now() - INTERVAL 30 DAY;

-- 6) Последний published_at в dedup (контроль «живости» загрузки)
SELECT
  'dedup_max_published_at' AS kind,
  toString(max(published_at)) AS value
FROM media_intel.articles_dedup;