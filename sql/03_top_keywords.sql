USE media_intel;

-- 1) TOP keywords по всем дедуп-данным
WITH
  ['стать известно','стать','известно','назвать','рассказать','объяснить','раскрыть','новый'] AS stop_kw
SELECT
  kw,
  count() AS cnt
FROM
(
  SELECT arrayJoin(
           arrayFilter(
             x -> lengthUTF8(x) > 2 AND NOT has(stop_kw, x),
             arrayMap(
               x -> lowerUTF8(trimBoth(x, ' ')),
               splitByString(';', coalesce(keywords, ''))
             )
           )
       ) AS kw
  FROM articles_dedup
)
GROUP BY kw
ORDER BY cnt DESC
LIMIT 30
FORMAT Pretty;

-- 2) TOP keywords по последнему загруженному batch (автоматически из load_log)
WITH
  (SELECT argMax(object_name, loaded_at) FROM load_log WHERE layer = 'gold') AS last_batch,
  ['стать известно','стать','известно','назвать','рассказать','объяснить','раскрыть','новый'] AS stop_kw
SELECT
  last_batch AS ingest_object_name,
  kw,
  count() AS cnt
FROM
(
  SELECT arrayJoin(
           arrayFilter(
             x -> lengthUTF8(x) > 2 AND NOT has(stop_kw, x),
             arrayMap(
               x -> lowerUTF8(trimBoth(x, ' ')),
               splitByString(';', coalesce(keywords, ''))
             )
           )
       ) AS kw
  FROM articles_dedup
  WHERE ingest_object_name = last_batch
)
GROUP BY kw
ORDER BY cnt DESC
LIMIT 30
FORMAT Pretty;