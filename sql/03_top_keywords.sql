USE media_intel;

-- 1) TOP keywords по всем дедуп-данным (без дайджестов)
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
  WHERE is_digest = 0
)
GROUP BY kw
ORDER BY cnt DESC
LIMIT 30
FORMAT Pretty;

-- 2) TOP keywords по последнему batch_id
WITH
  (SELECT max(batch_id) FROM articles WHERE batch_id != '') AS last_batch_id,
  ['стать известно','стать','известно','назвать','рассказать','объяснить','раскрыть','новый'] AS stop_kw
SELECT
  last_batch_id AS batch_id,
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
  WHERE batch_id = last_batch_id
    AND is_digest = 0
)
GROUP BY kw
ORDER BY cnt DESC
LIMIT 30
FORMAT Pretty;