USE media_intel;

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
LIMIT 30;