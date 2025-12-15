USE media_intel;

WITH
  ['стать известно','назвать','рассказать','объяснить','раскрыть','новый'] AS stop_kw
SELECT
  toStartOfHour(published_at) AS hour,
  kw,
  count() AS cnt
FROM
(
  SELECT
    published_at,
    arrayJoin(
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
GROUP BY hour, kw
ORDER BY hour DESC, cnt DESC
LIMIT 50;