USE media_intel;

CREATE OR REPLACE VIEW articles_dedup AS
SELECT
  dedup_key,

  argMax(id, score)                 AS id,
  argMax(title, score)              AS title,
  argMax(link, score)               AS link,
  argMax(published_at, score)       AS published_at,
  argMax(source, score)             AS source,

  argMax(raw_text, score)           AS raw_text,
  argMax(clean_text, score)         AS clean_text,
  argMax(nlp_text, score)           AS nlp_text,
  argMax(summary, score)            AS summary,
  argMax(keywords, score)           AS keywords,

  argMax(text_length_chars, score)  AS text_length_chars,
  argMax(num_sentences, score)      AS num_sentences,
  argMax(num_keywords, score)       AS num_keywords,

  argMax(ingest_object_name, score) AS ingest_object_name,
  argMax(clean_len, score)          AS best_clean_len
FROM
(
  SELECT
    a.*,
    if(empty(a.link), a.id, a.link) AS dedup_key,
    lengthUTF8(coalesce(a.clean_text, '')) AS clean_len,
    toUInt8(clean_len >= 40) AS has_body,

    coalesce(
      ll.loaded_at,
      toDateTime64('1970-01-01 00:00:00', 9, 'Europe/Moscow')
    ) AS loaded_at,

    toUInt8(positionCaseInsensitive(a.ingest_object_name, '_v2_') > 0) AS batch_priority,

    (has_body, clean_len, batch_priority, a.published_at, loaded_at) AS score
  FROM articles AS a
  LEFT JOIN
  (
    SELECT
      object_name,
      max(loaded_at) AS loaded_at
    FROM media_intel.load_log
    WHERE layer = 'gold'
    GROUP BY object_name
  ) AS ll
    ON a.ingest_object_name = ll.object_name
) AS t
GROUP BY dedup_key;