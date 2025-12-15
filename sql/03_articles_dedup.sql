USE media_intel;

DROP VIEW IF EXISTS articles_dedup;

CREATE VIEW articles_dedup AS
WITH
  if(empty(link), id, link) AS dedup_key,
  lengthUTF8(coalesce(clean_text, '')) AS clean_len,
  (clean_len, published_at) AS score
SELECT
  dedup_key,

  argMax(id, score) AS id,
  argMax(title, score) AS title,
  argMax(link, score) AS link,
  max(published_at) AS published_at,
  argMax(source, score) AS source,

  argMax(raw_text, score) AS raw_text,
  argMax(clean_text, score) AS clean_text,
  argMax(nlp_text, score) AS nlp_text,
  argMax(summary, score) AS summary,
  argMax(keywords, score) AS keywords,

  max(text_length_chars) AS text_length_chars,
  max(num_sentences) AS num_sentences,
  max(num_keywords) AS num_keywords,

  -- NEW: какой батч дал “лучшую” строку для этого dedup_key
  argMax(ingest_object_name, score) AS ingest_object_name,

  max(clean_len) AS best_clean_len
FROM articles
GROUP BY dedup_key;