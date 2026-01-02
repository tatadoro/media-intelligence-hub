DROP VIEW IF EXISTS media_intel.articles_dedup;

CREATE VIEW media_intel.articles_dedup
(
    `dedup_key` String,
    `id` String,
    `title` String,
    `link` String,
    `published_at` DateTime64(9, 'Europe/Moscow'),
    `source` String,
    `raw_text` String,
    `clean_text` String,
    `nlp_text` String,
    `summary` String,
    `keywords` String,

    -- NEW NLP EXTRAS
    `lang` LowCardinality(String),
    `keyphrases` String,
    `sentiment_label` LowCardinality(String),
    `sentiment_score` Float32,

    `text_length_chars` Int64,
    `num_sentences` Int64,
    `num_keywords` Int64,
    `batch_id` String,
    `is_digest` UInt8,
    `entities_persons` String,
    `entities_orgs` String,
    `entities_geo` String,
    `num_persons` UInt16,
    `num_orgs` UInt16,
    `num_geo` UInt16,
    `ingest_object_name` String,
    `best_clean_len` UInt64
)
AS
SELECT
    dedup_key,
    argMax(id, score) AS id,
    argMax(title, score) AS title,
    argMax(link, score) AS link,
    argMax(published_at, score) AS published_at,
    argMax(source, score) AS source,
    argMax(raw_text, score) AS raw_text,
    argMax(clean_text, score) AS clean_text,
    argMax(nlp_text, score) AS nlp_text,
    argMax(summary, score) AS summary,
    argMax(keywords, score) AS keywords,

    -- NEW NLP EXTRAS (choose the same "best row" by score)
    argMax(lang, score) AS lang,
    argMax(keyphrases, score) AS keyphrases,
    argMax(sentiment_label, score) AS sentiment_label,
    argMax(sentiment_score, score) AS sentiment_score,

    argMax(text_length_chars, score) AS text_length_chars,
    argMax(num_sentences, score) AS num_sentences,
    argMax(num_keywords, score) AS num_keywords,
    argMax(batch_id, score) AS batch_id,
    argMax(is_digest, score) AS is_digest,
    argMax(entities_persons, score) AS entities_persons,
    argMax(entities_orgs, score) AS entities_orgs,
    argMax(entities_geo, score) AS entities_geo,
    argMax(num_persons, score) AS num_persons,
    argMax(num_orgs, score) AS num_orgs,
    argMax(num_geo, score) AS num_geo,
    argMax(ingest_object_name, score) AS ingest_object_name,
    argMax(clean_len, score) AS best_clean_len
FROM
(
    WITH (
        SELECT max(loaded_at)
        FROM media_intel.load_log
        WHERE layer = 'gold'
    ) AS max_loaded
    SELECT
        a.*,
        if(empty(a.link), a.id, a.link) AS dedup_key,
        lengthUTF8(coalesce(a.clean_text, '')) AS clean_len,
        toUInt8(clean_len >= 40) AS has_body,
        coalesce(ll.loaded_at, toDateTime64('1970-01-01 00:00:00', 9, 'Europe/Moscow')) AS loaded_at,
        toUInt8(coalesce(ll.loaded_at, toDateTime('1970-01-01 00:00:00')) = max_loaded) AS batch_priority,
        (has_body, clean_len, batch_priority, a.published_at, loaded_at) AS score
    FROM media_intel.articles AS a
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