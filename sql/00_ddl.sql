CREATE DATABASE IF NOT EXISTS media_intel;

CREATE TABLE IF NOT EXISTS media_intel.articles
(
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

    `text_length_chars` Int64,
    `num_sentences` Int64,
    `num_keywords` Int64,

    -- batch identity (preferred for grouping)
    `batch_id` String DEFAULT '',

    -- NER entities
    `entities_persons` String DEFAULT '',
    `entities_orgs` String DEFAULT '',
    `entities_geo` String DEFAULT '',
    `num_persons` UInt16 DEFAULT 0,
    `num_orgs` UInt16 DEFAULT 0,
    `num_geo` UInt16 DEFAULT 0,

    -- digest/noise flag (for analytics filtering)
    `is_digest` UInt8 DEFAULT 0,

    `ingest_object_name` String DEFAULT ''
)
ENGINE = MergeTree
ORDER BY (source, published_at, id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS media_intel.load_log
(
    `layer` String,
    `object_name` String,
    `loaded_at` DateTime,
    `rows_loaded` UInt64
)
ENGINE = Log;