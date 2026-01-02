-- sql/00_ddl.sql
-- Base schema for Media Intelligence Hub (ClickHouse)
-- Idempotent: CREATE IF NOT EXISTS + ALTER ADD COLUMN IF NOT EXISTS

CREATE DATABASE IF NOT EXISTS media_intel;

CREATE TABLE IF NOT EXISTS media_intel.articles
(
    `id` String,

    -- Telegram-specific (optional; empty for RSS)
    `uid` String DEFAULT '',
    `channel` String DEFAULT '',

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

    -- ingest traceability (gold parquet filename)
    `ingest_object_name` String DEFAULT '',

    -- NER entities (raw from Natasha / extractor)
    `entities_persons` String DEFAULT '',
    `entities_orgs` String DEFAULT '',
    `entities_geo` String DEFAULT '',
    `num_persons` UInt16 DEFAULT 0,
    `num_orgs` UInt16 DEFAULT 0,
    `num_geo` UInt16 DEFAULT 0,

    -- Canonical entities (after lemmatization + batch-level mapping)
    `persons` String DEFAULT '',
    `geo` String DEFAULT '',

    -- Actions (verbs) for persons (optional feature)
    `persons_actions` String DEFAULT '',
    `actions_verbs` String DEFAULT '',

    -- digest/noise flag (for analytics filtering)
    `is_digest` UInt8 DEFAULT 0
)
ENGINE = MergeTree
ORDER BY (source, published_at, id)
SETTINGS index_granularity = 8192;

-- ------------------------------------------------------------------
-- Online-migration block:
-- If table already exists, CREATE IF NOT EXISTS won't add new columns.
-- These ALTERs make bootstrap safely "upgrade" older schemas.
-- ------------------------------------------------------------------

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `uid` String DEFAULT '' AFTER `id`;

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `channel` String DEFAULT '' AFTER `uid`;

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `batch_id` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `ingest_object_name` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `entities_persons` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `entities_orgs` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `entities_geo` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `num_persons` UInt16 DEFAULT 0;

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `num_orgs` UInt16 DEFAULT 0;

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `num_geo` UInt16 DEFAULT 0;

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `persons` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `geo` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `persons_actions` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `actions_verbs` String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS `is_digest` UInt8 DEFAULT 0;

-- Load log (for simple audit)
CREATE TABLE IF NOT EXISTS media_intel.load_log
(
    `layer` String,
    `object_name` String,
    `loaded_at` DateTime,
    `rows_loaded` UInt64
)
ENGINE = Log;

-- ------------------------------------------------------------------
-- Ingestion coverage: monitoring table for collectors (RSS/Telegram/API)
-- Used by Superset charts (lag/volume) and operational checks.
-- ------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media_intel.ingestion_coverage
(
    `batch_id` String DEFAULT '',
    `source_type` LowCardinality(String) DEFAULT '',
    `source` String DEFAULT '',

    `run_started_at` DateTime64(3, 'Europe/Moscow'),
    `run_finished_at` DateTime64(3, 'Europe/Moscow'),
    `duration_ms` UInt32 DEFAULT 0,

    `items_found` UInt32 DEFAULT 0,
    `items_saved` UInt32 DEFAULT 0,

    `min_published_at` Nullable(DateTime64(3, 'Europe/Moscow')),
    `max_published_at` Nullable(DateTime64(3, 'Europe/Moscow')),

    `status` LowCardinality(String) DEFAULT 'ok',          -- ok / warn / error
    `error_message` String DEFAULT '',
    `raw_object_name` String DEFAULT ''                   -- mih_s3_key preferred; local path fallback
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(run_finished_at)
ORDER BY (source_type, source, run_finished_at, batch_id)
SETTINGS index_granularity = 8192;

-- Online-migration block for ingestion_coverage
ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `batch_id` String DEFAULT '';

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `source_type` LowCardinality(String) DEFAULT '';

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `source` String DEFAULT '';

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `run_started_at` DateTime64(3, 'Europe/Moscow');

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `run_finished_at` DateTime64(3, 'Europe/Moscow');

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `duration_ms` UInt32 DEFAULT 0;

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `items_found` UInt32 DEFAULT 0;

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `items_saved` UInt32 DEFAULT 0;

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `min_published_at` Nullable(DateTime64(3, 'Europe/Moscow'));

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `max_published_at` Nullable(DateTime64(3, 'Europe/Moscow'));

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `status` LowCardinality(String) DEFAULT 'ok';

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `error_message` String DEFAULT '';

ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `raw_object_name` String DEFAULT '';