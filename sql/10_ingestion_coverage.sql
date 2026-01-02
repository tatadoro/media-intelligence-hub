-- sql/10_ingestion_coverage.sql
-- Coverage / lag / errors per source per run

CREATE TABLE IF NOT EXISTS media_intel.ingestion_coverage
(
    `batch_id` String,

    `source_type` LowCardinality(String),  -- rss | telegram | api (на будущее)
    `source` LowCardinality(String),       -- lenta.ru | telegram:lenta_ru | ...

    `run_started_at` DateTime64(3, 'Europe/Moscow'),
    `run_finished_at` DateTime64(3, 'Europe/Moscow'),
    `duration_ms` UInt32,

    `items_found` UInt32,
    `items_saved` UInt32,

    `min_published_at` Nullable(DateTime64(3, 'Europe/Moscow')),
    `max_published_at` Nullable(DateTime64(3, 'Europe/Moscow')),

    `status` LowCardinality(String),       -- ok | warn | error
    `error_message` String,

    -- куда реально сохранили raw (s3 key или локальный путь)
    `raw_object_name` String DEFAULT ''
)
ENGINE = MergeTree
ORDER BY (source_type, source, run_started_at, batch_id)
SETTINGS index_granularity = 8192;

-- Online-migration (на случай, если таблицу уже создавали урезанной)
ALTER TABLE media_intel.ingestion_coverage
    ADD COLUMN IF NOT EXISTS `raw_object_name` String DEFAULT '';

CREATE OR REPLACE VIEW media_intel.ingestion_coverage_latest AS
SELECT
    source_type,
    source,

    argMax(batch_id, run_finished_at)        AS batch_id,
    argMax(run_started_at, run_finished_at)  AS run_started_at,
    max(run_finished_at)                    AS run_finished_at,
    argMax(duration_ms, run_finished_at)     AS duration_ms,

    argMax(items_found, run_finished_at)     AS items_found,
    argMax(items_saved, run_finished_at)     AS items_saved,

    argMax(min_published_at, run_finished_at) AS min_published_at,
    argMax(max_published_at, run_finished_at) AS max_published_at,

    argMax(status, run_finished_at)          AS status,
    argMax(error_message, run_finished_at)   AS error_message,
    argMax(raw_object_name, run_finished_at) AS raw_object_name,

    if(
      isNull(argMax(max_published_at, run_finished_at)),
      NULL,
      dateDiff('minute', argMax(max_published_at, run_finished_at), now('Europe/Moscow'))
    ) AS lag_minutes
FROM media_intel.ingestion_coverage
GROUP BY source_type, source;