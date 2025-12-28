USE media_intel;

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS batch_id String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS is_digest UInt8 DEFAULT 0;

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS entities_persons String DEFAULT '',
    ADD COLUMN IF NOT EXISTS entities_orgs String DEFAULT '',
    ADD COLUMN IF NOT EXISTS entities_geo String DEFAULT '';

ALTER TABLE media_intel.articles
    ADD COLUMN IF NOT EXISTS num_persons Int64 DEFAULT 0,
    ADD COLUMN IF NOT EXISTS num_orgs Int64 DEFAULT 0,
    ADD COLUMN IF NOT EXISTS num_geo Int64 DEFAULT 0;