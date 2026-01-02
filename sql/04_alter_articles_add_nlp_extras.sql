ALTER TABLE media_intel.articles
  ADD COLUMN IF NOT EXISTS lang LowCardinality(String) DEFAULT 'unknown';

ALTER TABLE media_intel.articles
  ADD COLUMN IF NOT EXISTS keyphrases String DEFAULT '';

ALTER TABLE media_intel.articles
  ADD COLUMN IF NOT EXISTS sentiment_label LowCardinality(String) DEFAULT 'neu';

ALTER TABLE media_intel.articles
  ADD COLUMN IF NOT EXISTS sentiment_score Float32 DEFAULT 0;