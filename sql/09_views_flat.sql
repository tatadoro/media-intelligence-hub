CREATE OR REPLACE VIEW media_intel.v_keywords_flat AS
SELECT
  published_at AS ts,
  source,
  lowerUTF8(trim(k)) AS keyword
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar(';', keywords) AS k
WHERE trim(k) != '';

CREATE OR REPLACE VIEW media_intel.v_keyphrases_flat AS
SELECT
  published_at AS ts,
  source,
  lowerUTF8(trim(k)) AS keyphrase
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar(';', keyphrases) AS k
WHERE trim(k) != '';

CREATE OR REPLACE VIEW media_intel.v_entities_persons_flat AS
SELECT
  published_at AS ts,
  source,
  lowerUTF8(trim(p)) AS person
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar(';', entities_persons_canon) AS p
WHERE trim(p) != '';

CREATE OR REPLACE VIEW media_intel.v_entities_orgs_flat AS
SELECT
  published_at AS ts,
  source,
  lowerUTF8(trim(o)) AS org
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar(';', entities_orgs_canon) AS o
WHERE trim(o) != '';

CREATE OR REPLACE VIEW media_intel.v_entities_geo_flat AS
SELECT
  published_at AS ts,
  source,
  lowerUTF8(trim(g)) AS geo
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar(';', entities_geo_canon) AS g
WHERE trim(g) != '';

CREATE OR REPLACE VIEW media_intel.v_persons_actions_flat AS
SELECT
  published_at AS ts,
  source,
  lowerUTF8(trim(splitByChar(':', pa)[1])) AS person,
  lowerUTF8(trim(v)) AS verb
FROM media_intel.articles_dedup
ARRAY JOIN splitByChar('|', persons_actions) AS pa
ARRAY JOIN splitByChar(',', splitByChar(':', pa)[2]) AS v
WHERE persons_actions != '' AND position(pa, ':') > 0 AND trim(v) != '';