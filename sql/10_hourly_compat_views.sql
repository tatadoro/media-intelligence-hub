/* sql/10_hourly_compat_views.sql
   Обёртки для унификации времени: dt -> ts_hour
*/

CREATE OR REPLACE VIEW media_intel.v_geo_hourly_ts AS
SELECT
  dt AS ts_hour,
  source,
  geo,
  mentions
FROM media_intel.v_geo_hourly;

CREATE OR REPLACE VIEW media_intel.v_keywords_hourly_ts AS
SELECT
  dt AS ts_hour,
  source,
  keyword,
  mentions
FROM media_intel.v_keywords_hourly;

CREATE OR REPLACE VIEW media_intel.v_sources_hourly_ts AS
SELECT
  dt AS ts_hour,
  source,
  articles
FROM media_intel.v_sources_hourly;