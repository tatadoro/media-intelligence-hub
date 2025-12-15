.PHONY: views health quality topkw hour batches survival dupes report gold load etl

PYTHON ?= python

# ----------- ClickHouse reports -----------
views:
	./scripts/ch_run_sql.sh sql/00_views.sql

health:
	./scripts/ch_run_sql.sh sql/01_healthcheck.sql

quality:
	./scripts/ch_run_sql.sh sql/02_content_quality.sql

topkw:
	./scripts/ch_run_sql.sh sql/03_top_keywords.sql

hour:
	./scripts/ch_run_sql.sh sql/04_topics_by_hour.sql

batches:
	./scripts/ch_run_sql.sh sql/05_batches.sql

survival:
	./scripts/ch_run_sql.sh sql/06_batch_survival.sql

dupes:
	./scripts/ch_run_sql.sh sql/07_batch_internal_dupes.sql

report:
	./scripts/ch_run_sql.sh sql/00_views.sql
	./scripts/ch_run_sql.sh sql/01_healthcheck.sql
	./scripts/ch_run_sql.sh sql/02_content_quality.sql
	./scripts/ch_run_sql.sh sql/03_top_keywords.sql
	./scripts/ch_run_sql.sh sql/04_topics_by_hour.sql
	./scripts/ch_run_sql.sh sql/05_batches.sql
	./scripts/ch_run_sql.sh sql/06_batch_survival.sql
	./scripts/ch_run_sql.sh sql/07_batch_internal_dupes.sql

# ----------- ETL helpers -----------
# 1) Silver -> Gold
# usage:
#   make gold IN=data/silver/xxx_clean.json
gold:
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input $(IN)

# 2) Gold -> ClickHouse
# usage:
#   make load IN=data/gold/xxx_processed.parquet
load:
	$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input $(IN)

# 3) Full run: Silver -> Gold -> ClickHouse -> Report
# usage:
#   make etl IN=data/silver/xxx_clean.json
etl:
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input $(IN)
	$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input data/gold/$(notdir $(IN:_clean.json=_processed.parquet))
	$(MAKE) report