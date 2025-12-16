-include .env
export

.PHONY: up down ps logs restart reset \
        wait-minio wait-clickhouse \
        create-bucket init bootstrap ch-show-schema clean-sql \
        views health quality topkw hour batches survival dupes report \
        gold load etl md-report reports etl-latest run validate-silver

PYTHON  ?= python
COMPOSE ?= docker compose
SILVER_GLOB ?= data/silver/articles*_clean.json

# ----------- Local infra helpers -----------
up:
	$(COMPOSE) up -d
	$(MAKE) wait-clickhouse
	$(MAKE) wait-minio || true

down:
	$(COMPOSE) down

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=200

restart: down up

# clean start: removes volumes
reset:
	$(COMPOSE) down -v

# Wait for services to become ready (best-effort)
wait-clickhouse:
	@echo "[INFO] Waiting for ClickHouse..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		docker exec clickhouse clickhouse-client -q "SELECT 1" >/dev/null 2>&1 && { echo "[OK] ClickHouse ready"; exit 0; }; \
		sleep 1; \
	done; \
	echo "[WARN] ClickHouse not ready (timeout)."; exit 1

wait-minio:
	@echo "[INFO] Waiting for MinIO..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		curl -sf http://localhost:9000/minio/health/ready >/dev/null 2>&1 && { echo "[OK] MinIO ready"; exit 0; }; \
		sleep 1; \
	done; \
	echo "[WARN] MinIO not ready (timeout)."; exit 1

# Cleanup local SQL junk files
clean-sql:
	rm -f sql/.DS_Store

# MinIO bucket (requires mc installed locally)
# - if mc is not installed, print hint and exit with success (so bootstrap still works)
# - if MinIO is not ready yet, skip without failing bootstrap
create-bucket:
	@command -v mc >/dev/null 2>&1 || { \
		echo "[WARN] mc not found. Skip bucket creation."; exit 0; \
	}
	@test -n "$$MINIO_ENDPOINT" -a -n "$$MINIO_ACCESS_KEY" -a -n "$$MINIO_SECRET_KEY" -a -n "$$MINIO_BUCKET" || { \
		echo "[WARN] MinIO env is not set (MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY/MINIO_BUCKET). Skip."; \
		exit 0; \
	}
	@curl -sf http://localhost:9000/minio/health/ready >/dev/null 2>&1 || { \
		echo "[WARN] MinIO not ready yet. Skip bucket creation."; \
		exit 0; \
	}
	mc alias set local "$$MINIO_ENDPOINT" "$$MINIO_ACCESS_KEY" "$$MINIO_SECRET_KEY" >/dev/null
	mc mb -p "local/$$MINIO_BUCKET" || true

# ClickHouse init: DDL + views
init:
	./scripts/ch_run_sql.sh sql/00_ddl.sql
	$(MAKE) views

# One-command local bootstrap (robust)
bootstrap: clean-sql up init ps
	$(MAKE) create-bucket || true

# Quick schema check (debug helper)
ch-show-schema:
	./scripts/ch_run_sql.sh sql/00_ddl.sql
	./scripts/ch_run_sql.sh sql/00_views.sql
	./scripts/ch_run_sql.sh sql/01_healthcheck.sql

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

md-report:
	@echo "Generating Markdown report..."
	@ARGS=""; \
	if [ -n "$(LAST_HOURS)" ]; then ARGS="$$ARGS --last-hours $(LAST_HOURS)"; fi; \
	if [ -n "$(FROM)" ]; then ARGS="$$ARGS --from \"$(FROM)\""; fi; \
	if [ -n "$(TO)" ]; then ARGS="$$ARGS --to \"$(TO)\""; fi; \
	if [ -n "$(TABLE)" ]; then ARGS="$$ARGS --table $(TABLE)"; fi; \
	if [ -n "$(TOP_K)" ]; then ARGS="$$ARGS --top-k $(TOP_K)"; fi; \
	if [ -n "$(OUTDIR)" ]; then ARGS="$$ARGS --outdir $(OUTDIR)"; fi; \
	python -m src.reporting.generate_report $$ARGS

# Run SQL reports + generate Markdown report
# Run SQL reports + generate Markdown report
# You can pass params for Markdown report:
#   make reports LAST_HOURS=6
#   make reports FROM="2025-12-10 14:00:00" TO="2025-12-10 19:00:00"
#   make reports TABLE=articles_dedup TOP_K=10
reports:
	@$(MAKE) report
	@$(MAKE) md-report LAST_HOURS="$(LAST_HOURS)" FROM="$(FROM)" TO="$(TO)" TABLE="$(TABLE)" TOP_K="$(TOP_K)" OUTDIR="$(OUTDIR)"
	@echo "Done: SQL + Markdown reports generated."

# ----------- ETL helpers -----------

# Validate silver input contract
# usage:
#   make validate-silver IN=data/silver/xxx_clean.json
validate-silver:
	$(PYTHON) scripts/validate_silver.py --input $(IN)

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
	$(MAKE) validate-silver IN=$(IN)
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input $(IN)
	$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input data/gold/$(notdir $(IN:_clean.json=_processed.parquet))
	$(MAKE) reports
etl-latest:
	@LATEST="$$(ls -t $(SILVER_GLOB) 2>/dev/null | head -n 1)"; \
	if [ -z "$$LATEST" ]; then \
		echo "[ERROR] No files found: $(SILVER_GLOB)"; \
		exit 1; \
	fi; \
	echo "[INFO] Using latest silver file: $$LATEST"; \
	$(MAKE) etl IN="$$LATEST"