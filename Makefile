-include .env
export

.PHONY: up down ps logs restart reset \
        wait-minio wait-clickhouse \
        create-bucket init bootstrap smoke ch-show-schema clean-sql \
        views health quality topkw hour batches survival dupes report \
        gold load etl md-report reports etl-latest run validate-silver \
        validate-gold validate gate etl-latest-strict env-check env-ensure

PYTHON  ?= python
SILVER_GLOB ?= data/silver/articles_*_clean.json

PROJECT_DIR := $(CURDIR)
ENV_FILE    := $(PROJECT_DIR)/.env

COMPOSE_BIN ?= docker compose
COMPOSE     := $(COMPOSE_BIN) --project-directory $(PROJECT_DIR) --env-file $(ENV_FILE) \
               -f $(PROJECT_DIR)/docker-compose.yml \
               -f $(PROJECT_DIR)/docker-compose.airflow.yml \
               -f $(PROJECT_DIR)/docker-compose.superset.yml

# ----------- Connection defaults (local scripts) -----------
# These are used by scripts/ch_run_sql.sh and src.reporting.generate_report.
# Priority: real env (docker compose) > .env (-include) > defaults

CH_USER     ?= admin
CH_PASSWORD ?= $(CLICKHOUSE_PASSWORD)
CH_DATABASE ?= media_intel
CH_HOST     ?= localhost
CH_PORT     ?= 18123

env-check:
	@echo "[INFO] Effective env for ClickHouse (Makefile)"
	@echo "CH_USER=$(CH_USER)"
	@if [ -n "$(CH_PASSWORD)" ]; then echo "CH_PASSWORD=$$(printf '%s' "$(CH_PASSWORD)" | sed -E 's/^(.{2}).*(.{2})$$/\1***\2/') (len=$$(printf '%s' "$(CH_PASSWORD)" | wc -c | tr -d ' '))"; else echo "CH_PASSWORD=<empty>"; fi
	@echo "CH_DATABASE=$(CH_DATABASE)"
	@echo "CH_HOST=$(CH_HOST)"
	@echo "CH_PORT=$(CH_PORT)"
	@echo "CLICKHOUSE_USER=$(CLICKHOUSE_USER)"
	@if [ -n "$(CLICKHOUSE_PASSWORD)" ]; then echo "CLICKHOUSE_PASSWORD=$$(printf '%s' "$(CLICKHOUSE_PASSWORD)" | sed -E 's/^(.{2}).*(.{2})$$/\1***\2/') (len=$$(printf '%s' "$(CLICKHOUSE_PASSWORD)" | wc -c | tr -d ' '))"; else echo "CLICKHOUSE_PASSWORD=<empty>"; fi
	@python scripts/env_check.py

env-ensure:
	@test -f "$(ENV_FILE)" || (echo "[ERROR] .env not found: $(ENV_FILE)"; exit 2)
	@python scripts/env_check.py

# ----------- Local infra helpers -----------

up: env-ensure
	$(COMPOSE) up -d
	$(MAKE) wait-clickhouse
	$(MAKE) wait-minio || true

down:
	$(COMPOSE) down

ps: env-ensure
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
		docker exec clickhouse clickhouse-client -u "$${CLICKHOUSE_USER:-admin}" --password "$${CLICKHOUSE_PASSWORD:-}" -q "SELECT 1" >/dev/null 2>&1 && { echo "[OK] ClickHouse ready"; exit 0; }; \
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
bootstrap: env-ensure clean-sql up init ps
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
	$(PYTHON) -m src.reporting.generate_report $$ARGS

upload-report: env-check
	@test -n "$(REPORT)" || (echo "[ERROR] REPORT is required. Example: make upload-report REPORT=reports/daily_report_xxx.md LAST_HOURS=6" && exit 2)
	./scripts/upload_report_to_minio.sh "$(REPORT)" "$(LAST_HOURS)"

etl-and-report: env-ensure env-check
	@echo "[INFO] Running ETL (latest, strict)..."
	@$(MAKE) etl-latest-strict
	@echo
	@echo "[INFO] Generating Markdown report..."
	@$(MAKE) md-report LAST_HOURS=$(LAST_HOURS)
	@echo
	@echo "[INFO] Uploading latest report to MinIO..."
	@REPORT_FILE=$$(ls -1t reports/daily_report_*.md | head -n 1); \
	echo "[INFO] Latest report: $$REPORT_FILE"; \
	$(MAKE) upload-report REPORT="$$REPORT_FILE" LAST_HOURS=$(LAST_HOURS)

# Run only SQL reports (no Markdown)
reports-sql:
	@$(MAKE) report
	@echo "Done: SQL reports generated."

# Run SQL reports + generate Markdown report (explicit)
reports:
	@$(MAKE) reports-sql
	@$(MAKE) md-report LAST_HOURS="$(LAST_HOURS)" FROM="$(FROM)" TO="$(TO)" TABLE="$(TABLE)" TOP_K="$(TOP_K)" OUTDIR="$(OUTDIR)"
	@echo "Done: SQL + Markdown reports generated."

# ----------- ETL helpers -----------

validate-silver:
	$(PYTHON) scripts/validate_silver.py --input $(IN)

validate-gold:
	$(PYTHON) scripts/validate_gold.py --input $(IN)

validate:
	@echo "[INFO] Running validations..."
	@if [ -n "$(SILVER_IN)" ]; then $(MAKE) validate-silver IN="$(SILVER_IN)"; else echo "[SKIP] SILVER_IN not set"; fi
	@if [ -n "$(GOLD_IN)" ]; then $(MAKE) validate-gold IN="$(GOLD_IN)"; else echo "[SKIP] GOLD_IN not set"; fi
	@$(MAKE) health || true
	@echo "[OK] validate finished"

GATE_MIN_ROWS          ?= 20
GATE_MIN_BODY_SHARE    ?= 0.05
GATE_MIN_BODY_CHARS    ?= 50
GATE_MAX_EMPTY_SUMMARY ?= 0.95
GATE_MAX_EMPTY_KEYWORDS?= 0.95
STRICT                 ?= 0

gate:
	@ARGS="--input $(IN) --min-rows $(GATE_MIN_ROWS) --min-body-share $(GATE_MIN_BODY_SHARE) --min-body-chars $(GATE_MIN_BODY_CHARS) --max-empty-summary-share $(GATE_MAX_EMPTY_SUMMARY) --max-empty-keywords-share $(GATE_MAX_EMPTY_KEYWORDS)"; \
	if [ "$(STRICT)" = "1" ]; then ARGS="$$ARGS --strict"; fi; \
	$(PYTHON) scripts/quality_gate_gold.py $$ARGS

gold:
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input $(IN)

load:
	$(MAKE) validate-gold IN=$(IN)
	$(MAKE) gate IN=$(IN) STRICT=$(STRICT)
	$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input $(IN)

etl:
	$(MAKE) validate-silver IN=$(IN)
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input $(IN)
	$(MAKE) load IN=data/gold/$(notdir $(IN:_clean.json=_processed.parquet)) STRICT=$(STRICT)
	$(MAKE) reports-sql

etl-latest:
	@LATEST="$$(ls -t $(SILVER_GLOB) 2>/dev/null | head -n 1)"; \
	if [ -z "$$LATEST" ]; then \
		echo "[ERROR] No files found: $(SILVER_GLOB)"; \
		exit 1; \
	fi; \
	echo "[INFO] Using latest silver file: $$LATEST"; \
	$(MAKE) etl IN="$$LATEST"

etl-latest-strict:
	@$(MAKE) etl-latest STRICT=1

smoke: reset bootstrap
	$(PYTHON) -m src.utils.s3_smoke_test
	$(MAKE) etl-latest
