-include .env
export

.PHONY: up down ps logs restart reset \
        wait-minio wait-clickhouse \
        create-bucket init bootstrap smoke ch-show-schema clean-sql \
        views health quality topkw hour batches survival dupes report \
        gold load etl md-report reports etl-latest run validate-silver \
        validate-gold validate gate etl-latest-strict env-check env-ensure tg-raw tg-raw-latest tg-silver tg-silver-latest tg-gold tg-gold-latest tg-load tg-etl

PYTHON  ?= python
SILVER_GLOB ?= data/silver/articles_*_clean.json

# ----------- Telegram collector defaults -----------
TG_CHANNELS_FILE ?= config/telegram_channels.txt
TG_CHANNELS      ?=
TG_CHANNEL       ?= bbbreaking
TG_TARGET        ?= 50
TG_OUT_DIR       ?= data/raw
TG_DEBUG_DIR     ?= .
TG_MIH_ONLY      ?= 1
TG_MIH_SCHEMA    ?= list
TG_COMBINED      ?= 1
TG_HEADLESS      ?= 1

TG_RAW_COMBINED_GLOB    ?= $(TG_OUT_DIR)/articles_*_telegram_combined.json
TG_SILVER_COMBINED_GLOB ?= data/silver/articles_*_telegram_combined*_clean.json
TG_GOLD_COMBINED_GLOB   ?= data/gold/articles_*_telegram_combined*_processed.parquet
TG_RAW_CH_GLOB          ?= $(TG_OUT_DIR)/articles_*_telegram_$(TG_CHANNEL).json
TG_SILVER_CH_GLOB       ?= data/silver/articles_*_telegram_$(TG_CHANNEL)*_clean.json
TG_GOLD_CH_GLOB         ?= data/gold/articles_*_telegram_$(TG_CHANNEL)*_processed.parquet

PROJECT_DIR := $(CURDIR)
ENV_FILE    := $(PROJECT_DIR)/.env

COMPOSE_BIN ?= docker compose
COMPOSE     := $(COMPOSE_BIN) --project-directory $(PROJECT_DIR) --env-file $(ENV_FILE) \
               -f $(PROJECT_DIR)/docker-compose.yml

CH_DB       ?= $(or $(CH_DATABASE),$(CLICKHOUSE_DB),media_intel)
CH_USER     ?= $(or $(CH_USER),$(CLICKHOUSE_USER),admin)
CH_PASSWORD ?= $(or $(CH_PASSWORD),$(CLICKHOUSE_PASSWORD),admin12345)
CH_HOST     ?= $(or $(CH_HOST),localhost)
CH_PORT     ?= $(or $(CH_PORT),18123)
CH_TZ       ?= $(or $(CH_TZ),Europe/Belgrade)

MINIO_ENDPOINT ?= $(or $(MINIO_ENDPOINT),http://localhost:9000)
MINIO_ACCESS_KEY ?= $(or $(MINIO_ACCESS_KEY),admin)
MINIO_SECRET_KEY ?= $(or $(MINIO_SECRET_KEY),admin12345)
MINIO_BUCKET ?= $(or $(MINIO_BUCKET),mih)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

define banner
	@echo "============================================================"
	@echo "$(1)"
	@echo "============================================================"
endef

env-check:
	@$(call banner,"Env check")
	@echo "CH_USER=$(CH_USER)"
	@echo "CH_PASSWORD=$$(python - <<'PY'\nimport os\np=os.getenv('CH_PASSWORD') or os.getenv('CLICKHOUSE_PASSWORD') or ''\nprint(p[:2]+'***'+p[-2:]+' (len='+str(len(p))+')' if p else '(empty)')\nPY)"
	@echo "CH_DATABASE=$(CH_DB)"
	@echo "CH_HOST=$(CH_HOST)"
	@echo "CH_PORT=$(CH_PORT)"
	@echo "CLICKHOUSE_USER=$(or $(CLICKHOUSE_USER),$(CH_USER))"
	@echo "CLICKHOUSE_PASSWORD=$$(python - <<'PY'\nimport os\np=os.getenv('CLICKHOUSE_PASSWORD') or os.getenv('CH_PASSWORD') or ''\nprint(p[:2]+'***'+p[-2:]+' (len='+str(len(p))+')' if p else '(empty)')\nPY)"

env-ensure:
	@$(call banner,"Ensure env")
	@test -f .env || (echo "[ERROR] .env not found. Create it from .env.example"; exit 1)

up:
	@$(call banner,"Docker up")
	@$(COMPOSE) up -d

down:
	@$(call banner,"Docker down")
	@$(COMPOSE) down

ps:
	@$(call banner,"Docker ps")
	@$(COMPOSE) ps

logs:
	@$(call banner,"Docker logs")
	@$(COMPOSE) logs -f --tail 200

restart:
	@$(call banner,"Docker restart")
	@$(COMPOSE) restart

reset:
	@$(call banner,"Docker reset")
	@$(COMPOSE) down -v --remove-orphans

wait-minio:
	@$(call banner,"Wait MinIO")
	@bash scripts/wait_minio.sh

wait-clickhouse:
	@$(call banner,"Wait ClickHouse")
	@bash scripts/wait_clickhouse.sh

create-bucket:
	@$(call banner,"Create bucket")
	@bash scripts/create_bucket.sh

init:
	@$(call banner,"Init")
	@$(MAKE) env-ensure
	@$(MAKE) up
	@$(MAKE) wait-minio
	@$(MAKE) wait-clickhouse
	@$(MAKE) create-bucket

bootstrap:
	@$(call banner,"Bootstrap")
	@$(MAKE) init
	@$(MAKE) ch-show-schema

ch-show-schema:
	@$(call banner,"ClickHouse schema")
	@bash scripts/ch_run_sql.sh sql/00_ddl.sql
	@bash scripts/ch_run_sql.sh sql/00_views.sql

clean-sql:
	@$(call banner,"SQL clean")
	@bash scripts/ch_run_sql.sh sql/00_ddl.sql

views:
	@$(call banner,"Views")
	@bash scripts/ch_run_sql.sh sql/00_views.sql

health:
	@$(call banner,"Health checks")
	@bash scripts/ch_run_sql.sh sql/01_healthcheck.sql

quality:
	@$(call banner,"Content quality checks")
	@bash scripts/ch_run_sql.sh sql/02_content_quality.sql

topkw:
	@$(call banner,"Top keywords")
	@bash scripts/ch_run_sql.sh sql/03_top_keywords.sql

hour:
	@$(call banner,"By hour")
	@bash scripts/ch_run_sql.sh sql/04_by_hour.sql

batches:
	@$(call banner,"Batches")
	@bash scripts/ch_run_sql.sh sql/05_batches.sql

survival:
	@$(call banner,"Survival")
	@bash scripts/ch_run_sql.sh sql/06_survival.sql

dupes:
	@$(call banner,"Dupes")
	@bash scripts/ch_run_sql.sh sql/07_dupes.sql

report:
	@$(call banner,"Report")
	@bash scripts/ch_run_sql.sh sql/08_report.sql

md-report:
	@$(call banner,"Markdown report")
	@$(PYTHON) -m src.reporting.generate_report

reports:
	@$(call banner,"Reports list")
	@ls -1t reports/*.md | head -n 20 || true

run:
	@$(call banner,"Run RSS -> raw")
	@$(PYTHON) -m src.collectors.rss_collector

validate-silver:
	@$(call banner,"Validate silver")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_silver_json>"; exit 1)
	@$(PYTHON) scripts/validate_silver.py --input "$(IN)"

validate-gold:
	@$(call banner,"Validate gold")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_gold_parquet>"; exit 1)
	@$(PYTHON) scripts/validate_gold.py --input "$(IN)"

validate:
	@$(call banner,"Validate latest silver")
	@$(PYTHON) scripts/validate_silver.py --input "$$(ls -1t $(SILVER_GLOB) | head -n 1)"

gate:
	@$(call banner,"Quality gate")
	@bash scripts/ch_run_sql.sh sql/02_content_quality.sql

gold:
	@$(call banner,"Silver -> gold")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_silver_json>"; exit 1)
	@$(PYTHON) -m src.pipeline.silver_to_gold_local --input "$(IN)"

load:
	@$(call banner,"Gold -> ClickHouse")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_gold_parquet>"; exit 1)
	@test -n "$(BATCH_ID)" || (echo "[ERROR] Provide BATCH_ID=<iso_timestamp_utc>"; exit 1)
	@$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input "$(IN)" --batch-id "$(BATCH_ID)"

etl:
	@$(call banner,"ETL (raw->silver->gold->clickhouse)")
	@$(PYTHON) -m src.pipeline.clean_raw_to_silver_local
	@$(PYTHON) -m src.pipeline.silver_to_gold_local
	@$(PYTHON) -m src.pipeline.gold_to_clickhouse_local

etl-latest:
	@$(call banner,"ETL latest")
	@$(PYTHON) -m src.pipeline.etl_latest

etl-latest-strict:
	@$(MAKE) etl-latest STRICT=1

# ----------- Telegram pipeline (collector -> raw -> silver -> gold -> ClickHouse) -----------

tg-raw:
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	echo "[INFO] Telegram raw ingest. BATCH_ID=$$BATCH_ID"; \
	ARGS=""; \
	ARGS="$$ARGS --target $(TG_TARGET) --out-dir $(TG_OUT_DIR) --debug-dir $(TG_DEBUG_DIR) --mih-schema $(TG_MIH_SCHEMA)"; \
	if [ "$(TG_MIH_ONLY)" = "1" ]; then ARGS="$$ARGS --mih-only"; fi; \
	if [ "$(TG_HEADLESS)" = "1" ]; then ARGS="$$ARGS --headless"; fi; \
	if [ "$(TG_COMBINED)" = "1" ]; then ARGS="$$ARGS --combined"; fi; \
	if [ -n "$(TG_CHANNELS_FILE)" ] && [ -f "$(TG_CHANNELS_FILE)" ]; then \
		ARGS="$$ARGS --channels-file $(TG_CHANNELS_FILE)"; \
	elif [ -n "$(TG_CHANNELS)" ]; then \
		ARGS="$$ARGS --channels $(TG_CHANNELS)"; \
	else \
		ARGS="$$ARGS --channel $(TG_CHANNEL)"; \
	fi; \
	BATCH_ID="$$BATCH_ID" $(PYTHON) -m src.collectors.telegram_scraper $$ARGS

tg-raw-latest:
	@set -e; \
	if [ "$(TG_COMBINED)" = "1" ]; then \
		ls -1t $(TG_RAW_COMBINED_GLOB) 2>/dev/null | head -n 1; \
	else \
		ls -1t $(TG_RAW_CH_GLOB) 2>/dev/null | head -n 1; \
	fi

tg-silver:
	@set -e; \
	RAW_IN="$(RAW_IN)"; \
	if [ -z "$$RAW_IN" ]; then \
		if [ "$(TG_COMBINED)" = "1" ]; then RAW_IN="$$(ls -1t $(TG_RAW_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
		else RAW_IN="$$(ls -1t $(TG_RAW_CH_GLOB) 2>/dev/null | head -n 1)"; fi; \
	fi; \
	test -n "$$RAW_IN" || (echo "[ERROR] No raw file found for Telegram. Run: make tg-raw"; exit 1); \
	echo "[INFO] raw -> silver: $$RAW_IN"; \
	$(PYTHON) -m src.pipeline.clean_raw_to_silver_local --input "$$RAW_IN"

tg-silver-latest:
	@set -e; \
	if [ "$(TG_COMBINED)" = "1" ]; then \
		ls -1t $(TG_SILVER_COMBINED_GLOB) 2>/dev/null | head -n 1; \
	else \
		ls -1t $(TG_SILVER_CH_GLOB) 2>/dev/null | head -n 1; \
	fi

tg-gold:
	@set -e; \
	SILVER_IN="$(SILVER_IN)"; \
	if [ -z "$$SILVER_IN" ]; then \
		if [ "$(TG_COMBINED)" = "1" ]; then SILVER_IN="$$(ls -1t $(TG_SILVER_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
		else SILVER_IN="$$(ls -1t $(TG_SILVER_CH_GLOB) 2>/dev/null | head -n 1)"; fi; \
	fi; \
	test -n "$$SILVER_IN" || (echo "[ERROR] No silver file found for Telegram. Run: make tg-silver"; exit 1); \
	echo "[INFO] silver -> gold: $$SILVER_IN"; \
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input "$$SILVER_IN"

tg-gold-latest:
	@set -e; \
	if [ "$(TG_COMBINED)" = "1" ]; then \
		ls -1t $(TG_GOLD_COMBINED_GLOB) 2>/dev/null | head -n 1; \
	else \
		ls -1t $(TG_GOLD_CH_GLOB) 2>/dev/null | head -n 1; \
	fi

tg-load: env-ensure env-check
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	GOLD_IN="$(GOLD_IN)"; \
	if [ -z "$$GOLD_IN" ]; then \
		if [ "$(TG_COMBINED)" = "1" ]; then GOLD_IN="$$(ls -1t $(TG_GOLD_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
		else GOLD_IN="$$(ls -1t $(TG_GOLD_CH_GLOB) 2>/dev/null | head -n 1)"; fi; \
	fi; \
	test -n "$$GOLD_IN" || (echo "[ERROR] No gold parquet found for Telegram. Run: make tg-gold"; exit 1); \
	echo "[INFO] gold -> ClickHouse: $$GOLD_IN"; \
	echo "[INFO] Using batch-id: $$BATCH_ID"; \
	$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input "$$GOLD_IN" --batch-id "$$BATCH_ID"

tg-etl:
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	echo "[INFO] Telegram ETL. BATCH_ID=$$BATCH_ID"; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-raw; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-silver; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-gold; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-load

smoke: reset bootstrap
	$(PYTHON) -m src.utils.s3_smoke_test
	$(MAKE) etl-latest
