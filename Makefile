# ============================================================
# Makefile проекта Media Intelligence Hub
# Команды для локального запуска инфраструктуры (Docker),
# прогонов пайплайна (raw->silver->gold->ClickHouse),
# проверок качества, отчётов и Telegram-ингеста.
# ============================================================

# ------------------------------------------------------------
# Загружаем переменные окружения из .env (если файл существует)
# -include: не падаем, если .env пока нет
# export: делаем переменные доступными для команд/скриптов
# ------------------------------------------------------------
-include .env
export

# ------------------------------------------------------------
# PHONY-таргеты: перечисляем команды make, которые не являются
# файлами (чтобы make не пытался искать одноимённые файлы)
# ------------------------------------------------------------
.PHONY: up down ps logs restart reset \
        wait-minio wait-clickhouse \
        create-bucket init bootstrap smoke ch-show-schema clean-sql \
        views health quality topkw hour batches survival dupes report \
        gold load etl md-report reports etl-latest run validate-silver \
        validate-gold validate gate etl-latest-strict env-check env-ensure \
        tg-raw tg-raw-latest tg-raw-latest-any tg-raw-latest-chan \
        tg-silver tg-silver-latest tg-gold tg-gold-latest \
        tg-load tg-etl tg-etl-refresh \
        superset-up superset-down superset-stop superset-ps superset-logs \
        superset-backup superset-backup-list superset-restore \
        etl-latest-rss etl-latest-tg etl-latest-all \
        stopwords-auto stopwords-auto-show dupes-json report-json topkw-json hour-json \
        rss-raw rss-raw-latest rss-silver rss-silver-latest rss-gold rss-gold-latest rss-load rss-etl rss-etl-refresh \
        ch ch-file \
        ch-running-check ch-http-ping

# ------------------------------------------------------------
# Настройки Python/путей по умолчанию
# ------------------------------------------------------------
PYTHON      ?= python
SILVER_GLOB ?= data/silver/articles_*_clean.json

# --- Source-specific latest silver globs ---
# ВАЖНО: RSS и TG должны быть разными (иначе "latest rss" может схватить telegram)
RSS_SILVER_GLOB ?= data/silver/articles_*_rss*_clean.json
TG_SILVER_GLOB  ?= data/silver/articles_*telegram*_clean.json

# ============================================================
# RSS defaults (tg-style combined)
# ============================================================
RSS_RAW_DIR      ?= data/raw
RSS_COMBINED     ?= 1          # 1 = сохраняем combined raw как articles_*_rss_combined.json
RSS_RAW_BACKEND  ?= local      # для локальной разработки: raw кладём в data/raw

RSS_RAW_COMBINED_GLOB    ?= $(RSS_RAW_DIR)/articles_*_rss_combined.json
RSS_SILVER_COMBINED_GLOB ?= data/silver/articles_*_rss_combined*_clean.json
RSS_GOLD_COMBINED_GLOB   ?= data/gold/articles_*_rss_combined*_processed.parquet

etl-latest-rss:
	@$(MAKE) rss-etl STRICT=1

etl-latest-tg:
	@$(MAKE) tg-etl

tg-etl-refresh:
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	echo "[INFO] Telegram ETL + refresh. BATCH_ID=$$BATCH_ID"; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-raw; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-silver; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-gold; \
	BATCH_ID="$$BATCH_ID" $(MAKE) tg-load; \
	$(MAKE) refresh

# NEW: RSS ETL + refresh (симметрия с tg-etl-refresh)
rss-etl-refresh:
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	echo "[INFO] RSS ETL + refresh (tg-style). BATCH_ID=$$BATCH_ID"; \
	BATCH_ID="$$BATCH_ID" $(MAKE) rss-etl STRICT="$(STRICT)"; \
	$(MAKE) refresh

etl-latest-all: etl-latest-rss etl-latest-tg
	@echo "============================================================"
	@echo "[OK] ETL latest for ALL sources finished."
	@echo "============================================================"

# ============================================================
# Telegram collector defaults
# Настройки по умолчанию для парсера Telegram (scraper)
# ============================================================
TG_CHANNELS_FILE ?= config/telegram_channels.txt  # файл со списком каналов
TG_CHANNELS      ?=                               # список каналов через запятую
TG_CHANNEL       ?= bbbreaking                    # одиночный канал по умолчанию
TG_TARGET        ?= 50                            # сколько постов собирать
TG_OUT_DIR       ?= data/raw                      # куда класть raw
TG_DEBUG_DIR     ?= .                             # куда писать debug (если есть)
TG_MIH_ONLY      ?= 1                             # собирать только MIH-схему
TG_MIH_SCHEMA    ?= list                          # схема MIH (list/json и т.п.)
TG_COMBINED      ?= 1                             # объединять каналы в один файл
TG_HEADLESS      ?= 1                             # запуск браузера headless

# NEW: принудительно использовать один канал TG_CHANNEL (игнорировать channels-file/channels)
TG_FORCE_CHANNEL ?= 0                             # 1 = всегда --channel $(TG_CHANNEL)

# --- Normalize возможных "липких" пробелов/табов в значениях (частая причина падений) ---
TG_CHANNELS_FILE := $(strip $(TG_CHANNELS_FILE))
TG_CHANNELS      := $(strip $(TG_CHANNELS))
TG_CHANNEL       := $(strip $(TG_CHANNEL))
TG_TARGET        := $(strip $(TG_TARGET))
TG_OUT_DIR       := $(strip $(TG_OUT_DIR))
TG_DEBUG_DIR     := $(strip $(TG_DEBUG_DIR))
TG_MIH_ONLY      := $(strip $(TG_MIH_ONLY))
TG_MIH_SCHEMA    := $(strip $(TG_MIH_SCHEMA))
TG_COMBINED      := $(strip $(TG_COMBINED))
TG_HEADLESS      := $(strip $(TG_HEADLESS))
TG_FORCE_CHANNEL := $(strip $(TG_FORCE_CHANNEL))

# Шаблоны путей для последних файлов (combined/канал)
TG_RAW_COMBINED_GLOB    ?= $(TG_OUT_DIR)/articles_*_telegram_combined.json
TG_SILVER_COMBINED_GLOB ?= data/silver/articles_*_telegram_combined*_clean.json
TG_GOLD_COMBINED_GLOB   ?= data/gold/articles_*_telegram_combined*_processed.parquet

TG_RAW_CH_GLOB          ?= $(TG_OUT_DIR)/articles_*_telegram_$(TG_CHANNEL).json
TG_SILVER_CH_GLOB       ?= data/silver/articles_*_telegram_$(TG_CHANNEL)*_clean.json
TG_GOLD_CH_GLOB         ?= data/gold/articles_*_telegram_$(TG_CHANNEL)*_processed.parquet

# Универсальные (устойчивые) globs: не зависят от TG_COMBINED и пробелов в TG_CHANNEL
TG_RAW_ANY_GLOB    ?= $(TG_OUT_DIR)/articles_*_telegram*.json
TG_SILVER_ANY_GLOB ?= data/silver/articles_*_telegram*_clean.json
TG_GOLD_ANY_GLOB   ?= data/gold/articles_*_telegram*_processed.parquet

# ------------------------------------------------------------
# Telegram: сбор аргументов на уровне Make (без shell if)
# ------------------------------------------------------------
TG_SCRAPER_ARGS := --target $(TG_TARGET) --out-dir $(TG_OUT_DIR) --debug-dir $(TG_DEBUG_DIR) --mih-schema $(TG_MIH_SCHEMA)

ifeq ($(TG_MIH_ONLY),1)
TG_SCRAPER_ARGS += --mih-only
endif

ifeq ($(TG_HEADLESS),1)
TG_SCRAPER_ARGS += --headless
endif

ifeq ($(TG_COMBINED),1)
TG_SCRAPER_ARGS += --combined
endif

# ВАЖНО: принудительный одиночный канал должен иметь максимальный приоритет
ifeq ($(TG_FORCE_CHANNEL),1)
TG_SCRAPER_ARGS += --channel $(TG_CHANNEL)
else
  # Если channels-file существует — используем его
  ifneq ($(wildcard $(TG_CHANNELS_FILE)),)
TG_SCRAPER_ARGS += --channels-file $(TG_CHANNELS_FILE)
  else ifneq ($(strip $(TG_CHANNELS)),)
TG_SCRAPER_ARGS += --channels $(TG_CHANNELS)
  else
TG_SCRAPER_ARGS += --channel $(TG_CHANNEL)
  endif
endif

# ============================================================
# Базовые переменные проекта/окружения
# ============================================================
PROJECT_DIR := $(CURDIR)
ENV_FILE    := $(PROJECT_DIR)/.env

# ------------------------------------------------------------
# Docker Compose для основного стека проекта (ClickHouse/MinIO/и т.п.)
# ------------------------------------------------------------
COMPOSE_BIN ?= docker compose
COMPOSE     := $(COMPOSE_BIN) --project-directory $(PROJECT_DIR) --env-file $(ENV_FILE) \
               -f $(PROJECT_DIR)/docker-compose.yml

# ------------------------------------------------------------
# Docker Compose для полного стека (core + Superset) — одна команда без orphan warning
# ------------------------------------------------------------
COMPOSE_ALL := COMPOSE_IGNORE_ORPHANS=1 $(COMPOSE_BIN) --project-directory $(PROJECT_DIR) --env-file $(ENV_FILE) \
               -f $(PROJECT_DIR)/docker-compose.yml \
               -f $(PROJECT_DIR)/docker-compose.superset.yml

# ------------------------------------------------------------
# Docker Compose для полного стека (core + Superset + Airflow)
# ------------------------------------------------------------
COMPOSE_FULL := COMPOSE_IGNORE_ORPHANS=1 $(COMPOSE_BIN) --project-directory $(PROJECT_DIR) --env-file $(ENV_FILE) \
               -f $(PROJECT_DIR)/docker-compose.yml \
               -f $(PROJECT_DIR)/docker-compose.superset.yml \
               -f $(PROJECT_DIR)/docker-compose.airflow.yml

# ------------------------------------------------------------
# ClickHouse (переменные с fallback’ами)
# ВАЖНО: НЕ ссылаемся на CH_* внутри определения CH_* (иначе рекурсия)
#
# ВАЖНО 2: когда make запускается ВНУТРИ контейнера (airflow-webserver),
# localhost указывает на сам контейнер, поэтому для ping нужен clickhouse:8123.
# На хосте, наоборот, используем localhost:18123 (проброшенный порт).
# ------------------------------------------------------------

# Определяем, выполняемся ли мы внутри Docker-контейнера
IN_DOCKER := $(shell test -f /.dockerenv && echo 1 || echo 0)

ENV_CH_DATABASE  := $(CH_DATABASE)
ENV_CH_USER      := $(CH_USER)
ENV_CH_PASSWORD  := $(CH_PASSWORD)
ENV_CH_HOST      := $(CH_HOST)
ENV_CH_PORT      := $(CH_PORT)
ENV_CH_TZ        := $(CH_TZ)

CH_DB       := $(or $(ENV_CH_DATABASE),$(CLICKHOUSE_DB),media_intel)
CH_USER     := $(or $(ENV_CH_USER),$(CLICKHOUSE_USER),admin)
CH_PASSWORD := $(or $(ENV_CH_PASSWORD),$(CLICKHOUSE_PASSWORD),admin12345)

ifeq ($(IN_DOCKER),1)
  # Внутри docker-сети: имя сервиса + внутренний порт
  CH_HOST := $(or $(ENV_CH_HOST),clickhouse)
  CH_PORT := $(or $(ENV_CH_PORT),8123)
else
  # На хосте: проброшенный порт
  CH_HOST := $(or $(ENV_CH_HOST),localhost)
  CH_PORT := $(or $(ENV_CH_PORT),18123)
endif

CH_TZ       := $(or $(ENV_CH_TZ),Europe/Belgrade)

# NEW: формат вывода clickhouse-client для bash-скриптов (scripts/ch_run_sql.sh)
CH_FORMAT ?= TSVRaw

# NEW: безопасный способ определить running контейнер ClickHouse (а не stopped/exited)
CH_CONTAINER_ID ?= $(shell docker ps -q --filter "name=^/clickhouse$$" --filter "status=running" | head -n 1)

# ------------------------------------------------------------
# MinIO (переменные с fallback’ами) — также без рекурсии
# ------------------------------------------------------------
ENV_MINIO_ENDPOINT   := $(MINIO_ENDPOINT)
ENV_MINIO_ACCESS_KEY := $(MINIO_ACCESS_KEY)
ENV_MINIO_SECRET_KEY := $(MINIO_SECRET_KEY)
ENV_MINIO_BUCKET     := $(MINIO_BUCKET)

MINIO_ENDPOINT   := $(or $(ENV_MINIO_ENDPOINT),http://localhost:9000)
MINIO_ACCESS_KEY := $(or $(ENV_MINIO_ACCESS_KEY),admin)
MINIO_SECRET_KEY := $(or $(ENV_MINIO_SECRET_KEY),admin12345)
MINIO_BUCKET     := $(or $(ENV_MINIO_BUCKET),mih)

# ============================================================
# Helpers
# ============================================================

define banner
	@echo "============================================================"
	@echo $(1)
	@echo "============================================================"
endef

# ============================================================
# ClickHouse readiness helpers (для gate и любых SQL проверок)
# ============================================================

# Определяем, что make выполняется внутри Docker-контейнера (airflow-webserver и т.п.)
IN_DOCKER := $(shell test -f /.dockerenv && echo 1 || echo 0)

# В docker-сети ClickHouse доступен по имени сервиса и внутреннему порту
CH_DOCKER_HOST ?= clickhouse
CH_DOCKER_PORT ?= 8123

# Для ping выбираем корректный адрес:
# - внутри Docker: clickhouse:8123
# - на хосте: $(CH_HOST):$(CH_PORT)
CH_PING_HOST := $(if $(filter 1,$(IN_DOCKER)),$(CH_DOCKER_HOST),$(CH_HOST))
CH_PING_PORT := $(if $(filter 1,$(IN_DOCKER)),$(CH_DOCKER_PORT),$(CH_PORT))

# Проверяем, что clickhouse контейнер реально RUNNING (иначе docker exec даст "container is not running")
ch-running-check:
	@if [ -z "$(CH_CONTAINER_ID)" ]; then \
	  echo "[ERROR] ClickHouse container 'clickhouse' is not running."; \
	  docker ps -a --filter "name=clickhouse" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"; \
	  exit 1; \
	fi

# Более стабильная проверка: ping по HTTP
ch-http-ping:
	@curl -sfS "http://$(CH_PING_HOST):$(CH_PING_PORT)/ping" >/dev/null || \
	  (echo "[ERROR] ClickHouse ping failed at http://$(CH_PING_HOST):$(CH_PING_PORT)/ping"; exit 1)

# ============================================================
# Env
# ============================================================

env-check:
	@$(call banner,"Env check")
	@echo "CH_USER=$(CH_USER)"
	@echo "CH_PASSWORD=$${CH_PASSWORD:+(set)}"
	@echo "CH_PASSWORD_LEN=$${#CH_PASSWORD}"
	@echo "CH_DATABASE=$(CH_DB)"
	@echo "CH_HOST=$(CH_HOST)"
	@echo "CH_PORT=$(CH_PORT)"
	@echo "CH_FORMAT=$(CH_FORMAT)"
	@echo "CH_CONTAINER_ID=$(CH_CONTAINER_ID)"
	@echo "CLICKHOUSE_USER=$(or $(CLICKHOUSE_USER),$(CH_USER))"
	@echo "CLICKHOUSE_PASSWORD=$${CLICKHOUSE_PASSWORD:+(set)}"
	@echo "CLICKHOUSE_PASSWORD_LEN=$${#CLICKHOUSE_PASSWORD}"

env-ensure:
	@$(call banner,"Ensure env")
	@test -f .env || (echo "[ERROR] .env not found. Create it from .env.example"; exit 1)

# ============================================================
# Docker (основной стек проекта: ClickHouse/MinIO/…)
# ============================================================

up: env-ensure
	@$(call banner,"Docker up (core + Superset + Airflow)")
	@test -f docker-compose.superset.yml || (echo "[ERROR] docker-compose.superset.yml not found"; exit 1)
	@test -f docker-compose.airflow.yml || (echo "[ERROR] docker-compose.airflow.yml not found"; exit 1)

	@$(COMPOSE_FULL) up -d clickhouse minio superset-postgres superset-redis airflow-postgres
	@$(COMPOSE_FULL) up superset-init
	@$(COMPOSE_FULL) up airflow-init
	@$(COMPOSE_FULL) up -d superset airflow-webserver airflow-scheduler
	@$(COMPOSE_FULL) ps

down:
	@$(call banner,"Docker down (core + Superset + Airflow)")
	@$(COMPOSE_FULL) down

ps:
	@$(call banner,"Docker ps (core + Superset + Airflow)")
	@$(COMPOSE_FULL) ps

logs:
	@$(call banner,"Docker logs")
	@$(COMPOSE) logs -f --tail 200

restart:
	@$(call banner,"Docker restart")
	@$(COMPOSE) restart

reset:
	@$(call banner,"Docker reset (DANGEROUS: down -v)")
	@$(COMPOSE) down -v --remove-orphans

# ============================================================
# Wait/Init (MinIO/ClickHouse)
# ============================================================

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

# ============================================================
# ClickHouse SQL (DDL / views / checks / reports)
# ============================================================

ch: env-ensure env-check ch-running-check
	@docker exec -i clickhouse clickhouse-client \
	  -u "$(CH_USER)" --password "$(CH_PASSWORD)" --database "$(CH_DB)" \
	  --query "$(Q)"

ch-file: env-ensure env-check ch-running-check
	@test -n "$(F)" || (echo "[ERROR] Provide F=<path_to_sql_file>"; exit 1)
	@docker exec -i clickhouse clickhouse-client \
	  -u "$(CH_USER)" --password "$(CH_PASSWORD)" --database "$(CH_DB)" \
	  --multiquery < "$(F)"

ch-show-schema:
	@$(call banner,"ClickHouse schema")
	@bash scripts/ch_run_sql.sh sql/00_ddl.sql
	@bash scripts/ch_run_sql.sh sql/04_alter_articles_add_nlp_extras.sql
	@bash scripts/ch_run_sql.sh sql/00_views.sql
	@bash scripts/ch_run_sql.sh sql/06_update_view_articles_dedup_score_prefers_nlp.sql
	@bash scripts/ch_run_sql.sh sql/09_actions_views.sql

clean-sql:
	@$(call banner,"SQL clean (DDL)")
	@bash scripts/ch_run_sql.sh sql/00_ddl.sql

views-bash:
	@$(call banner,"Views (bash)")
	@bash scripts/ch_run_sql.sh sql/00_views.sql
	@bash scripts/ch_run_sql.sh sql/06_update_view_articles_dedup_score_prefers_nlp.sql
	@bash scripts/ch_run_sql.sh sql/09_actions_views.sql
	@bash scripts/ch_run_sql.sh sql/10_hourly_compat_views.sql
	@bash scripts/ch_run_sql.sh sql/11_entities_events_views.sql

views: views-bash
hour: hour-bash

views-py: env-ensure env-check
hour-py: env-ensure env-check

refresh: views hour
	@echo "[INFO] refreshed: views + hourly aggregates"

health:
	@$(call banner,"Health checks")
	@bash scripts/ch_run_sql.sh sql/01_healthcheck.sql

quality:
	@$(call banner,"Content quality checks")
	@bash scripts/ch_run_sql.sh sql/02_content_quality.sql

stopwords-auto:
	@set -e; \
	IN="$(IN)"; \
	if [ -z "$$IN" ]; then IN="$(SILVER_GLOB)"; fi; \
	echo "[INFO] build auto-stopwords from: $$IN"; \
	$(PYTHON) scripts/build_stopwords_auto.py --input "$$IN" --out config/stopwords_auto_ru.txt

stopwords-auto-show:
	@echo "[INFO] head:"; sed -n '1,50p' config/stopwords_auto_ru.txt || true

topkw:
	@$(call banner,"Top keywords")
	@bash scripts/ch_run_sql.sh sql/03_top_keywords.sql

hour-bash:
	@$(call banner,"By hour (bash)")
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

dupes-json:
	@CH_FORMAT=JSONEachRow $(MAKE) dupes

report-json:
	@CH_FORMAT=JSONEachRow $(MAKE) report

topkw-json:
	@CH_FORMAT=JSONEachRow $(MAKE) topkw

hour-json:
	@CH_FORMAT=JSONEachRow $(MAKE) hour

# ============================================================
# Reports (Markdown)
# ============================================================

md-report:
	@$(call banner,"Markdown report")
	@$(PYTHON) -m src.reporting.generate_report

reports:
	@$(call banner,"Reports list")
	@ls -1t reports/*.md | head -n 20 || true

# ============================================================
# Collector / Validators
# ============================================================

# Запуск RSS сборщика -> raw (S3/MinIO или local — зависит от переменных)
run:
	@$(call banner,"Run RSS -> raw")
	@$(PYTHON) -m src.collectors.rss_collector

# ------------------------------------------------------------
# RSS pipeline (tg-style combined)
# ------------------------------------------------------------

rss-raw:
	@$(call banner,"RSS -> raw (tg-style combined, local)")
	@RAW_BACKEND=$(RSS_RAW_BACKEND) RAW_DIR=$(RSS_RAW_DIR) RSS_COMBINED=$(RSS_COMBINED) \
	  $(PYTHON) -m src.collectors.rss_collector

# Latest RSS raw (важно: печатает ТОЛЬКО путь)
rss-raw-latest:
	@set -e; \
	f="$$(ls -1t $(RSS_RAW_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
	test -n "$$f" || (echo "[ERROR] No RSS combined raw found: $(RSS_RAW_COMBINED_GLOB)"; exit 1); \
	echo "$$f"

rss-silver:
	@set -e; \
	RAW_IN="$${RAW_IN:-}"; \
	if [ -z "$$RAW_IN" ]; then RAW_IN="$$( $(MAKE) -s rss-raw-latest )"; fi; \
	$(PYTHON) -m src.pipeline.clean_raw_to_silver_local --input "$$RAW_IN"

rss-silver-latest:
	@set -e; \
	f="$$(ls -1t $(RSS_SILVER_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
	test -n "$$f" || (echo "[ERROR] No RSS combined silver found: $(RSS_SILVER_COMBINED_GLOB)"; exit 1); \
	echo "$$f"

rss-gold:
	@set -e; \
	SILVER_IN="$${SILVER_IN:-}"; \
	if [ -z "$$SILVER_IN" ]; then SILVER_IN="$$( $(MAKE) -s rss-silver-latest )"; fi; \
	echo "[INFO] silver -> gold: $$SILVER_IN"; \
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input "$$SILVER_IN"

rss-gold-latest:
	@set -e; \
	f="$$(ls -1t $(RSS_GOLD_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
	test -n "$$f" || (echo "[ERROR] No RSS combined gold found: $(RSS_GOLD_COMBINED_GLOB)"; exit 1); \
	echo "$$f"

rss-load: env-ensure env-check
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	GOLD_IN="$${GOLD_IN:-}"; \
	if [ -z "$$GOLD_IN" ]; then GOLD_IN="$$( $(MAKE) -s rss-gold-latest )"; fi; \
	echo "[INFO] gold -> ClickHouse: $$GOLD_IN"; \
	echo "[INFO] Using batch-id: $$BATCH_ID"; \
	$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input "$$GOLD_IN" --batch-id "$$BATCH_ID"

rss-etl:
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	echo "[INFO] RSS ETL (tg-style). BATCH_ID=$$BATCH_ID"; \
	BATCH_ID="$$BATCH_ID" $(MAKE) rss-raw; \
	BATCH_ID="$$BATCH_ID" $(MAKE) rss-silver; \
	BATCH_ID="$$BATCH_ID" $(MAKE) rss-gold; \
	BATCH_ID="$$BATCH_ID" $(MAKE) rss-load; \
	if [ "$(STRICT)" = "1" ] || [ "$(STRICT)" = "true" ]; then $(MAKE) gate; fi

# ------------------------------------------------------------
# Validators
# ------------------------------------------------------------

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

# ВАЖНО: gate должен быть устойчивым и не падать из-за "docker exec в stopped контейнер".
# Поэтому:
# 1) сначала проверяем, что контейнер clickhouse RUNNING
# 2) затем делаем /ping (host/port выбираются автоматически: хост vs контейнер)
# 3) и только потом запускаем SQL проверки
gate: ch-running-check ch-http-ping
	@$(call banner,"Quality gate")
	@bash scripts/ch_run_sql.sh sql/02_content_quality.sql

# ============================================================
# Pipeline (silver->gold, gold->ClickHouse, полный ETL)
# ============================================================

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

# ============================================================
# Telegram pipeline (collector -> raw -> silver -> gold -> ClickHouse)
# ============================================================

tg-raw:
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	echo "[INFO] Telegram raw ingest. BATCH_ID=$$BATCH_ID"; \
	echo "[INFO] TG args: $(TG_SCRAPER_ARGS)"; \
	BATCH_ID="$$BATCH_ID" $(PYTHON) -m src.collectors.telegram_scraper $(TG_SCRAPER_ARGS)

tg-raw-latest:
	@set -e; \
	if [ "$(TG_COMBINED)" = "1" ]; then \
		f="$$(ls -1t $(TG_RAW_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
		if [ -z "$$f" ]; then echo "[ERROR] No combined Telegram raw found: $(TG_RAW_COMBINED_GLOB)"; exit 1; fi; \
		echo "$$f"; \
	else \
		f="$$(ls -1t $(TG_RAW_CH_GLOB) 2>/dev/null | head -n 1)"; \
		if [ -z "$$f" ]; then \
			echo "[ERROR] No channel Telegram raw found for '$(TG_CHANNEL)': $(TG_RAW_CH_GLOB)"; \
			echo "[HINT] If you scrape multiple channels via channels-file, set TG_COMBINED=1 or use tg-raw-latest-any."; \
			exit 1; \
		fi; \
		echo "$$f"; \
	fi

# FIX: убран лишний символ ')' в блоке if (он ломал таргет)
tg-raw-latest-any:
	@set -e; \
	f="$$(ls -1t $(TG_RAW_ANY_GLOB) 2>/dev/null | head -n 1)"; \
	if [ -z "$$f" ]; then echo "[ERROR] No Telegram raw files found: $(TG_RAW_ANY_GLOB)"; exit 1; fi; \
	echo "$$f"

tg-raw-latest-chan:
	@set -e; \
	f="$$(ls -1t $(TG_RAW_CH_GLOB) 2>/dev/null | head -n 1)"; \
	if [ -z "$$f" ]; then echo "[ERROR] No channel Telegram raw found for '$(TG_CHANNEL)': $(TG_RAW_CH_GLOB)"; exit 1; fi; \
	echo "$$f"

tg-silver:
	@set -e; \
	RAW_IN="$${RAW_IN:-}"; \
	if [ -z "$$RAW_IN" ]; then \
		RAW_IN="$$( $(MAKE) -s tg-raw-latest )"; \
	fi; \
	test -n "$$RAW_IN" || (echo "[ERROR] No raw file found for Telegram. Run: make tg-raw"; exit 1); \
	echo "[INFO] raw -> silver: $$RAW_IN"; \
	$(PYTHON) -m src.pipeline.clean_raw_to_silver_local --input "$$RAW_IN"

tg-silver-latest:
	@set -e; \
	if [ "$(TG_COMBINED)" = "1" ]; then \
		f="$$(ls -1t $(TG_SILVER_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
		if [ -z "$$f" ]; then echo "[ERROR] No combined Telegram silver found: $(TG_SILVER_COMBINED_GLOB)"; exit 1; fi; \
		echo "$$f"; \
	else \
		f="$$(ls -1t $(TG_SILVER_CH_GLOB) 2>/dev/null | head -n 1)"; \
		if [ -z "$$f" ]; then echo "[ERROR] No channel Telegram silver found for '$(TG_CHANNEL)': $(TG_SILVER_CH_GLOB)"; exit 1; fi; \
		echo "$$f"; \
	fi

tg-gold:
	@set -e; \
	SILVER_IN="$${SILVER_IN:-}"; \
	if [ -z "$$SILVER_IN" ]; then \
		SILVER_IN="$$( $(MAKE) -s tg-silver-latest )"; \
	fi; \
	test -n "$$SILVER_IN" || (echo "[ERROR] No silver file found for Telegram. Run: make tg-silver"; exit 1); \
	echo "[INFO] silver -> gold: $$SILVER_IN"; \
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input "$$SILVER_IN"

tg-gold-latest:
	@set -e; \
	if [ "$(TG_COMBINED)" = "1" ]; then \
		f="$$(ls -1t $(TG_GOLD_COMBINED_GLOB) 2>/dev/null | head -n 1)"; \
		if [ -z "$$f" ]; then echo "[ERROR] No combined Telegram gold found: $(TG_GOLD_COMBINED_GLOB)"; exit 1; fi; \
		echo "$$f"; \
	else \
		f="$$(ls -1t $(TG_GOLD_CH_GLOB) 2>/dev/null | head -n 1)"; \
		if [ -z "$$f" ]; then echo "[ERROR] No channel Telegram gold found for '$(TG_CHANNEL)': $(TG_GOLD_CH_GLOB)"; exit 1; fi; \
		echo "$$f"; \
	fi

tg-load: env-ensure env-check
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	GOLD_IN="$${GOLD_IN:-}"; \
	if [ -z "$$GOLD_IN" ]; then \
		GOLD_IN="$$( $(MAKE) -s tg-gold-latest )"; \
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

# ============================================================
# Superset (безопасные команды + бэкап/восстановление мета-БД)
# ============================================================

COMPOSE_SUPERSET := $(COMPOSE_BIN) --project-directory $(PROJECT_DIR) --env-file $(ENV_FILE) \
  -f $(PROJECT_DIR)/docker-compose.superset.yml

SUP_BACKUP_DIR ?= backups/superset
FILE ?=

superset-up: env-ensure
	@$(call banner,"Superset up (safe)")
	@test -f docker-compose.superset.yml || (echo "[ERROR] docker-compose.superset.yml not found"; exit 1)
	@$(COMPOSE_SUPERSET) up -d superset-postgres superset-redis
	@$(COMPOSE_SUPERSET) up superset-init
	@$(COMPOSE_SUPERSET) up -d superset
	@$(COMPOSE_SUPERSET) ps

superset-down:
	@$(call banner,"Superset down (safe, keep volumes)")
	@$(COMPOSE_SUPERSET) down

superset-stop:
	@$(call banner,"Superset stop (keep containers/volumes)")
	@$(COMPOSE_SUPERSET) stop superset superset-init 2>/dev/null || true

superset-ps:
	@$(call banner,"Superset ps")
	@$(COMPOSE_SUPERSET) ps

superset-logs:
	@$(call banner,"Superset logs")
	@$(COMPOSE_SUPERSET) logs --tail=200 superset

superset-backup: env-ensure
	@$(call banner,"Superset backup (metadata DB)")
	@mkdir -p "$(SUP_BACKUP_DIR)"
	@set -e; \
	  TS="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	  OUT="$(SUP_BACKUP_DIR)/superset_meta_$${TS}.sql.gz"; \
	  echo "[INFO] Backup -> $$OUT"; \
	  $(COMPOSE_SUPERSET) up -d superset-postgres >/dev/null; \
	  $(COMPOSE_SUPERSET) exec -T superset-postgres bash -lc '\
	    PGUSER="$${POSTGRES_USER:-superset}"; \
	    PGDB="$${POSTGRES_DB:-superset}"; \
	    PGPASSWORD="$${POSTGRES_PASSWORD:-superset}"; \
	    export PGPASSWORD; \
	    pg_dump -U "$$PGUSER" -d "$$PGDB" --no-owner --no-privileges \
	  ' | gzip > "$$OUT"; \
	  echo "[OK] $$OUT"

superset-backup-list:
	@$(call banner,"Superset backups")
	@ls -1t "$(SUP_BACKUP_DIR)"/*.sql.gz 2>/dev/null || true

superset-restore: env-ensure
	@$(call banner,"Superset restore (metadata DB)")
	@test -n "$(FILE)" || (echo "[ERROR] Usage: make superset-restore FILE=backups/superset/<file>.sql.gz"; exit 1)
	@test -f "$(FILE)" || (echo "[ERROR] FILE not found: $(FILE)"; exit 1)
	@set -e; \
	  echo "[WARN] This will overwrite current Superset metadata DB."; \
	  $(COMPOSE_SUPERSET) stop superset superset-init >/dev/null 2>&1 || true; \
	  $(COMPOSE_SUPERSET) up -d superset-postgres >/dev/null; \
	  echo "[INFO] Drop & recreate public schema..."; \
	  $(COMPOSE_SUPERSET) exec -T superset-postgres bash -lc '\
	    PGUSER="$${POSTGRES_USER:-superset}"; \
	    PGDB="$${POSTGRES_DB:-superset}"; \
	    PGPASSWORD="$${POSTGRES_PASSWORD:-superset}"; \
	    export PGPASSWORD; \
	    psql -U "$$PGUSER" -d "$$PGDB" -v ON_ERROR_STOP=1 -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;" \
	  '; \
	  echo "[INFO] Restore from $(FILE) ..."; \
	  if echo "$(FILE)" | grep -q '\.gz$$'; then \
	    gunzip -c "$(FILE)" | $(COMPOSE_SUPERSET) exec -T superset-postgres bash -lc '\
	      PGUSER="$${POSTGRES_USER:-superset}"; \
	      PGDB="$${POSTGRES_DB:-superset}"; \
	      PGPASSWORD="$${POSTGRES_PASSWORD:-superset}"; \
	      export PGPASSWORD; \
	      psql -U "$$PGUSER" -d "$$PGDB" -v ON_ERROR_STOP=1 \
	    '; \
	  else \
	    cat "$(FILE)" | $(COMPOSE_SUPERSET) exec -T superset-postgres bash -lc '\
	      PGUSER="$${POSTGRES_USER:-superset}"; \
	      PGDB="$${POSTGRES_DB:-superset}"; \
	      PGPASSWORD="$${POSTGRES_PASSWORD:-superset}"; \
	      export PGPASSWORD; \
	      psql -U "$$PGUSER" -d "$$PGDB" -v ON_ERROR_STOP=1 \
	    '; \
	  fi; \
	  echo "[OK] Restored. Now run: make superset-up"

# ============================================================
# Smoke-test
# ============================================================

smoke: reset bootstrap
	$(PYTHON) -m src.utils.s3_smoke_test
	$(MAKE) etl-latest