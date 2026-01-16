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
        tg-raw tg-raw-latest tg-silver tg-silver-latest tg-gold tg-gold-latest \
        tg-load tg-etl \
        superset-up superset-down superset-stop superset-ps superset-logs \
        superset-backup superset-backup-list superset-restore \
        etl-latest-rss etl-latest-tg etl-latest-all tg-etl-refresh

# ------------------------------------------------------------
# Настройки Python/путей по умолчанию
# ------------------------------------------------------------
PYTHON      ?= python
SILVER_GLOB ?= data/silver/articles_*_clean.json

# --- Source-specific latest silver globs ---
RSS_SILVER_GLOB ?= data/silver/articles_[0-9]*_clean.json
TG_SILVER_GLOB  ?= data/silver/articles_*telegram*_clean.json

etl-latest-rss:
	@latest="$$(ls -1t $(RSS_SILVER_GLOB) 2>/dev/null | grep -v telegram | head -n 1)"; \
	if [ -z "$$latest" ]; then \
	  echo "[ERROR] No RSS silver files found (non-telegram): $(RSS_SILVER_GLOB)"; \
	  exit 1; \
	fi; \
	echo "[INFO] Latest RSS silver: $$latest"
	@$(MAKE) etl-latest-strict SILVER_GLOB='$(RSS_SILVER_GLOB)'

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
# Важно: используем safe-порядок запуска Superset (init отдельно).
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
# ВАЖНО: избегаем рекурсивных определений вида CH_USER ?= $(or $(CH_USER),...)
# ------------------------------------------------------------
CH_DB ?=
CH_DB := $(or $(CH_DATABASE),$(CLICKHOUSE_DB),media_intel)

CH_USER ?=
CH_USER := $(or $(CH_USER),$(CLICKHOUSE_USER),admin)

CH_PASSWORD ?=
CH_PASSWORD := $(or $(CH_PASSWORD),$(CLICKHOUSE_PASSWORD),admin12345)

CH_HOST ?=
CH_HOST := $(or $(CH_HOST),localhost)

CH_PORT ?=
CH_PORT := $(or $(CH_PORT),18123)

CH_TZ ?=
CH_TZ := $(or $(CH_TZ),Europe/Belgrade)

# ------------------------------------------------------------
# MinIO (переменные с fallback’ами) — также без рекурсии
# ------------------------------------------------------------
MINIO_ENDPOINT ?=
MINIO_ENDPOINT := $(or $(MINIO_ENDPOINT),http://localhost:9000)

MINIO_ACCESS_KEY ?=
MINIO_ACCESS_KEY := $(or $(MINIO_ACCESS_KEY),admin)

MINIO_SECRET_KEY ?=
MINIO_SECRET_KEY := $(or $(MINIO_SECRET_KEY),admin12345)

MINIO_BUCKET ?=
MINIO_BUCKET := $(or $(MINIO_BUCKET),mih)

# ============================================================
# Helpers
# ============================================================

# Красивый баннер для логов make-таргетов
define banner
	@echo "============================================================"
	@echo $(1)
	@echo "============================================================"
endef

# ============================================================
# Env
# ============================================================

# Проверка ключевых переменных окружения (для диагностики)
env-check:
	@$(call banner,"Env check")
	@echo "CH_USER=$(CH_USER)"
	@echo "CH_PASSWORD=$$($(PYTHON) - <<'PY'\nimport os\np=os.getenv('CH_PASSWORD') or os.getenv('CLICKHOUSE_PASSWORD') or ''\nprint(p[:2]+'***'+p[-2:]+' (len='+str(len(p))+')' if p else '(empty)')\nPY)"
	@echo "CH_DATABASE=$(CH_DB)"
	@echo "CH_HOST=$(CH_HOST)"
	@echo "CH_PORT=$(CH_PORT)"
	@echo "CLICKHOUSE_USER=$(or $(CLICKHOUSE_USER),$(CH_USER))"
	@echo "CLICKHOUSE_PASSWORD=$$($(PYTHON) - <<'PY'\nimport os\np=os.getenv('CLICKHOUSE_PASSWORD') or os.getenv('CH_PASSWORD') or ''\nprint(p[:2]+'***'+p[-2:]+' (len='+str(len(p))+')' if p else '(empty)')\nPY)"

# Гарантия, что .env существует (иначе объясняем, что делать)
env-ensure:
	@$(call banner,"Ensure env")
	@test -f .env || (echo "[ERROR] .env not found. Create it from .env.example"; exit 1)

# ============================================================
# Docker (основной стек проекта: ClickHouse/MinIO/…)
# ============================================================

# Поднять инфраструктуру (core + Superset) в фоне, тихо (без orphan warning)
up: env-ensure
	@$(call banner,"Docker up (core + Superset + Airflow)")
	@test -f docker-compose.superset.yml || (echo "[ERROR] docker-compose.superset.yml not found"; exit 1)
	@test -f docker-compose.airflow.yml || (echo "[ERROR] docker-compose.airflow.yml not found"; exit 1)

	# Базовые сервисы
	@$(COMPOSE_FULL) up -d clickhouse minio superset-postgres superset-redis airflow-postgres

	# One-off init контейнеры (должны завершиться успешно)
	@$(COMPOSE_FULL) up superset-init
	@$(COMPOSE_FULL) up airflow-init

	# Основные сервисы
	@$(COMPOSE_FULL) up -d superset airflow-webserver airflow-scheduler
	@$(COMPOSE_FULL) ps

# Остановить весь стек (core + Superset), volumes НЕ удаляем
down:
	@$(call banner,"Docker down (core + Superset + Airflow)")
	@$(COMPOSE_FULL) down

ps:
	@$(call banner,"Docker ps (core + Superset + Airflow)")
	@$(COMPOSE_FULL) ps

# Логи основного стека
logs:
	@$(call banner,"Docker logs")
	@$(COMPOSE) logs -f --tail 200

# Перезапуск контейнеров основного стека
restart:
	@$(call banner,"Docker restart")
	@$(COMPOSE) restart

# Сброс основного стека (ОПАСНО): удаляет volumes, данные пропадут
reset:
	@$(call banner,"Docker reset (DANGEROUS: down -v)")
	@$(COMPOSE) down -v --remove-orphans

# ============================================================
# Wait/Init (MinIO/ClickHouse)
# ============================================================

# Дождаться готовности MinIO (healthcheck/ожидание порта)
wait-minio:
	@$(call banner,"Wait MinIO")
	@bash scripts/wait_minio.sh

# Дождаться готовности ClickHouse (healthcheck/ожидание порта)
wait-clickhouse:
	@$(call banner,"Wait ClickHouse")
	@bash scripts/wait_clickhouse.sh

# Создать bucket в MinIO (если ещё не создан)
create-bucket:
	@$(call banner,"Create bucket")
	@bash scripts/create_bucket.sh

# Полная инициализация инфраструктуры: up + ожидание + bucket
init:
	@$(call banner,"Init")
	@$(MAKE) env-ensure
	@$(MAKE) up
	@$(MAKE) wait-minio
	@$(MAKE) wait-clickhouse
	@$(MAKE) create-bucket

# Bootstrap: init + применение схемы/вьюх ClickHouse
bootstrap:
	@$(call banner,"Bootstrap")
	@$(MAKE) init
	@$(MAKE) ch-show-schema

# ============================================================
# ClickHouse SQL (DDL / views / checks / reports)
# ============================================================

# Применить базовую схему и вьюхи ClickHouse
ch-show-schema:
	@$(call banner,"ClickHouse schema")
	@bash scripts/ch_run_sql.sh sql/00_ddl.sql
	@bash scripts/ch_run_sql.sh sql/04_alter_articles_add_nlp_extras.sql
	@bash scripts/ch_run_sql.sh sql/00_views.sql
	@bash scripts/ch_run_sql.sh sql/06_update_view_articles_dedup_score_prefers_nlp.sql
	@bash scripts/ch_run_sql.sh sql/09_actions_views.sql

# Пересоздать DDL (часто используют как “очистить и создать заново”)
clean-sql:
	@$(call banner,"SQL clean (DDL)")
	@bash scripts/ch_run_sql.sh sql/00_ddl.sql

# Пересоздать вьюхи ClickHouse
views:
	@$(call banner,"Views")
	@bash scripts/ch_run_sql.sh sql/00_views.sql
	@bash scripts/ch_run_sql.sh sql/06_update_view_articles_dedup_score_prefers_nlp.sql
	@bash scripts/ch_run_sql.sh sql/09_actions_views.sql


# --- Витрины должны выполняться только при валидном окружении ---
views: env-ensure env-check
	@$(PYTHON) -m src.cli.ch_sql --file sql/00_views.sql
	@$(PYTHON) -m src.cli.ch_sql --file sql/06_update_view_articles_dedup_score_prefers_nlp.sql
	@$(PYTHON) -m src.cli.ch_sql --file sql/09_actions_views.sql

hour: env-ensure env-check
	@$(PYTHON) -m src.cli.ch_sql --file sql/04_by_hour.sql

.PHONY: refresh
refresh: views hour
	@echo "[INFO] refreshed: views + hourly aggregates"

# Healthchecks SQL
health:
	@$(call banner,"Health checks")
	@bash scripts/ch_run_sql.sh sql/01_healthcheck.sql

# Контентные проверки качества (quality gate)
quality:
	@$(call banner,"Content quality checks")
	@bash scripts/ch_run_sql.sh sql/02_content_quality.sql

# Топ ключевых слов
topkw:
	@$(call banner,"Top keywords")
	@bash scripts/ch_run_sql.sh sql/03_top_keywords.sql

# Агрегация по часам
hour:
	@$(call banner,"By hour")
	@bash scripts/ch_run_sql.sh sql/04_by_hour.sql

# Отчёт по batch-ам
batches:
	@$(call banner,"Batches")
	@bash scripts/ch_run_sql.sh sql/05_batches.sql

# “Выживаемость” (retention-like) отчёт
survival:
	@$(call banner,"Survival")
	@bash scripts/ch_run_sql.sh sql/06_survival.sql

# Дубликаты
dupes:
	@$(call banner,"Dupes")
	@bash scripts/ch_run_sql.sh sql/07_dupes.sql

# Сводный SQL-отчёт
report:
	@$(call banner,"Report")
	@bash scripts/ch_run_sql.sh sql/08_report.sql

# ============================================================
# Reports (Markdown)
# ============================================================

# Сгенерировать Markdown-отчёт (Python модуль)
md-report:
	@$(call banner,"Markdown report")
	@$(PYTHON) -m src.reporting.generate_report

# Показать последние отчёты (20 штук)
reports:
	@$(call banner,"Reports list")
	@ls -1t reports/*.md | head -n 20 || true

# ============================================================
# Collector / Validators
# ============================================================

# Запуск RSS сборщика -> raw
run:
	@$(call banner,"Run RSS -> raw")
	@$(PYTHON) -m src.collectors.rss_collector

# Проверка silver-файла по контракту (IN обязателен)
validate-silver:
	@$(call banner,"Validate silver")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_silver_json>"; exit 1)
	@$(PYTHON) scripts/validate_silver.py --input "$(IN)"

# Проверка gold-файла по контракту (IN обязателен)
validate-gold:
	@$(call banner,"Validate gold")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_gold_parquet>"; exit 1)
	@$(PYTHON) scripts/validate_gold.py --input "$(IN)"

# Проверка “последнего” silver-файла по glob
validate:
	@$(call banner,"Validate latest silver")
	@$(PYTHON) scripts/validate_silver.py --input "$$(ls -1t $(SILVER_GLOB) | head -n 1)"

# Запуск SQL-проверок качества как “гейт” (останавливает процесс при ошибках SQL)
gate:
	@$(call banner,"Quality gate")
	@bash scripts/ch_run_sql.sh sql/02_content_quality.sql

# ============================================================
# Pipeline (silver->gold, gold->ClickHouse, полный ETL)
# ============================================================

# Silver -> Gold (локально), IN обязателен
gold:
	@$(call banner,"Silver -> gold")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_silver_json>"; exit 1)
	@$(PYTHON) -m src.pipeline.silver_to_gold_local --input "$(IN)"

# Gold -> ClickHouse, IN и BATCH_ID обязательны
load:
	@$(call banner,"Gold -> ClickHouse")
	@test -n "$(IN)" || (echo "[ERROR] Provide IN=<path_to_gold_parquet>"; exit 1)
	@test -n "$(BATCH_ID)" || (echo "[ERROR] Provide BATCH_ID=<iso_timestamp_utc>"; exit 1)
	@$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input "$(IN)" --batch-id "$(BATCH_ID)"

# Полный ETL (raw->silver->gold->ClickHouse) “как есть”
etl:
	@$(call banner,"ETL (raw->silver->gold->clickhouse)")
	@$(PYTHON) -m src.pipeline.clean_raw_to_silver_local
	@$(PYTHON) -m src.pipeline.silver_to_gold_local
	@$(PYTHON) -m src.pipeline.gold_to_clickhouse_local

# Прогон ETL по “последним” данным (внутри логика выбора файла)
etl-latest:
	@$(call banner,"ETL latest")
	@$(PYTHON) -m src.pipeline.etl_latest

# Строгий режим для etl-latest (STRICT=1)
etl-latest-strict:
	@$(MAKE) etl-latest STRICT=1

# ============================================================
# Telegram pipeline (collector -> raw -> silver -> gold -> ClickHouse)
# ============================================================

# Сбор Telegram -> raw (выбираем каналы из файла/списка/одного канала)
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

# Показать “последний” raw-файл Telegram (устойчиво)
tg-raw-latest:
	@set -e; \
	ls -1t $(TG_RAW_ANY_GLOB) 2>/dev/null | head -n 1

# Raw -> Silver для Telegram (по RAW_IN или берём latest; устойчиво)
tg-silver:
	@set -e; \
	RAW_IN="$(RAW_IN)"; \
	if [ -z "$$RAW_IN" ]; then \
		RAW_IN="$$(ls -1t $(TG_RAW_ANY_GLOB) 2>/dev/null | head -n 1)"; \
	fi; \
	test -n "$$RAW_IN" || (echo "[ERROR] No raw file found for Telegram in $(TG_OUT_DIR). Run: make tg-raw"; exit 1); \
	echo "[INFO] raw -> silver: $$RAW_IN"; \
	$(PYTHON) -m src.pipeline.clean_raw_to_silver_local --input "$$RAW_IN"

# Показать “последний” silver-файл Telegram (устойчиво)
tg-silver-latest:
	@set -e; \
	ls -1t $(TG_SILVER_ANY_GLOB) 2>/dev/null | head -n 1

# Silver -> Gold для Telegram (по SILVER_IN или берём latest; устойчиво)
tg-gold:
	@set -e; \
	SILVER_IN="$(SILVER_IN)"; \
	if [ -z "$$SILVER_IN" ]; then \
		SILVER_IN="$$(ls -1t $(TG_SILVER_ANY_GLOB) 2>/dev/null | head -n 1)"; \
	fi; \
	test -n "$$SILVER_IN" || (echo "[ERROR] No silver file found for Telegram. Run: make tg-silver"; exit 1); \
	echo "[INFO] silver -> gold: $$SILVER_IN"; \
	$(PYTHON) -m src.pipeline.silver_to_gold_local --input "$$SILVER_IN"

# Показать “последний” gold-файл Telegram (устойчиво)
tg-gold-latest:
	@set -e; \
	ls -1t $(TG_GOLD_ANY_GLOB) 2>/dev/null | head -n 1

# Gold -> ClickHouse для Telegram (по GOLD_IN или берём latest; batch-id генерируем; устойчиво)
tg-load: env-ensure env-check
	@set -e; \
	BATCH_ID="$${BATCH_ID:-$$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; \
	GOLD_IN="$(GOLD_IN)"; \
	if [ -z "$$GOLD_IN" ]; then \
		GOLD_IN="$$(ls -1t $(TG_GOLD_ANY_GLOB) 2>/dev/null | head -n 1)"; \
	fi; \
	test -n "$$GOLD_IN" || (echo "[ERROR] No gold parquet found for Telegram. Run: make tg-gold"; exit 1); \
	echo "[INFO] gold -> ClickHouse: $$GOLD_IN"; \
	echo "[INFO] Using batch-id: $$BATCH_ID"; \
	$(PYTHON) -m src.pipeline.gold_to_clickhouse_local --input "$$GOLD_IN" --batch-id "$$BATCH_ID"

# Полный Telegram ETL одной командой (все этапы под одним BATCH_ID)
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

# Отдельный compose-файл для Superset.
# Важно: здесь НЕТ down -v, чтобы случайно не стереть мета-БД (дашборды/чарты/DB connections).
COMPOSE_SUPERSET := $(COMPOSE_BIN) --project-directory $(PROJECT_DIR) --env-file $(ENV_FILE) \
  -f $(PROJECT_DIR)/docker-compose.superset.yml

# Каталог для бэкапов мета-БД Superset
SUP_BACKUP_DIR ?= backups/superset

# Путь к файлу для восстановления (передаётся как FILE=...)
FILE ?=

# Поднять Superset безопасно:
# 1) postgres/redis (фон)
# 2) init (one-off, создаёт структуру/админа)
# 3) webserver superset (фон)
superset-up: env-ensure
	@$(call banner,"Superset up (safe)")
	@test -f docker-compose.superset.yml || (echo "[ERROR] docker-compose.superset.yml not found"; exit 1)
	@$(COMPOSE_SUPERSET) up -d superset-postgres superset-redis
	@$(COMPOSE_SUPERSET) up superset-init
	@$(COMPOSE_SUPERSET) up -d superset
	@$(COMPOSE_SUPERSET) ps

# Остановить Superset-стек безопасно (volumes сохраняем)
superset-down:
	@$(call banner,"Superset down (safe, keep volumes)")
	@$(COMPOSE_SUPERSET) down

# Остановить только superset-сервисы (без удаления контейнеров/volumes)
superset-stop:
	@$(call banner,"Superset stop (keep containers/volumes)")
	@$(COMPOSE_SUPERSET) stop superset superset-init 2>/dev/null || true

# Статус контейнеров Superset-стека
superset-ps:
	@$(call banner,"Superset ps")
	@$(COMPOSE_SUPERSET) ps

# Логи веб-сервиса Superset
superset-logs:
	@$(call banner,"Superset logs")
	@$(COMPOSE_SUPERSET) logs --tail=200 superset

# Бэкап мета-БД Superset (Postgres внутри superset-postgres).
# Сохраняем в backups/superset/superset_meta_<UTC>.sql.gz
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

# Показать список доступных бэкапов (самые новые сверху)
superset-backup-list:
	@$(call banner,"Superset backups")
	@ls -1t "$(SUP_BACKUP_DIR)"/*.sql.gz 2>/dev/null || true

# Восстановление мета-БД Superset из бэкапа.
# ВНИМАНИЕ: это перезапишет текущую мета-БД (дашборды/чарты/DB connections).
# Использование:
#   make superset-restore FILE=backups/superset/superset_meta_YYYYmmddTHHMMSSZ.sql.gz
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

# Smoke: сброс (опасно) + bootstrap + проверка MinIO + etl-latest
# ВНИМАНИЕ: reset удаляет volumes основного стека, используйте осознанно.
smoke: reset bootstrap
	$(PYTHON) -m src.utils.s3_smoke_test
	$(MAKE) etl-latest