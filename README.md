# Media Intelligence Hub

Учебный пет-проект на стыке Data Analytics и Data Science.

Цель: построить автоматизированный пайплайн мониторинга медиа:
- сбор материалов из разных источников (RSS / API / парсеры);
- хранение данных в Data Lake (S3/MinIO) и локально;
- очистка текста и извлечение ключевой информации;
- семантическое обогащение (summary, keywords, фичи);
- загрузка витрин в ClickHouse;
- аналитика/дашборды и отчёты (Markdown сейчас, BI — в процессе).

Статус: **MVP работает** (raw → silver → gold → ClickHouse + SQL-проверки + Airflow DAG + Markdown-отчёт + Superset).

## Архитектура пайплайна (MVP)

- **collectors**: сбор сырья из источников (RSS, Telegram)
- **raw**: сырые выгрузки (JSON)
- **silver**: очищенный текст (нормализация/чистка/контракт)
- **gold**: обогащение (summary + keywords + текстовые фичи), контракт (parquet)
- **ClickHouse**:
  - `media_intel.articles` — основная витрина (загрузка батчами)
  - `media_intel.articles_dedup` — слой/представление для аналитики без дублей (используем в BI)
  - `media_intel.load_log` — журнал загрузок (защита от повторных заливок)
- **reports**: SQL-аналитика + Markdown-дайджест

## Источники данных

### RSS
Сбор из RSS-лент в `raw`, опционально с шагом enrich (подтянуть body, если нужно).

### Telegram (t.me/s/<channel>)
Коллектор: `src/collectors/telegram_scraper.py`

Поддерживает:
- список каналов (`--channels-file` или `--channels`)
- сбор N постов на канал (`--target`)
- `--combined` (дополнительно сохраняет общий raw по всем каналам)
- сохранение в формате raw, совместимом с дальнейшим шагом `raw → silver`

Файл со списком каналов:
- `config/telegram_channels.txt` (по одному каналу на строку, можно с `#` комментариями)

## Автоматизация через Makefile

Ключевые команды:
- `make tg-raw` — Telegram → raw
- `make tg-silver` — raw → silver (берёт самый свежий Telegram raw)
- `make tg-gold` — silver → gold
- `make tg-load` — gold → ClickHouse
- `make tg-etl` — полный цикл Telegram: raw → silver → gold → ClickHouse

Примечание: для ClickHouse-части убедись, что переменные `CH_*` подхвачены из `.env`:
```bash
set -a; source .env; set +a
```

## Airflow orchestration

Основной DAG: `mih_etl_latest_strict` (ручной запуск).

Что делает:
1) выбирает **самый свежий** silver-файл по паттерну `data/silver/articles_*.json`
2) `precheck_silver`: быстрый sanity-check, чтобы не подхватить legacy-файл
3) `validate_silver`: валидация по контракту silver
4) `silver_to_gold`: преобразование silver → gold parquet
5) `compute_gold_path`: вычисляет путь до gold-файла
6) `validate_gold`: валидация по контракту gold
7) `quality_gate`: quality gate (STRICT=1)
8) `load_to_clickhouse`: загрузка gold → ClickHouse (с защитой от дублей)
9) `sql_reports`: прогон SQL-отчётов (`make report`)
10) `compute_report_window`: строит окно отчёта по **реальным данным в ClickHouse**: `[max(published_at) - last_hours, max(published_at)]`
11) `md_report`: генерирует Markdown-отчёт

Запуск DAG:
```bash
docker exec -it airflow-scheduler airflow dags trigger mih_etl_latest_strict
```

Проверка статусов по последнему run_id (через папку логов):
```bash
RUN_ID=$(ls -1t airflow/logs/dag_id=mih_etl_latest_strict | head -n 1 | sed 's/^run_id=//')
docker exec -it airflow-scheduler airflow tasks states-for-dag-run mih_etl_latest_strict "$RUN_ID"
```

### Доступ Airflow к Docker (если используешь `docker exec` внутри задач)

Чтобы шаги, которые делают `docker exec ...` (например, вызов `clickhouse-client`), работали **изнутри Airflow-контейнера**, в `docker-compose.airflow.yml` для `airflow-scheduler` (и при необходимости `airflow-webserver`) должен быть примонтирован Docker socket:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

Также проект монтируется в контейнер, чтобы DAG мог вызывать `make` и скрипты проекта:

```yaml
volumes:
  - ./:/opt/mih
```

### Артефакты отчётов

- Runtime Markdown-отчёты: `reports/daily_report_*.md` (в `.gitignore`)
- Пример отчёта для репозитория: `reports/examples/daily_report_sample.md` (коммитится)

Обновить пример отчёта:
```bash
cp "$(ls -1t reports/daily_report_*.md | head -n 1)" reports/examples/daily_report_sample.md
git add reports/examples/daily_report_sample.md
git commit -m "docs: update report sample"
git push
```

## Конфигурация и переменные окружения

Секреты не коммитятся. Локально создай файл `.env` (он должен быть в `.gitignore`).
В репозитории хранится только шаблон `.env.example` (если добавишь).

### MinIO / S3 (пример)
```bash
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=YOUR_MINIO_USER
MINIO_SECRET_KEY=YOUR_MINIO_PASSWORD
MINIO_BUCKET=media-intel
MINIO_REGION=us-east-1
MINIO_SECURE=false
```

### ClickHouse (пример)

Важно: SQL-раннер `scripts/ch_run_sql.sh` читает переменные `CH_*`.

```bash
CH_HOST=clickhouse
CH_PORT=8123
CH_USER=YOUR_CH_USER
CH_PASSWORD=YOUR_CH_PASSWORD
CH_DATABASE=media_intel
```

Примечания:
- внутри Docker-сети обычно `CH_HOST=clickhouse`, `CH_PORT=8123`
- с хоста ClickHouse HTTP может быть проброшен на `http://localhost:18123`

Генератор Markdown-отчёта (`src.reporting.generate_report`) читает `CH_*` и `CLICKHOUSE_*` переменные (для совместимости).

## Контракты данных (silver/gold)

В репозитории зафиксированы контракты слоёв (JSON Schema):
- `contracts/silver.schema.json` — формат **silver** (`*_clean.json`)
- `contracts/gold.schema.json` — формат **gold** (`*_processed.parquet`)

Скрипты валидации:
```bash
python scripts/validate_silver.py --input data/silver/<file>_clean.json
python scripts/validate_gold.py --input data/gold/<file>_processed.parquet
```

## Идемпотентная загрузка в ClickHouse

Загрузка выполняется батчами. В таблице `articles` есть:
- `ingest_object_name` — имя объекта/артефакта загрузки (используется для защиты от повторных заливок)
- `batch_id` — идентификатор запуска/батча (часто UTC timestamp, прокидывается через `BATCH_ID`)

Для аналитики без дублей используется `articles_dedup` (рекомендуется подключать именно её в Superset).

## Запуск пайплайна (локально без Airflow)

### 1) RSS → raw
```bash
python -m src.collectors.rss_collector
```

### 2) raw → silver (очистка)
```bash
python -m src.pipeline.clean_raw_to_silver_local --input data/raw/<raw_file>.json
```

### 3) silver → gold (summary + keywords)
```bash
python -m src.pipeline.silver_to_gold_local --input data/silver/<silver_file>.json
```

### 4) gold → ClickHouse
```bash
python -m src.pipeline.gold_to_clickhouse_local --input data/gold/<gold_file>.parquet
```

### Telegram ETL через Makefile
```bash
make tg-etl
```

Перед первой загрузкой и SQL-проверками подготовь ClickHouse-объекты:
```bash
make init
```

## SQL-отчёты и проверки (ClickHouse)

SQL-файлы в `sql/` предназначены для контроля качества и аналитики поверх загруженных данных.

Запуск набора отчётов:
```bash
make report
```

## Superset (дашборды)

Для BI-аналитики подключаем ClickHouse и строим чарты/дашборды поверх:
- `media_intel.articles_dedup` (рекомендуется для большинства графиков)
- агрегатных датасетов из SQL Lab (saved datasets)

## Безопасность и артефакты
- `.env` и любые секреты **не коммитятся**
- `data/` и локальные артефакты **не коммитятся**
- runtime отчёты `reports/daily_report_*.md` **не коммитятся**
- в `reports/` в git хранится только `reports/examples/`

## Troubleshooting

### ClickHouse не готов / таймаут
```bash
docker compose ps
docker compose logs -f --tail=200 clickhouse
curl -sSf http://localhost:18123/ping
```

### MinIO не готов / таймаут
```bash
docker compose logs -f --tail=200 minio
curl -sSf http://localhost:9000/minio/health/ready
```

### `mc not found. Skip bucket creation.`
Это не блокирует MVP: raw/silver/gold/ClickHouse работают локально.
Если хочешь автосоздание бакета — установи MinIO Client:
```bash
brew install minio/stable/mc
```

### Переменные окружения не подхватываются
Проверь, что `.env` лежит в корне репозитория:
```bash
ls -la .env
```

## Roadmap
- обогащение gold: эмбеддинги, тематическое моделирование, тональность
- улучшение извлечения ключевых фраз и суммаризации
- витрины ClickHouse под BI (Superset/DataLens)
- расписания и уведомления (ежедневные дайджесты)
- расширение источников (несколько RSS/API/Telegram)
