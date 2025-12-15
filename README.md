# Media Intelligence Hub

Пет-проект на стыке Data Analytics и Data Science.

Цель: построить автоматизированный пайплайн мониторинга медиа:
- сбор материалов из разных источников (RSS / API / парсеры);
- хранение данных в Data Lake (S3/MinIO) и локально;
- очистка текста и извлечение ключевой информации;
- семантическое обогащение (summary, keywords, фичи);
- загрузка витрин в ClickHouse;
- аналитика/дашборды (Superset, DataLens) и ежедневные отчёты (в планах).

Статус: **MVP в разработке** (raw → silver → gold → ClickHouse + базовые SQL-проверки).

## Что уже реализовано (MVP)
- Ingestion RSS → raw (локально и/или MinIO)
- Raw → Silver: очистка текста
- Silver → Gold: summary + keywords (TF-IDF), текстовые фичи
- Gold → ClickHouse: загрузка витрин локально
- SQL-набор для healthcheck/quality/keywords/дедупликации и аналитики в `sql/`
- EDA-ноутбук по gold-слою: `notebooks/03_gold_eda.ipynb`

## Архитектура пайплайна
Слои данных: `raw → silver → gold → ClickHouse/BI`

- **raw**: данные “как пришли” (минимум преобразований)
- **silver**: очищенный текст (готов к NLP)
- **gold**: семантическое обогащение (summary, keywords, фичи)
- **витрины**: загрузка в ClickHouse + SQL-проверки/аналитика

### Где лежат данные
- локально: `data/` (в git **не добавляем**)
- в MinIO: `raw/YYYY-MM-DD/<source>/...`

### Форматы данных (по слоям)
- **raw**: `data/raw/*.json` (JSON-массив записей)
- **silver**: `data/silver/*_clean.json` (очищенный текст)
- **gold**: `data/gold/*_processed.parquet` (обогащённые данные)

## Структура репозитория
- `src/` — код (коллекторы, обработка, пайплайны)
- `config/` — конфигурация (настройки запуска, параметры)
- `sql/` — SQL для ClickHouse (DDL, проверки качества, аналитика)
- `scripts/` — вспомогательные скрипты
- `notebooks/` — EDA/эксперименты
- `data/` — локальные данные (не коммитятся)
- `docker-compose.yml` — локальная инфраструктура (MinIO/ClickHouse)
- `Makefile` — команды для локального запуска и проверок
- `requirements.txt` — зависимости Python

## Требования
- Python 3.11+
- Docker (для MinIO/ClickHouse)
- (опционально) MinIO Client `mc` — для авто-создания бакета в `make bootstrap`

## Быстрый старт (локально)
```bash
git clone https://github.com/tatadoro/media-intelligence-hub-.git
cd media-intelligence-hub-

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

## Быстрый старт через Makefile (рекомендуется)
Если у тебя установлен `make`, можно прогонять инфраструктуру + DDL + пайплайн одной командой.

```bash
# Поднять сервисы, дождаться готовности, накатить DDL + views, (опционально) создать bucket в MinIO
make bootstrap

# Полный прогон: silver -> gold -> ClickHouse -> SQL-отчёты
make etl IN=data/silver/<silver_file>_clean.json

# Только отчётные SQL (health/quality/keywords и т.д.)
make report
```

Примечание: при повторном запуске `make etl` загрузка в ClickHouse может показать
`[SKIP] ... уже загружен` — это ожидаемо (идемпотентность по `load_log` и факту строк).

## Быстрый старт (one-liner, запуск с нуля)
```bash
# Полный чистый старт: инфраструктура + DDL + views + (опционально) бакет MinIO
make reset && make bootstrap

# Прогон MVP-цепочки (silver -> gold -> ClickHouse -> SQL-отчёт)
make etl IN=data/silver/articles_20251210_155554_enriched_clean.json
```

### Что должно получиться после запуска
```bash
make health
make quality
make dupes
```

Ожидаемые признаки “всё ок”:
- `make health`: `rows > 0`, `bad_dates = 0`, `share_empty = 0`
- `make quality`: `share_has_body` близко к `1`
- `make dupes`: `duplicate_rows = 0` (как минимум внутри одного батча)

## Конфигурация и переменные окружения
Секреты не коммитятся. Локально создай файл `.env` (он должен быть в `.gitignore`).
В репозитории хранится только шаблон `.env.example`.

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
CH_CONTAINER=clickhouse
CH_USER=YOUR_CH_USER
CH_PASSWORD=YOUR_CH_PASSWORD
CH_DATABASE=media_intel
```

В `config/settings.yaml` управляется, куда писать raw-данные:
- `raw_backend: local | s3 | both`

## Проверка сервисов (если нужно руками)
- MinIO S3 endpoint: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`
- ClickHouse HTTP: `http://localhost:8123`

Проверить контейнеры:
```bash
docker compose ps
```

## MinIO: бакет для проекта
Проект ожидает бакет `media-intel` (или значение из `MINIO_BUCKET` / `config/settings.yaml`).

- `make bootstrap` вызывает `make create-bucket` **только если установлен `mc` и заполнены переменные MinIO в `.env`**.
- Если `mc` не установлен — это не блокирует запуск MVP (raw/silver/gold/ClickHouse работают локально).

Способ 1: через MinIO Console
1) открой `http://localhost:9001`
2) зайди под своими кредами
3) создай бакет `media-intel`

Способ 2: через MinIO Client (если установлен `mc`)
```bash
mc alias set local http://localhost:9000 YOUR_MINIO_USER YOUR_MINIO_PASSWORD
mc mb local/media-intel
```

## Запуск пайплайна

### 1 Сбор RSS → raw
```bash
python -m src.collectors.rss_collector
```

### 2 raw → silver (очистка)
```bash
python -m src.pipeline.clean_raw_to_silver_local --input data/raw/<raw_file>.json
```

### 3 silver → gold (summary + keywords)
```bash
python -m src.pipeline.silver_to_gold_local --input data/silver/<silver_file>.json
```

### 4 gold → ClickHouse
```bash
python -m src.pipeline.gold_to_clickhouse_local --input data/gold/<gold_file>.parquet
```

Перед первой загрузкой и SQL-проверками подготовь ClickHouse-объекты:
```bash
make init
```

## SQL: проверки качества и аналитика (ClickHouse)
Файлы в `sql/` предназначены для контроля качества и аналитики поверх загруженных данных.

Рекомендуемый порядок:
1) `sql/00_ddl.sql` (создание базы/таблиц)
2) `sql/00_views.sql` (вьюхи/представления)
3) `sql/01_healthcheck.sql`
4) `sql/02_content_quality.sql`
5) `sql/03_top_keywords.sql`
6) батчевые проверки и аналитика: `05_*`, `06_*`, `07_*`

Запуск через Makefile:
```bash
make report
```

## Примеры аналитики
- `notebooks/03_gold_eda.ipynb` — EDA gold-слоя (keywords, динамика по дням, базовые проверки)

Запуск:
```bash
source venv/bin/activate
jupyter notebook
```

## Безопасность и артефакты
- `.env` и любые секреты **не коммитятся**
- `data/` и локальные артефакты **не коммитятся**

## Roadmap
- обогащение gold: эмбеддинги, тематическое моделирование, тональность
- улучшение извлечения ключевых фраз и суммаризации
- стабильные витрины ClickHouse под BI (Superset/DataLens)
- ежедневные отчёты/уведомления (дайджесты)
- оркестрация (Airflow) и расписания