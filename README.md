# Media Intelligence Hub

Учебный пет-проект на стыке Data Analytics и Data Science.

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
- `sql/` — SQL для ClickHouse (проверки качества, аналитика)
- `scripts/` — вспомогательные скрипты
- `notebooks/` — EDA/эксперименты
- `data/` — локальные данные (не коммитятся)
- `docker-compose.yml` — локальная инфраструктура (MinIO/ClickHouse)
- `Makefile` — удобные команды для запуска типовых действий (если используется)
- `requirements.txt` — зависимости Python

## Требования
- Python 3.11+
- Docker (для MinIO/ClickHouse)

## Быстрый старт (локально)
```bash
git clone https://github.com/tatadoro/media-intelligence-hub-.git
cd media-intelligence-hub-

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# создать локальный env-файл из шаблона (если добавлен в репо)
cp .env.example .env

# поднять сервисы (MinIO/ClickHouse)
docker compose up -d
docker compose ps
```

### Проверка сервисов
- MinIO S3 endpoint: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`
- ClickHouse HTTP: `http://localhost:8123`

## Конфигурация и переменные окружения

Секреты не коммитятся. Локально создай файл `.env` (он должен быть в `.gitignore`).
В репозитории хранится только шаблон `.env.example`.

### Переменные окружения (пример)
```bash
# MinIO / S3
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=YOUR_MINIO_USER
MINIO_SECRET_KEY=YOUR_MINIO_PASSWORD
MINIO_BUCKET=media-intel
MINIO_REGION=us-east-1
MINIO_SECURE=false

# ClickHouse
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=YOUR_CLICKHOUSE_PASSWORD
CLICKHOUSE_DATABASE=default
```

В `config/settings.yaml` управляется, куда писать raw-данные:
- `raw_backend: local | s3 | both`

## MinIO: бакет для проекта
Проект ожидает бакет `media-intel` (или значение из `MINIO_BUCKET` / `config/settings.yaml`).

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

Примечание: если при первой загрузке в ClickHouse не хватает таблиц/вьюх, выполни подготовительные SQL-скрипты из `sql/` (в первую очередь `sql/00_views.sql`, если он используется в твоей схеме).

## SQL: проверки качества и аналитика (ClickHouse)

Файлы в `sql/` предназначены для контроля качества и аналитики поверх загруженных данных.

Рекомендуемый порядок:
1) `sql/00_views.sql` (если используются вьюхи/представления)
2) `sql/01_healthcheck.sql`
3) `sql/02_content_quality.sql`
4) `sql/03_top_keywords.sql`
5) дополнительные проверки: дедупликация и аналитика по батчам (файлы `03_*`, `05_*`, `06_*`, `07_*`)

Пример запуска через `clickhouse-client`:
```bash
clickhouse-client --query "$(cat sql/01_healthcheck.sql)"
```

## Быстрая проверка, что всё работает
После шага 4 (загрузка в ClickHouse) запусти:
- `sql/01_healthcheck.sql` — базовый healthcheck по данным
- `sql/02_content_quality.sql` — проверка качества контента
- `sql/03_top_keywords.sql` — топ ключевых слов

Если healthcheck не показывает данные, сначала проверь:
1) что gold-файл действительно создан в `data/gold/`
2) что загрузка в ClickHouse отработала без ошибок
3) что подключение к ClickHouse соответствует `.env`

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
- улучшение извлечения ключевых фраз и саммаризации
- стабильные витрины ClickHouse под BI (Superset/DataLens)
- ежедневные отчёты/уведомления (дайджесты)
- оркестрация (Airflow) и расписания