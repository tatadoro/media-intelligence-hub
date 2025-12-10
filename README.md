# Media Intelligence Hub

Учебный пет-проект на стыке Data Analytics и Data Science.

Цель: построить автоматизированный пайплайн мониторинга медиа:
- сбор статей и материалов из разных источников (RSS, API, парсеры);
- сохранение данных в Data Lake (S3);
- очистка текстов и извлечение ключевой информации;
- суммаризация и базовая NLP-обработка;
- загрузка витрин в ClickHouse;
- визуализация в Superset и DataLens;
- формирование ежедневного отчета и отправка на почту.

Технологии (планируемые):
- Python, pandas
- S3 / MinIO (Data Lake)
- ClickHouse
- Apache Airflow
- Apache Superset
- Yandex DataLens

## Структура проекта
- `src/` — исходный код проекта (основная логика)
- `config/` — файлы конфигурации (пути к данным, параметры запуска и т.д.)
- `data/` — данные (сырые и обработанные, большие файлы в git не добавляем)
- `notebooks/` — Jupyter-ноутбуки для экспериментов и исследования данных
- `templates/` — шаблоны (репорты, SQL, Jinja2 и т.п.)
- `requirements.txt` — список Python-зависимостей проекта
- `venv/` — виртуальное окружение (не хранится в git)


## Формат сырых данных (data/raw)

Файлы `data/raw/articles_*.json` содержат список объектов (JSON-массив), каждый объект — одна статья/новость со следующими полями:

- `id` — идентификатор записи (если в RSS есть `id`, иначе используется ссылка `link`);
- `title` — заголовок статьи;
- `link` — ссылка на исходный материал;
- `published_at` — дата и время публикации (как отданы в RSS: `published` или `updated`);
- `source` — название источника (например, `lenta.ru`);
- `raw_text` — исходный текст/аннотация из RSS (`summary` или `description`).

### Хранилище данных (Data Lake на MinIO)

Сырые данные новостей (RSS) сохраняются в S3-совместимое хранилище MinIO.

- Бакет: `media-intel`
- Структура raw-зоны: `raw/<YYYY-MM-DD>/<source_name>/articles_<YYYYMMDD_HHMMSS>.json`
- Пример: `raw/2025-12-10/lenta.ru/articles_20251210_123456.json`

Параметры доступа к MinIO задаются в `config/settings.yaml` и `config/.env` (последний не хранится в Git).

### Templates
Папка `templates/` будет использоваться для шаблонов:
- отчетов (например, Jinja2 для HTML/Markdown),
- SQL-запросов,
- текстов/конфигов, которые переиспользуются.

### Установка и запуск

1. Клонировать репозиторий:

```bash
git clone https://github.com/<user>/media-intelligence-hub-.git
cd media-intelligence-hub-
```
2.	Создать и активировать виртуальное окружение (пример для Unix/macOS):
```bash
python3 -m venv venv
source venv/bin/activate
```
3.	Установить зависимости:
```bash
pip install -r requirements.txt
```
Запустить первый сбор данных из RSS:
```bash
python -m src.collectors.rss_collector
```
После успешного запуска в папке data/raw появится файл вида:
```bash
data/raw/articles_YYYYMMDD_HHMMSS.json
```

### Работа с проектом на нескольких устройствах

1. Перед началом работы на любом устройстве:
   - перейти в папку проекта;
   - активировать виртуальное окружение;
   - выполнить `git pull`.

2. После окончания работы:
   - проверить изменения через `git status`;
   - сделать `git add` и `git commit`;
   - выполнить `git push`.

Таким образом изменения синхронизируются через GitHub.