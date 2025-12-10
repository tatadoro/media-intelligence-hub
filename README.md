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

### Настройка S3 / MinIO (Data Lake)

Проект использует S3-совместимое хранилище (MinIO) для хранения сырых данных (raw layer).  
Все реальные ключи и пароли хранятся только локально в `.env` и не попадают в репозиторий.

#### 1. Запуск MinIO в Docker

Установите Docker и запустите MinIO, указав свои логин и пароль:

```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=<YOUR_MINIO_USER>" \
  -e "MINIO_ROOT_PASSWORD=<YOUR_MINIO_PASSWORD>" \
  minio/minio server /data --console-address ":9001"
```
	•	S3-эндпоинт: http://localhost:9000
	•	Веб-консоль MinIO: http://localhost:9001

Логин и пароль задаются только на вашей машине и не должны попадать в Git.

2. Создание бакета для проекта
Проект ожидает бакет с именем media-intel.

Через веб-консоль MinIO:
	1.	Откройте http://localhost:9001
	2.	Авторизуйтесь с вашим логином/паролем
	3.	Создайте бакет media-intel

Либо через MinIO Client (mc):

```bash
# добавить алиас для локального MinIO
mc alias set local http://localhost:9000 <YOUR_MINIO_USER> <YOUR_MINIO_PASSWORD>

# создать бакет
mc mb local/media-intel
```
3. Локальный .env с настройками S3
Для подключения к MinIO проект использует файл окружения (например, config/.env), который добавлен в .gitignore и не коммитится.

Пример содержимого (со своими значениями):
```bash
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=<YOUR_MINIO_USER>
MINIO_SECRET_KEY=<YOUR_MINIO_PASSWORD>
MINIO_BUCKET=media-intel
MINIO_REGION=us-east-1
MINIO_SECURE=false
```

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