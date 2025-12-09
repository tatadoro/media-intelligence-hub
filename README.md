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