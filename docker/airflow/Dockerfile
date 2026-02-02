FROM apache/airflow:2.9.3

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends make curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

USER airflow

# Кладем requirements отдельно, чтобы Docker кешировал слой
COPY requirements.txt /tmp/requirements.txt

# Ставим зависимости проекта в окружение Airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir pymorphy3 pymorphy3-dicts-ru