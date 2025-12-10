import json
import logging
import os
import io
import yaml

import boto3
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Путь к корню проекта: /Users/.../media-intelligence-hub-
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

ENV_PATH = os.path.join(BASE_DIR, "config", ".env")
SETTINGS_PATH = os.path.join(BASE_DIR, "config", "settings.yaml")

# 1. Подгружаем секреты из .env (логин/пароль к MinIO)
load_dotenv(ENV_PATH)

# 2. Читаем общие настройки из settings.yaml
with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)

# 3. Инициализируем S3/MinIO-клиент на основе переменных окружения
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "media-intel")

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name=MINIO_REGION,
)

storage_cfg = settings["storage"]["minio"]

# Достаём параметры MinIO из конфига
MINIO_ENDPOINT = storage_cfg["endpoint"]
MINIO_BUCKET = storage_cfg["bucket"]
MINIO_REGION = storage_cfg.get("region", "us-east-1")


def get_s3_client():
    """Создаём и возвращаем клиент S3 (MinIO) с параметрами из .env и settings.yaml."""
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=MINIO_REGION,
    )
    return s3



def upload_json_bytes(bucket: str, key: str, json_str: str) -> None:
    """
    Загружает JSON-строку в S3/MinIO по указанному ключу.

    :param bucket: имя бакета
    :param key: ключ (путь внутри бакета)
    :param json_str: строка с JSON (уже сериализованный dict)
    """
    bytes_io = io.BytesIO(json_str.encode("utf-8"))

    logger.info("Uploading object to S3: bucket=%s, key=%s", bucket, key)

    try:
        s3.upload_fileobj(bytes_io, bucket, key)
    except Exception as e:
        logger.error(
            "Failed to upload object to S3: bucket=%s, key=%s, error=%s",
            bucket,
            key,
            e,
        )
        raise
    else:
        logger.info(
            "Successfully uploaded object to S3: bucket=%s, key=%s",
            bucket,
            key,
        )

def list_bucket_objects(bucket: str, prefix: str = ""):
    """
    Возвращаем список объектов в бакете, начинающихся с prefix.
    Удобно для отладки: посмотреть, какие файлы лежат в raw/.
    """
    s3 = get_s3_client()
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return resp.get("Contents", [])
