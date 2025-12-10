import os
import io
import yaml

import boto3
from dotenv import load_dotenv

# Путь к корню проекта: /Users/.../media-intelligence-hub-
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

ENV_PATH = os.path.join(BASE_DIR, "config", ".env")
SETTINGS_PATH = os.path.join(BASE_DIR, "config", "settings.yaml")

# 1. Подгружаем секреты из .env (логин/пароль к MinIO)
load_dotenv(ENV_PATH)

# 2. Читаем общие настройки из settings.yaml
with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)

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


def upload_json_bytes(bucket: str, key: str, json_str: str):
    """
    Загружаем JSON-строку в объект S3.

    bucket – имя бакета (например, MINIO_BUCKET),
    key – путь внутри бакета (raw/2025-12-10/source/articles_....json),
    json_str – JSON-строка (df.to_json(...)).
    """
    s3 = get_s3_client()
    bytes_io = io.BytesIO(json_str.encode("utf-8"))
    s3.upload_fileobj(bytes_io, bucket, key)


def list_bucket_objects(bucket: str, prefix: str = ""):
    """
    Возвращаем список объектов в бакете, начинающихся с prefix.
    Удобно для отладки: посмотреть, какие файлы лежат в raw/.
    """
    s3 = get_s3_client()
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return resp.get("Contents", [])