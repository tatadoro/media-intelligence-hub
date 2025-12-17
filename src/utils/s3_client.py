import json
import logging
import os
import io
import yaml
import re
import boto3
from botocore.config import Config
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Путь к корню проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

ROOT_ENV_PATH = os.path.join(BASE_DIR, ".env")
LEGACY_ENV_PATH = os.path.join(BASE_DIR, "config", ".env")  # на всякий случай
SETTINGS_PATH = os.path.join(BASE_DIR, "config", "settings.yaml")

# 1) Подгружаем .env (сначала актуальный корневой, потом legacy без override)
load_dotenv(ROOT_ENV_PATH)
load_dotenv(LEGACY_ENV_PATH, override=False)

# 2) Читаем settings.yaml (если нет файла — не падаем)
try:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
except FileNotFoundError:
    settings = {}

storage_cfg = (settings.get("storage", {}) or {}).get("minio", {}) or {}

_PLACEHOLDER_RE = re.compile(r'^\$\{([A-Z0-9_]+):-"?(.*?)"?\}$')


def _resolve(value: str) -> str:
    """
    Поддержка шаблонов вида ${VAR:-"default"} из settings.yaml.
    Если формат другой — возвращаем как есть.
    """
    if not isinstance(value, str):
        return value
    m = _PLACEHOLDER_RE.match(value.strip())
    if not m:
        return value
    var, default = m.group(1), m.group(2)
    return os.getenv(var, default)


# 3) Итоговые параметры MinIO: env имеет приоритет, иначе берём из settings.yaml (с подстановкой)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT") or _resolve(storage_cfg.get("endpoint", "http://localhost:9000"))
MINIO_BUCKET = os.getenv("MINIO_BUCKET") or _resolve(storage_cfg.get("bucket", "media-intel"))
MINIO_REGION = os.getenv("MINIO_REGION") or _resolve(storage_cfg.get("region", "us-east-1"))

# Достаём параметры MinIO из конфига
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY") or _resolve(storage_cfg.get("access_key", ""))
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY") or _resolve(storage_cfg.get("secret_key", ""))
MINIO_SECURE = (os.getenv("MINIO_SECURE") or _resolve(storage_cfg.get("secure", "false"))).lower() == "true"

def get_s3_client():
    """Создаём и возвращаем клиент S3 (MinIO) с параметрами из .env и settings.yaml."""
    access_key = MINIO_ACCESS_KEY
    secret_key = MINIO_SECRET_KEY

    if not access_key or not secret_key:
        raise ValueError(
            "Не заданы MINIO_ACCESS_KEY/MINIO_SECRET_KEY. "
            "Заполни .env (см. .env.example) и повтори."
        )

    session = boto3.session.Session()

    # Для MinIO внутри docker-сети чаще всего надёжнее path-style:
    # http://minio:9000/bucket/key вместо http://bucket.minio:9000/key
    cfg = Config(
        s3={"addressing_style": "path"},
        retries={"max_attempts": 5, "mode": "standard"},
    )

    return session.client(
        service_name="s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=MINIO_REGION,
        config=cfg,
    )

def upload_json_bytes(bucket: str, key: str, json_str: str) -> None:
    """
    Загружает JSON-строку в S3/MinIO по указанному ключу.

    :param bucket: имя бакета
    :param key: ключ (путь внутри бакета)
    :param json_str: строка с JSON (уже сериализованный dict)
    """
    bytes_io = io.BytesIO(json_str.encode("utf-8"))

    logger.info("Uploading object to S3: bucket=%s, key=%s", bucket, key)
    s3 = get_s3_client()
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

def upload_file(bucket: str, key: str, file_path: str, content_type: str | None = None) -> None:
    """
    Загружает локальный файл в S3/MinIO по указанному ключу.
    """
    logger.info("Uploading file to S3: bucket=%s, key=%s, file=%s", bucket, key, file_path)

    s3 = get_s3_client()
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    try:
        if extra_args:
            s3.upload_file(file_path, bucket, key, ExtraArgs=extra_args)
        else:
            s3.upload_file(file_path, bucket, key)
    except Exception as e:
        logger.error(
            "Failed to upload file to S3: bucket=%s, key=%s, file=%s, error=%s",
            bucket,
            key,
            file_path,
            e,
        )
        raise
    else:
        logger.info("Successfully uploaded file to S3: bucket=%s, key=%s", bucket, key)