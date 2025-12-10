import json
from datetime import datetime, timezone

from src.utils.s3_client import upload_json_bytes, MINIO_BUCKET


def main() -> None:
    # Используем timezone-aware datetime
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    key = f"raw/{now}/smoke_test/smoke_{now}.json"
    payload = {"ok": True, "timestamp": now}

    # Преобразуем dict в JSON-строку
    json_str = json.dumps(payload, ensure_ascii=False)

    # ВАЖНО: передаём строку, а не dict
    upload_json_bytes(MINIO_BUCKET, key, json_str)

    print(f"Uploaded smoke test object to s3://{MINIO_BUCKET}/{key}")


if __name__ == "__main__":
    main()
