import json
from datetime import datetime, timezone

from src.utils.s3_client import upload_json_bytes, MINIO_BUCKET


def main() -> None:
    # Используем timezone-aware datetime
    now_dt = datetime.now(timezone.utc)
    date_str = now_dt.strftime("%Y-%m-%d")
    ts_str = now_dt.strftime("%Y%m%d_%H%M%S")

    key = f"raw/{date_str}/smoke_test/smoke_{ts_str}.json"
    payload = {"ok": True, "timestamp": ts_str}

    # Преобразуем dict в JSON-строку
    json_str = json.dumps(payload, ensure_ascii=False)

    # ВАЖНО: передаём строку, а не dict
    upload_json_bytes(MINIO_BUCKET, key, json_str)

    print(f"Uploaded smoke test object to s3://{MINIO_BUCKET}/{key}")


if __name__ == "__main__":
    main()
