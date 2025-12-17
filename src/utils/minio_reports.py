from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.utils.s3_client import MINIO_BUCKET, upload_file, get_s3_client


def _slug_dt(s: str) -> str:
    # "2025-12-10 17:54:26" -> "2025-12-10T175426"
    s = s.strip().replace(" ", "T")
    s = s.replace(":", "")
    s = re.sub(r"[^0-9T\-]", "", s)
    return s


def build_key_idempotent(dt_from: str, dt_to: str, last_hours: int) -> str:
    dt_from_s = _slug_dt(dt_from)
    dt_to_s = _slug_dt(dt_to)
    # фиксированное имя => повтор того же окна перезапишет тот же объект
    return f"reports/daily/last_hours={last_hours}/dt_from={dt_from_s}/dt_to={dt_to_s}/report.md"


def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-hours", type=int, default=1)
    parser.add_argument("--report-path", type=str, required=True)
    parser.add_argument("--dt-from", type=str, required=True)
    parser.add_argument("--dt-to", type=str, required=True)
    args = parser.parse_args()

    report_path = Path(args.report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    key = build_key_idempotent(args.dt_from, args.dt_to, args.last_hours)

    upload_file(
        bucket=MINIO_BUCKET,
        key=key,
        file_path=str(report_path),
        content_type="text/markdown; charset=utf-8",
    )

    s3 = get_s3_client()
    s3.head_object(Bucket=MINIO_BUCKET, Key=key)
    print(f"[OK] Uploaded & verified: s3://{MINIO_BUCKET}/{key}")
    return key


if __name__ == "__main__":
    main()