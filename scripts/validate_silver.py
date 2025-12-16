from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_KEYS = ("published_at", "title", "link", "source")


def load_first_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not data:
            raise ValueError("Файл пустой: JSON-массив без записей.")
        if not isinstance(data[0], dict):
            raise ValueError("Ожидается массив объектов (dict).")
        return data[0]

    if isinstance(data, dict):
        return data

    raise ValueError("Ожидается JSON-объект или JSON-массив объектов.")


def main() -> int:
    p = argparse.ArgumentParser(description="Validate silver JSON contract.")
    p.add_argument("--input", required=True, help="Path to silver *_clean.json")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 2

    try:
        rec = load_first_record(path)
    except Exception as e:
        print(f"[ERROR] Cannot parse silver file: {path}\n{e}", file=sys.stderr)
        return 2

    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        print(f"[ERROR] Silver contract failed: missing keys {missing} in {path}", file=sys.stderr)
        return 1

    # мягкие проверки типов/значений (без падения пайплайна)
    empty = [k for k in REQUIRED_KEYS if not rec.get(k)]
    if empty:
        print(f"[WARN] Some required keys are empty in first record: {empty}")

    print(f"[OK] Silver contract valid: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())