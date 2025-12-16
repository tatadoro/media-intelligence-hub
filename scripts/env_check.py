import os
import sys

REQUIRED = [
    "CH_HOST",
    "CH_PORT",
    "CH_DATABASE",
    "CH_USER",
    "CH_PASSWORD",
    "CLICKHOUSE_DB",
    "CLICKHOUSE_USER",
    "CLICKHOUSE_PASSWORD",
]

PAIRS = [
    ("CLICKHOUSE_DB", "CH_DATABASE"),
    ("CLICKHOUSE_USER", "CH_USER"),
    ("CLICKHOUSE_PASSWORD", "CH_PASSWORD"),
]

def main() -> int:
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        print("[ERROR] Missing required env vars:")
        for k in missing:
            print(f"  - {k}")
        print("\nTip: ensure you exported vars or have .env loaded in your shell/container.")
        return 2

    mismatched = []
    for a, b in PAIRS:
        if os.getenv(a) != os.getenv(b):
            mismatched.append((a, b, os.getenv(a), os.getenv(b)))

    if mismatched:
        print("[ERROR] Env mismatch (values must be identical):")
        for a, b, va, vb in mismatched:
            print(f"  - {a} != {b}")
            print(f"    {a}={va!r}")
            print(f"    {b}={vb!r}")
        return 3

    print("[OK] Env contract valid: CLICKHOUSE_* and CH_* are consistent.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
