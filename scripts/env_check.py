from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


# --- Existing contract (extended) ---

# Base keys that must always exist
REQUIRED_ENV_BASE = [
    # ClickHouse (compose)
    "CLICKHOUSE_DB",
    "CLICKHOUSE_USER",
    "CLICKHOUSE_PASSWORD",
    # ClickHouse (local scripts / Makefile / Python)
    "CH_HOST",
    "CH_PORT",
    "CH_DATABASE",
    "CH_USER",
    "CH_PASSWORD",
    # MinIO (compose)
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
    # Superset (compose)
    "SUPERSET_SECRET_KEY",
    "SUPERSET_ADMIN_PASSWORD",
    # Airflow secrets (compose)
    "AIRFLOW__CORE__FERNET_KEY",
    "AIRFLOW__WEBSERVER__SECRET_KEY",
]

# Airflow admin can be configured in ONE of two ways:
# 1) AIRFLOW_ADMIN_*  (your docker-compose.airflow.yml uses this)
# 2) _AIRFLOW_WWW_USER_* (alternative scheme from some Airflow compose examples)
AIRFLOW_ADMIN_SCHEMES: List[Tuple[str, List[str]]] = [
    (
        "AIRFLOW_ADMIN_*",
        [
            "AIRFLOW_ADMIN_USER",
            "AIRFLOW_ADMIN_PASSWORD",
            "AIRFLOW_ADMIN_EMAIL",
        ],
    ),
    (
        "_AIRFLOW_WWW_USER_*",
        [
            "_AIRFLOW_WWW_USER_USERNAME",
            "_AIRFLOW_WWW_USER_PASSWORD",
            "_AIRFLOW_WWW_USER_FIRSTNAME",
            "_AIRFLOW_WWW_USER_LASTNAME",
            "_AIRFLOW_WWW_USER_EMAIL",
        ],
    ),
]

PAIRS = [
    ("CLICKHOUSE_DB", "CH_DATABASE"),
    ("CLICKHOUSE_USER", "CH_USER"),
    ("CLICKHOUSE_PASSWORD", "CH_PASSWORD"),
]

PLACEHOLDERS = {"__FILL_ME__", "__REPLACE_ME__", "FILL_ME", "REPLACE_ME", "change-me-please"}

RE_KV = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$")


def _redact(value: str) -> str:
    if value is None:
        return "<none>"
    if value == "":
        return "<empty>"
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def _repo_root() -> Path:
    # scripts/env_check.py -> repo root = parents[1]
    return Path(__file__).resolve().parents[1]


def _parse_env_file(path: Path) -> Tuple[Dict[str, str], Dict[str, List[int]]]:
    """
    Returns:
      - kv: last value wins (docker-compose behavior)
      - dups: key -> list of line numbers where it appeared (1-based), only for duplicates
    """
    kv: Dict[str, str] = {}
    seen: Dict[str, List[int]] = {}

    if not path.exists():
        return kv, {}

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = RE_KV.match(raw)
        if not m:
            continue

        key = m.group(1).strip()
        val = m.group(2).strip()

        # strip quotes
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        kv[key] = val
        seen.setdefault(key, []).append(i)

    dups = {k: v for k, v in seen.items() if len(v) > 1}
    return kv, dups


def _is_set_env(key: str) -> bool:
    v = os.getenv(key)
    return v is not None and v != ""


def _is_set_kv(kv: Dict[str, str], key: str) -> bool:
    v = kv.get(key)
    return v is not None and v != ""


def _pick_airflow_scheme_from_env() -> Tuple[str, List[str]] | None:
    # choose scheme based on env vars
    any_set = []
    for name, keys in AIRFLOW_ADMIN_SCHEMES:
        any_set.append((name, keys, any(_is_set_env(k) for k in keys)))

    active = [x for x in any_set if x[2]]
    if not active:
        return None

    # prefer AIRFLOW_ADMIN_* (because compose uses it)
    for name, keys, is_active in any_set:
        if name == "AIRFLOW_ADMIN_*" and is_active:
            if len(active) > 1:
                print(
                    "[WARN] Both Airflow admin schemes are present in env. "
                    "Prefer keeping only one. Using: AIRFLOW_ADMIN_*"
                )
            return name, keys

    name, keys, _ = active[0]
    if len(active) > 1:
        print(
            "[WARN] Both Airflow admin schemes are present in env. "
            f"Prefer keeping only one. Using: {name}"
        )
    return name, keys


def _pick_airflow_scheme_from_kv(kv: Dict[str, str]) -> Tuple[str, List[str]] | None:
    # choose scheme based on .env file content
    any_set = []
    for name, keys in AIRFLOW_ADMIN_SCHEMES:
        any_set.append((name, keys, any(_is_set_kv(kv, k) for k in keys)))

    active = [x for x in any_set if x[2]]
    if not active:
        return None

    # prefer AIRFLOW_ADMIN_* (because compose uses it)
    for name, keys, is_active in any_set:
        if name == "AIRFLOW_ADMIN_*" and is_active:
            if len(active) > 1:
                print(
                    "[WARN] Both Airflow admin schemes are present in .env. "
                    "Prefer keeping only one. Using: AIRFLOW_ADMIN_*"
                )
            return name, keys

    name, keys, _ = active[0]
    if len(active) > 1:
        print(
            "[WARN] Both Airflow admin schemes are present in .env. "
            f"Prefer keeping only one. Using: {name}"
        )
    return name, keys


def main() -> int:
    # 1) Base check: required base env vars must exist
    missing_base = [k for k in REQUIRED_ENV_BASE if not _is_set_env(k)]
    if missing_base:
        print("[ERROR] Missing required env vars:")
        for k in missing_base:
            print(f"  - {k}")
        print("\nTip: ensure docker compose uses --env-file .env, or Makefile exports .env.")
        return 2

    # 2) Airflow admin scheme must be present (one of the supported schemes)
    scheme_env = _pick_airflow_scheme_from_env()
    if scheme_env is None:
        print("[ERROR] Airflow admin credentials are not set in env.")
        print("Set ONE of the following schemes in .env:")
        for name, keys in AIRFLOW_ADMIN_SCHEMES:
            print(f"  - {name}: " + ", ".join(keys))
        return 3

    scheme_name, scheme_keys = scheme_env
    missing_admin = [k for k in scheme_keys if not _is_set_env(k)]
    if missing_admin:
        print(f"[ERROR] Incomplete Airflow admin scheme in env: {scheme_name}")
        print("Missing keys:")
        for k in missing_admin:
            print(f"  - {k}")
        return 3

    # Full required list for env placeholders check etc.
    required_env = list(REQUIRED_ENV_BASE) + list(scheme_keys)

    # 3) Guard against placeholders left in the environment
    bad_placeholders = [k for k in required_env if os.getenv(k, "") in PLACEHOLDERS]
    if bad_placeholders:
        print("[ERROR] Placeholders detected in env (fill real values):")
        for k in bad_placeholders:
            print(f"  - {k}={os.getenv(k)!r}")
        return 3

    # 4) Contract: CH_* must match CLICKHOUSE_*
    mismatched = []
    for a, b in PAIRS:
        if os.getenv(a) != os.getenv(b):
            mismatched.append((a, b, os.getenv(a, ""), os.getenv(b, "")))

    if mismatched:
        print("[ERROR] Env mismatch (values must be identical):")
        for a, b, va, vb in mismatched:
            print(f"  - {a} != {b}")
            print(f"    {a}={_redact(va)}")
            print(f"    {b}={_redact(vb)}")
        return 4

    # 5) Sanity: CH_PORT must be int
    try:
        int(os.getenv("CH_PORT", ""))
    except Exception:
        print("[ERROR] CH_PORT must be an integer.")
        print(f"CH_PORT={os.getenv('CH_PORT')!r}")
        return 8

    # 6) Stronger check: if .env exists in repo root, detect duplicates there
    repo = _repo_root()
    env_path = repo / ".env"
    kv, dups = _parse_env_file(env_path)

    if env_path.exists():
        if dups:
            print("[ERROR] Duplicate keys found in .env (this often causes “it worked yesterday” issues):")
            for k, lines in sorted(dups.items()):
                print(f"  - {k}: lines {', '.join(map(str, lines))}")
            return 5

        # Determine which Airflow scheme is used in the .env file itself
        scheme_kv = _pick_airflow_scheme_from_kv(kv)
        if scheme_kv is None:
            print("[ERROR] .env file does not contain Airflow admin credentials.")
            print("Add ONE of the following schemes to .env:")
            for name, keys in AIRFLOW_ADMIN_SCHEMES:
                print(f"  - {name}: " + ", ".join(keys))
            return 7

        scheme_name_file, scheme_keys_file = scheme_kv
        required_in_file = list(REQUIRED_ENV_BASE) + list(scheme_keys_file)

        # Also ensure required keys are present in the .env file itself (not only exported elsewhere)
        missing_in_file = [k for k in required_in_file if k not in kv or kv.get(k, "") == ""]
        if missing_in_file:
            print("[ERROR] .env file is missing required keys (even if shell/container has them):")
            for k in missing_in_file:
                print(f"  - {k}")
            return 6

        placeholders_in_file = [k for k in required_in_file if kv.get(k, "") in PLACEHOLDERS]
        if placeholders_in_file:
            print("[ERROR] Placeholders detected in .env file:")
            for k in placeholders_in_file:
                print(f"  - {k}={kv.get(k)!r}")
            return 7

        # Validate CH_PORT in file as well (helps when running compose via --env-file)
        try:
            int(kv.get("CH_PORT", ""))
        except Exception:
            print("[ERROR] In .env: CH_PORT must be an integer.")
            print(f"CH_PORT={kv.get('CH_PORT')!r}")
            return 9

        # If both schemes are in file, warn (non-fatal)
        file_has_admin = any(_is_set_kv(kv, k) for k in AIRFLOW_ADMIN_SCHEMES[0][1])
        file_has_www = any(_is_set_kv(kv, k) for k in AIRFLOW_ADMIN_SCHEMES[1][1])
        if file_has_admin and file_has_www:
            print(
                "[WARN] Both Airflow admin schemes are present in .env. "
                f"Prefer keeping only one. Using: {scheme_name_file}"
            )

    print("[OK] Env contract valid: CLICKHOUSE_* and CH_* are consistent; required keys present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
