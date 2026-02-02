from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Режим источника: rss|tg|all
# - rss: по умолчанию берём “обычные” (не telegram) silver
# - tg:  telegram silver
# - all: сначала tg, затем rss (два прогона подряд)
SOURCE = os.getenv("SOURCE", "rss").strip().lower()
if SOURCE not in {"rss", "tg", "all"}:
    raise SystemExit("[ERROR] SOURCE must be one of: rss, tg, all")

# Дефолтные glob-паттерны (могут быть переопределены env-переменными SILVER_GLOB/RAW_GLOB)
DEFAULT_SILVER_GLOB_RSS = "data/silver/articles_*_clean.json"
DEFAULT_RAW_GLOB_RSS = "data/raw/articles_*.json"

# Для TG ориентируемся на clean-файлы с telegram в имени (у тебя они формируются clean_raw_to_silver_local)
DEFAULT_SILVER_GLOB_TG = "data/silver/articles_*_telegram_*_clean.json"
DEFAULT_RAW_GLOB_TG = "data/raw/articles_*_telegram*.json"

# Пользовательские overrides (если заданы)
SILVER_GLOB = os.getenv("SILVER_GLOB", "").strip()
RAW_GLOB = os.getenv("RAW_GLOB", "").strip()

# Опциональный фильтр исключения по подстроке имени файла
# Для RSS по умолчанию исключаем telegram, чтобы TG silver не “перебивал” RSS.
SILVER_EXCLUDE_CONTAINS = os.getenv("SILVER_EXCLUDE_CONTAINS", "").strip()

STRICT = os.getenv("STRICT", "").strip() not in {"", "0", "false", "False", "no", "No"}


@dataclass
class Paths:
    raw: Optional[Path]
    silver: Optional[Path]
    gold: Optional[Path]


def _resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _latest(glob_pattern: str, exclude_contains: str = "") -> Optional[Path]:
    paths = list(PROJECT_ROOT.glob(glob_pattern))
    if exclude_contains:
        paths = [p for p in paths if exclude_contains not in p.name]
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def _run(cmd: list[str], env: Optional[dict[str, str]] = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _infer_gold_v2_path_from_silver(silver_path: Path) -> Path:
    """
    silver: data/silver/articles_20251210_155554_..._clean.json
    gold:   data/gold/articles_20251210_155554_..._clean_v2_processed.parquet

    ВАЖНО: добавляем '_v2' чтобы articles_dedup предпочёл этот батч.
    """
    gold_dir = PROJECT_ROOT / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    name = silver_path.name
    for suf in (".jsonl", ".json", ".parquet", ".csv"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break

    # Если name уже содержит v2 (в любом виде) — не дублируем
    # Гарантируем аккуратное добавление суффикса "_v2" к имени батча.
    if "_v2" not in name:
        name = f"{name}_v2"

    return gold_dir / f"{name}_processed.parquet"


def _select_defaults_for_source(source: str) -> tuple[str, str, str]:
    """
    Возвращает (silver_glob, raw_glob, silver_exclude_contains_default)
    """
    if source == "tg":
        return DEFAULT_SILVER_GLOB_TG, DEFAULT_RAW_GLOB_TG, ""
    # rss
    return DEFAULT_SILVER_GLOB_RSS, DEFAULT_RAW_GLOB_RSS, "telegram"


def _collect_paths(silver_glob: str, raw_glob: str, exclude_contains: str) -> Paths:
    silver = _latest(silver_glob, exclude_contains=exclude_contains)

    raw = None
    if silver is None:
        raw = _latest(raw_glob)

    return Paths(raw=raw, silver=silver, gold=None)


def _etl_one(source: str) -> None:
    # Определяем дефолты для конкретного источника
    silver_glob_default, raw_glob_default, exclude_default = _select_defaults_for_source(source)

    # Применяем overrides, если заданы
    silver_glob = SILVER_GLOB or silver_glob_default
    raw_glob = RAW_GLOB or raw_glob_default

    # exclude: если пользователь явно задал — используем его, иначе дефолт (для rss это "telegram")
    exclude_contains = SILVER_EXCLUDE_CONTAINS if SILVER_EXCLUDE_CONTAINS else exclude_default

    print(f"[INFO] SOURCE={source}")
    print(f"[INFO] SILVER_GLOB={silver_glob}")
    print(f"[INFO] RAW_GLOB={raw_glob}")
    if exclude_contains:
        print(f"[INFO] SILVER_EXCLUDE_CONTAINS={exclude_contains}")

    paths = _collect_paths(silver_glob, raw_glob, exclude_contains)

    if paths.silver is None and paths.raw is None:
        raise SystemExit(
            f"[ERROR] Не найдено ни silver, ни raw.\n"
            f"Проверь папки data/silver и data/raw.\n"
            f"SILVER_GLOB={silver_glob}\nRAW_GLOB={raw_glob}\nEXCLUDE={exclude_contains or '(none)'}"
        )

    # 1) raw -> silver (если silver отсутствует)
    if paths.silver is None and paths.raw is not None:
        raw_path = _resolve_path(paths.raw)
        print(f"[INFO] Using latest raw: {raw_path}")
        _run(
            [
                os.getenv("PYTHON", "python"),
                "-m",
                "src.pipeline.clean_raw_to_silver_local",
                "--input",
                str(raw_path),
            ]
        )

        # после прогона берём latest silver (строго по тем же правилам выбора)
        paths.silver = _latest(silver_glob, exclude_contains=exclude_contains)

    if paths.silver is None:
        raise SystemExit(
            "[ERROR] Не удалось получить silver (после raw->silver тоже пусто). "
            f"SILVER_GLOB={silver_glob} EXCLUDE={exclude_contains or '(none)'}"
        )

    silver_path = _resolve_path(paths.silver)
    print(f"[INFO] Using latest silver file: {silver_path}")

    # 2) validate silver (strict)
    if STRICT:
        _run([os.getenv("PYTHON", "python"), "scripts/validate_silver.py", "--input", str(silver_path)])

    # 3) silver -> gold (с явным output, чтобы был _v2)
    gold_path = _infer_gold_v2_path_from_silver(silver_path)
    paths.gold = gold_path
    _run(
        [
            os.getenv("PYTHON", "python"),
            "-m",
            "src.pipeline.silver_to_gold_local",
            "--input",
            str(silver_path),
            "--output",
            str(gold_path),
        ]
    )

    # 4) validate gold (strict)
    if STRICT:
        _run([os.getenv("PYTHON", "python"), "scripts/validate_gold.py", "--input", str(gold_path)])

    # 5) gold -> ClickHouse (batch-id)
    batch_id = os.getenv("BATCH_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[INFO] Using batch-id: {batch_id}")

    # ВАЖНО: gold_to_clickhouse_local у тебя вызывает clickhouse-client внутри контейнера,
    # поэтому порт должен быть 9000 (native), иначе были network errors на 8123.
    env = os.environ.copy()
    env["CH_HOST_DOCKER"] = env.get("CH_HOST_DOCKER", "clickhouse")
    env["CH_PORT_DOCKER"] = env.get("CH_PORT_DOCKER", "9000")
    env["CH_HOST"] = env.get("CH_HOST", "localhost")
    env["CH_PORT"] = env.get("CH_PORT", "9000")

    _run(
        [
            os.getenv("PYTHON", "python"),
            "-m",
            "src.pipeline.gold_to_clickhouse_local",
            "--input",
            str(gold_path),
            "--batch-id",
            batch_id,
        ],
        env=env,
    )

    # 6) quality gate (strict)
    if STRICT:
        ch_runner = PROJECT_ROOT / "scripts" / "ch_run_sql.sh"
        quality_sql = PROJECT_ROOT / "sql" / "02_content_quality.sql"
        if ch_runner.exists() and quality_sql.exists():
            _run(["bash", str(ch_runner), str(quality_sql)])
        else:
            print("[WARN] quality gate skipped: scripts/ch_run_sql.sh or sql/02_content_quality.sql not found")

    print("[OK] ETL latest finished.")
    print(f"[OK] gold: {gold_path}")


def main() -> None:
    if SOURCE == "all":
        # Важно: сначала TG, затем RSS — чтобы “частый” TG не перекрывал RSS.
        _etl_one("tg")
        _etl_one("rss")
        return

    _etl_one(SOURCE)


if __name__ == "__main__":
    main()