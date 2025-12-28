from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# По умолчанию берём “обычные” (не telegram) silver-файлы
SILVER_GLOB = os.getenv("SILVER_GLOB", "data/silver/articles_*_clean.json")
RAW_GLOB = os.getenv("RAW_GLOB", "data/raw/articles_*.json")

STRICT = os.getenv("STRICT", "").strip() not in {"", "0", "false", "False", "no", "No"}


@dataclass
class Paths:
    raw: Optional[Path]
    silver: Optional[Path]
    gold: Optional[Path]


def _latest(glob_pattern: str) -> Optional[Path]:
    paths = list(PROJECT_ROOT.glob(glob_pattern))
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

    ВАЖНО: добавляем '_v2_' чтобы articles_dedup предпочёл этот батч.
    """
    gold_dir = PROJECT_ROOT / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    name = silver_path.name
    for suf in (".jsonl", ".json", ".parquet", ".csv"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break

    # гарантируем наличие '_v2_' (а не просто 'v2')
    if "_v2_" not in f"_{name}_":
        name = f"{name}_v2"

    return gold_dir / f"{name}_processed.parquet"


def _resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _collect_paths() -> Paths:
    # В приоритете: latest silver (как у тебя раньше в логах)
    silver = _latest(SILVER_GLOB)

    # Если silver нет — пробуем raw (и тогда соберём silver)
    raw = None
    if silver is None:
        raw = _latest(RAW_GLOB)

    return Paths(raw=raw, silver=silver, gold=None)


def main() -> None:
    paths = _collect_paths()

    if paths.silver is None and paths.raw is None:
        raise SystemExit(
            f"[ERROR] Не найдено ни silver, ни raw.\n"
            f"Проверь папки data/silver и data/raw.\n"
            f"SILVER_GLOB={SILVER_GLOB}\nRAW_GLOB={RAW_GLOB}"
        )

    # 1) raw -> silver (если silver отсутствует)
    if paths.silver is None and paths.raw is not None:
        raw_path = _resolve_path(paths.raw)
        print(f"[INFO] Using latest raw: {raw_path}")
        _run([os.getenv("PYTHON", "python"), "-m", "src.pipeline.clean_raw_to_silver_local", "--input", str(raw_path)])
        # после прогона берём latest silver
        paths.silver = _latest(SILVER_GLOB)

    if paths.silver is None:
        raise SystemExit("[ERROR] Не удалось получить silver (после raw->silver тоже пусто).")

    silver_path = _resolve_path(paths.silver)
    print(f"[INFO] Using latest silver file: {silver_path}")

    # 2) validate silver (strict)
    if STRICT:
        _run([os.getenv("PYTHON", "python"), "scripts/validate_silver.py", "--input", str(silver_path)])

    # 3) silver -> gold (с явным output, чтобы был _v2_)
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
        # если скрипт есть — прогоняем SQL-проверки
        ch_runner = PROJECT_ROOT / "scripts" / "ch_run_sql.sh"
        quality_sql = PROJECT_ROOT / "sql" / "02_content_quality.sql"
        if ch_runner.exists() and quality_sql.exists():
            _run(["bash", str(ch_runner), str(quality_sql)])
        else:
            print("[WARN] quality gate skipped: scripts/ch_run_sql.sh or sql/02_content_quality.sql not found")

    print("[OK] ETL latest finished.")
    print(f"[OK] gold: {gold_path}")


if __name__ == "__main__":
    main()
