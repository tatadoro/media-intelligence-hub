from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from src.processing.cleaning import clean_articles_df
import argparse
from typing import Optional


def find_latest_raw_file(raw_dir: Path) -> Path:
    """
    Находит самый новый raw-файл формата articles_*.json в каталоге raw_dir.

    Критерий "самый новый" — по времени модификации файла (mtime).
    Если файлов нет, поднимает FileNotFoundError.
    """
    candidates = list(raw_dir.glob("articles_*.json"))
    if not candidates:
        raise FileNotFoundError(f"Не найдено ни одного файла articles_*.json в {raw_dir}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest

def transform_raw_to_silver(input_path: Path, output_path: Path) -> None:
    """
    Шаг пайплайна: читаем raw JSON, чистим тексты, сохраняем cleaned JSON.

    input_path  — путь к сырым данным (data/raw/...json)
    output_path — путь к файлу silver (data/silver/...json)
    """
    # 1. Читаем сырой JSON
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Ожидаем список словарей:
    # [{"title": ..., "raw_text": ..., ...}, ...]

    # 2. Превращаем в DataFrame
    df = pd.DataFrame(raw_data)

    # 3. Чистим тексты
    df_clean = clean_articles_df(
        df,
        text_column="raw_text",   # здесь важно, чтобы совпадало с тем, как колонка называется в rss_collector
        new_column="clean_text",
    )

    # 4. Назад в список словарей
    cleaned_records = df_clean.to_dict(orient="records")

    # 5. Создаём папку под выходной файл, если её ещё нет
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Сохраняем в JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)


def build_output_path_from_input(input_path: Path) -> Path:
    """
    По пути raw-файла строим путь к silver-файлу.

    data/raw/articles_20251209_212948.json
    → data/silver/articles_20251209_212948_clean.json
    """
    filename = input_path.name

    if filename.endswith(".json"):
        filename_clean = filename[:-5] + "_clean.json"
    else:
        filename_clean = filename + "_clean"

    project_root = Path(__file__).resolve().parents[2]
    silver_dir = project_root / "data" / "silver"

    return silver_dir / filename_clean


def main() -> None:
    """
    Точка входа для шага raw -> silver.

    Запуск:
      - с явным input:
          python -m src.pipeline.clean_raw_to_silver_local \
              --input data/raw/articles_20251211_153500.json

      - без input (возьмёт последний raw-файл в data/raw):
          python -m src.pipeline.clean_raw_to_silver_local
    """
    parser = argparse.ArgumentParser(
        description="Преобразование raw JSON-файла сarticleями в silver-слой "
                    "(добавление clean_text)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Путь к raw JSON-файлу (от корня проекта или абсолютный). "
             "Если не указан, будет использован последний файл в data/raw.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    if args.input:
        input_path = Path(args.input)
        # если путь относительный — считаем его от корня проекта
        if not input_path.is_absolute():
            input_path = project_root / input_path
    else:
        input_path = find_latest_raw_file(raw_dir)
        print(f"[INFO] input не указан, используем последний raw-файл: {input_path}")

    output_path = build_output_path_from_input(input_path)

    print(f"[INFO] Преобразуем raw -> silver")
    print(f"       raw:    {input_path}")
    print(f"       silver: {output_path}")

    transform_raw_to_silver(input_path, output_path)

    print(f"[OK] Готово. Очищенные данные сохранены в {output_path}")


if __name__ == "__main__":
    main()