from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.processing.cleaning import clean_articles_df


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


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    # ВАЖНО: тут подставь реальное имя raw-файла, который у тебя уже лежит в data/raw
    input_file = project_root / "data" / "raw" / "articles_20251210_153232.json"

    output_file = build_output_path_from_input(input_file)

    transform_raw_to_silver(input_file, output_file)
    print(f"Готово. Очищенные данные сохранены в {output_file}")