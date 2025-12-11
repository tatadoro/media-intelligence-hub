from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from src.processing.summarization import enrich_articles_with_summary_and_keywords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Преобразовать silver-данные в gold-витрину (summary + keywords)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Путь к silver-файлу (JSON/Parquet/CSV), "
            "например data/silver/articles_20251211_153500_clean.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Необязательный путь к выходному gold-файлу. "
            "Если не указан, имя будет построено автоматически рядом с data/gold."
        ),
    )
    return parser.parse_args()


def infer_output_path(input_path: Path) -> Path:
    """
    Если вход: data/silver/articles_20251211_153500_clean.json,
    то выход: data/gold/articles_20251211_153500_processed.parquet
    """
    data_dir = input_path.parent.parent  # .../data
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    if stem.endswith("_clean"):
        stem = stem.replace("_clean", "_processed")
    else:
        stem = f"{stem}_processed"

    return gold_dir / f"{stem}.parquet"


def load_silver_df(input_path: Path) -> pd.DataFrame:
    """
    Загружаем silver-файл в DataFrame по расширению.
    """
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".json", ".jsonl"}:
        # для jsonl выставляем lines=True
        return pd.read_json(input_path, lines=(suffix == ".jsonl"))
    if suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")

    print("[INFO] Преобразуем silver -> gold (summary + keywords)")
    print(f"       silver: {input_path}")

    df_silver = load_silver_df(input_path)

    # Здесь происходит вся магия обогащения
    df_gold = enrich_articles_with_summary_and_keywords(df_silver)

    # Куда сохраняем
    output_path = Path(args.output) if args.output else infer_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"       gold:   {output_path}")
    df_gold.to_parquet(output_path, index=False)

    print("[INFO] Готово")


if __name__ == "__main__":
    main()
