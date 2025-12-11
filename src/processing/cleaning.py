from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup

def clean_html(raw_html: str) -> str:
    """
    Принимает строку с HTML/шумным текстом и возвращает очищенный текст.

    Шаги:
    1. Обрабатываем пустые и нестроковые значения.
    2. Парсим HTML через BeautifulSoup.
    3. Удаляем теги <script> и <style>.
    4. Достаём текст.
    5. Нормализуем пробелы и переносы строк.
    """
    if not isinstance(raw_html, str) or not raw_html.strip():
        return ""

    # Парсим HTML
    soup = BeautifulSoup(raw_html, "html.parser")

    # Удаляем скрипты и стили
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Получаем текст, между блоками ставим пробел
    text = soup.get_text(separator=" ")

    # Заменяем любые последовательности пробельных символов на один пробел
    text = re.sub(r"\s+", " ", text)

    # Обрезаем пробелы по краям
    return text.strip()

def clean_articles_df(
    df: pd.DataFrame,
    text_column: str = "raw_text",
    new_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Принимает DataFrame с колонкой text_column (сырой текст) и
    добавляет колонку new_column (очищенный текст).

    Возвращает НОВУЮ копию DataFrame, не меняя исходный df.
    """
    if text_column not in df.columns:
        raise ValueError(f"В DataFrame нет колонки '{text_column}'")

    df_clean = df.copy()
    df_clean[new_column] = df_clean[text_column].apply(clean_html)
    return df_clean

def _demo():
    """Небольшая демонстрация работы очистки."""
    raw_examples = [
        "<p>Привет, мир!</p>",
        "<div>Текст со <b>strong</b> тегами и скриптом"
        "<script>console.log('test');</script></div>",
        None,
        "Уже чистый текст без HTML.",
    ]

    df = pd.DataFrame({"raw_text": raw_examples})
    df_clean = clean_articles_df(df)

    print("=== ОРИГИНАЛ ===")
    print(df["raw_text"])
    print("\n=== ОЧИЩЕНО ===")
    print(df_clean["clean_text"])


if __name__ == "__main__":
    _demo()