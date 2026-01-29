from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd


# Регексы компилируем один раз
_WS_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.!?;:])")


def clean_html(raw_html: Any) -> str:
    """
    Очищает HTML/шумный текст и возвращает нормализованный plain-text.

    Шаги:
    1) пустые и нестроковые значения -> ""
    2) если похоже на HTML — вычищаем через BeautifulSoup (если доступен)
    3) html.unescape
    4) нормализация пробелов и пробелов перед пунктуацией
    """
    if not isinstance(raw_html, str) or not raw_html.strip():
        return ""

    text = raw_html

    # Быстрый путь: если нет признаков HTML и сущностей — просто нормализуем пробелы.
    # Это сильно ускоряет обработку "обычных" текстов.
    looks_like_html = ("<" in text and ">" in text) or ("&" in text and ";" in text)
    if looks_like_html:
        try:
            from bs4 import BeautifulSoup  # локальный импорт: модуль может быть не установлен в окружении

            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        except Exception:
            # Фоллбек: грубое удаление тегов (хуже, чем BS4, но лучше чем падение)
            text = re.sub(r"<[^>]+>", " ", text)

    text = html.unescape(text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_articles_df(
    df: pd.DataFrame,
    text_column: str = "raw_text",
    new_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Принимает DataFrame с колонкой text_column (сырой текст)
    и добавляет колонку new_column (очищенный текст).
    Возвращает новую копию DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"В DataFrame нет колонки '{text_column}'")

    df_clean = df.copy()
    df_clean[new_column] = df_clean[text_column].apply(clean_html)
    return df_clean