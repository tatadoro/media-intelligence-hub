from __future__ import annotations

from typing import List, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Попробуем подключить pymorphy2 для лемматизации
try:
    import pymorphy3
    _MORPH = pymorphy3.MorphAnalyzer()
except ImportError:
    _MORPH = None


# Простейший токенайзер: берем слова из русских/латинских букв и цифр
_TOKEN_REGEX = re.compile(r"\b[а-яёa-z0-9]+\b", re.IGNORECASE)


# Небольшой набор русских стоп-слов (можно потом расширить)
RU_STOPWORDS = {
    "и", "но", "или", "а", "же", "то", "что", "чтобы",
    "как", "так", "также",
    "в", "на", "к", "от", "по", "из", "у", "за", "для",
    "о", "об", "про", "при", "над", "под", "между",
    "же", "ли", "бы", "же",
    "это", "этот", "эта", "эти", "того", "этого", "тому",
    "он", "она", "оно", "они", "мы", "вы", "ты",
    "тот", "та", "те", "который", "которые", "какой",
    "свой", "наш", "ваш", "их", "его", "ее",
    "там", "тут", "здесь", "сюда", "туда",
    "где", "когда", "пока", "уже", "еще",
    "да", "нет",
}


def _lemmatize_token(token: str) -> str:
    """
    Приводим слово к начальной форме, если доступен pymorphy2.
    Если нет — возвращаем как есть.
    """
    if _MORPH is None:
        return token
    p = _MORPH.parse(token)
    if not p:
        return token
    return p[0].normal_form


def normalize_for_tfidf(text: str) -> str:
    """
    Подготовка текста для TF-IDF:
    - приводим к нижнему регистру,
    - разбиваем на токены (слова),
    - лемматизируем,
    - выкидываем стоп-слова и очень короткие токены.
    Возвращаем строку с токенами через пробел.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    tokens = _TOKEN_REGEX.findall(text)

    norm_tokens: list[str] = []
    for token in tokens:
        if token in RU_STOPWORDS:
            continue

        lemma = _lemmatize_token(token)
        if lemma in RU_STOPWORDS:
            continue

        # отсечем совсем короткие штуки типа "с", "в", "на"
        if len(lemma) <= 2:
            continue

        norm_tokens.append(lemma)

    return " ".join(norm_tokens)

# --- 1. Разбиение текста на предложения ---

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> List[str]:
    """
    Наивно разбивает текст на предложения по знакам . ! ?
    Подходит как простое решение для новостных текстов.
    """
    if not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    sentences = _SENTENCE_SPLIT_REGEX.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# --- 2. Простая extractive-саммаризация ---

def extractive_summary(
    text: str,
    max_sentences: int = 3,
    max_chars: int = 500,
) -> str:
    """
    Простая extractive-саммаризация:
    берем первые несколько предложений, пока не превысим max_sentences и max_chars.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    selected: List[str] = []
    total_chars = 0

    for sent in sentences:
        if len(selected) >= max_sentences:
            break

        # если добавление предложения сильно превышает лимит символов
        # и у нас уже есть хотя бы одно предложение — остановимся
        if total_chars + len(sent) > max_chars and selected:
            break

        selected.append(sent)
        total_chars += len(sent)

    return " ".join(selected)


# --- 3. Ключевые слова с помощью TF-IDF ---

def extract_keywords_tfidf(
    texts: Sequence[str],
    top_k: int = 10,
    max_features: int = 5000,
) -> List[List[str]]:
    """
    Принимает список текстов (по одному на документ) и
    возвращает для каждого документа список ключевых слов (top_k по TF-IDF).

    Перед TF-IDF:
    - нормализуем текст (лемматизация + удаление стоп-слов),
    - считаем TF-IDF по нормализованным токенам.
    """
    cleaned_texts = [
        t if isinstance(t, str) else ""
        for t in texts
    ]

    # Нормализуем тексты для TF-IDF
    normalized_texts = [normalize_for_tfidf(t) for t in cleaned_texts]

    # Если все тексты пустые после нормализации — возвращаем пустые списки
    if all(not nt for nt in normalized_texts):
        return [[] for _ in normalized_texts]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),        # униграммы и биграммы
        lowercase=False,           # уже привели к lower
        token_pattern=r"(?u)\b\w+\b",
    )

    tfidf_matrix = vectorizer.fit_transform(normalized_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    keywords_per_doc: List[List[str]] = []

    for row in tfidf_matrix:
        if row.nnz == 0:
            keywords_per_doc.append([])
            continue

        row_array = row.toarray().ravel()
        top_indices = row_array.argsort()[-top_k:][::-1]
        top_scores = row_array[top_indices]

        mask = top_scores > 0
        top_terms = feature_names[top_indices][mask]

        keywords_per_doc.append(top_terms.tolist())

    return keywords_per_doc

# --- 4. Обогащение DataFrame: silver -> gold (summary + keywords) ---

EXPECTED_TEXT_COLUMN = "clean_text"


def _build_nlp_text_row(row: pd.Series) -> str:
    """
    Строим текст, который будем использовать для summary/keywords.

    1) Если clean_text непустой — используем его.
    2) Иначе собираем из title / description / summary (что есть).
    """
    # 1. clean_text
    base = row.get(EXPECTED_TEXT_COLUMN, "")
    if isinstance(base, str):
        base_stripped = base.strip()
        if base_stripped:
            return base_stripped

    # 2. Фолбэк: title + description/summary (если есть)
    parts = []

    title = row.get("title", "")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())

    # если в будущем в silver появятся поля description / summary, они тоже попадут сюда
    for col in ("description", "summary"):
        val = row.get(col, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    return ". ".join(parts)


def enrich_articles_with_summary_and_keywords(
    df_silver: pd.DataFrame,
    top_k_keywords: int = 10,
) -> pd.DataFrame:
    """
    Принимает DataFrame из silver-слоя (должна быть колонка clean_text),
    добавляет summary, keywords и несколько простых текстовых фичей.
    Возвращает новый DataFrame (копию).

    ВАЖНО: если clean_text пустой, используем title/description/summary как фолбэк.
    """
    df = df_silver.copy()

    if EXPECTED_TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"В DataFrame нет колонки '{EXPECTED_TEXT_COLUMN}' с очищенным текстом"
        )

    # Строим текст для NLP: либо clean_text, либо title/description/summary
    nlp_texts = df.apply(_build_nlp_text_row, axis=1)
    df["nlp_text"] = nlp_texts

    # 4.1. Summary для каждой статьи
    df["summary"] = df["nlp_text"].apply(
        lambda txt: extractive_summary(
            txt,
            max_sentences=3,
            max_chars=500,
        )
    )

    # 4.2. Ключевые слова (TF-IDF по всему корпусу)
    keywords_lists = extract_keywords_tfidf(
        df["nlp_text"].tolist(),
        top_k=top_k_keywords,
    )
    df["keywords"] = [
        "; ".join(kw_list) if kw_list else ""
        for kw_list in keywords_lists
    ]

    # 4.3. Несколько вспомогательных фичей
    df["text_length_chars"] = df["nlp_text"].apply(len)
    df["num_sentences"] = df["nlp_text"].apply(
        lambda txt: len(split_into_sentences(txt))
    )
    df["num_keywords"] = df["keywords"].apply(
        lambda s: 0 if not s else len([x for x in s.split(";") if x.strip()])
    )

    return df


# --- 5. Мини-тест на фейковых данных (для локальной проверки) ---

if __name__ == "__main__":
    # Простой тест модуля: создаем маленький DataFrame из двух "статей"
    data = [
        {
            "article_id": "1",
            "source": "test_source",
            "title": "Первая тестовая статья",
            "published_at": "2025-12-11T10:00:00",
            "url": "https://example.com/1",
            "clean_text": (
                "Это первая тестовая новость. "
                "В ней мы проверяем работу простого summarizer. "
                "Текст не несет смысловой нагрузки, но помогает протестировать код."
            ),
        },
        {
            "article_id": "2",
            "source": "test_source",
            "title": "Вторая тестовая статья",
            "published_at": "2025-12-11T11:00:00",
            "url": "https://example.com/2",
            "clean_text": (
                "Это вторая новость для проверки TF-IDF. "
                "Она про анализ медиа и обработку текстов. "
                "Мы хотим выделить ключевые слова про анализ и тексты."
            ),
        },
    ]
    df_test = pd.DataFrame(data)

    df_gold_test = enrich_articles_with_summary_and_keywords(df_test, top_k_keywords=5)

    cols_to_show = ["article_id", "title", "summary", "keywords"]
    print(df_gold_test[cols_to_show])
