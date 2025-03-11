# text_processing.py

import re
import spacy

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Очищает текст от ненужных символов и пробелов.

    :param text: Исходный текст.
    :return: Очищенный текст.
    """
    text = re.sub(r"\s+", " ", text)  # Удаление лишних пробелов
    text = re.sub(r"\[.*?\]", "", text)  # Удаление текстов в скобках, если они есть
    return text.strip()


def extract_keywords(text: str) -> list:
    """
    Извлекает ключевые слова из текста.

    :param text: Исходный текст.
    :return: Список ключевых слов.
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN", "ADJ"}]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Кратко резюмирует текст, выделяя основные моменты.

    :param text: Исходный текст.
    :param max_sentences: Максимальное количество предложений в кратком изложении.
    :return: Резюмированный текст.
    """
    sentences = text.split(".")
    return ". ".join(sentences[:max_sentences]) + "."
