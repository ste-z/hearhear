import re
from functools import lru_cache

import pandas as pd
import spacy


def normalize_whitespace(text, preserve_linebreaks=False):
    value = "" if text is None else str(text)
    if preserve_linebreaks:
        value = re.sub(r"[^\S\n]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=1)
def get_sentence_nlp():
    """
    Use spaCy's sentence segmentation in a lightweight, dependency-safe way.
    If an English model is available we use it, otherwise we fall back to a
    blank English pipeline with the rule-based sentencizer.
    """
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner"])
    except Exception:
        nlp = spacy.blank("en")

    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def split_into_sentences(text):
    normalized = normalize_whitespace(text, preserve_linebreaks=True)
    if not normalized:
        return []

    doc = get_sentence_nlp()(normalized)
    sentences = []
    for sent in doc.sents:
        cleaned = normalize_whitespace(sent.text)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def sentence_rows_from_text(article_text, prefix="s"):
    sentences = split_into_sentences(article_text)
    return [
        {"sentence_id": f"{prefix}{idx}", "sentence": sentence}
        for idx, sentence in enumerate(sentences)
    ]


def sentence_table_from_text(article_text, prefix="s"):
    return pd.DataFrame(sentence_rows_from_text(article_text, prefix=prefix))


def sentence_lookup(sentence_rows):
    if isinstance(sentence_rows, pd.DataFrame):
        rows = sentence_rows.to_dict(orient="records")
    else:
        rows = list(sentence_rows)
    return {row["sentence_id"]: row["sentence"] for row in rows}
