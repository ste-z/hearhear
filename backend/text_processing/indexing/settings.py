import re

try:
    from spacy.lang.en.stop_words import STOP_WORDS as BASE_STOP_WORDS
except ModuleNotFoundError:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as BASE_STOP_WORDS

from backend.text_processing.text_normalization import normalize_text_for_vectorization


DEFAULT_TOKEN_PATTERN = r"(?u)\b[^\W\d_]+(?:[-'][^\W\d_]+)*\b"
_TOKEN_RE = re.compile(DEFAULT_TOKEN_PATTERN)
_CUSTOM_STOP_WORDS = {
    "d",
    "dr",
    "ll",
    "m",
    "mr",
    "mrs",
    "ms",
    "re",
    "s",
    "t",
    "ve",
}


def _build_default_stop_words():
    stop_words = set()
    for value in set(BASE_STOP_WORDS).union(_CUSTOM_STOP_WORDS):
        normalized = normalize_text_for_vectorization(value).lower()
        if not normalized:
            continue
        tokens = _TOKEN_RE.findall(normalized)
        if tokens:
            stop_words.update(tokens)
        else:
            stop_words.add(normalized)
    return sorted(stop_words)


DEFAULT_STOP_WORDS = _build_default_stop_words()

DEFAULT_TFIDF_PARAMS = {
    "analyzer": "word",
    "token_pattern": DEFAULT_TOKEN_PATTERN,
    "lowercase": True,
    "strip_accents": "unicode",
    "stop_words": DEFAULT_STOP_WORDS,
    "min_df": 5,
}

# Keep vector-index artifacts comfortably below the common 100 MB git hosting limit.
MAX_VECTOR_INDEX_ARTIFACT_BYTES = 95 * 1024 * 1024
MAX_TERM_DOC_MATRIX_CHUNK_BYTES = MAX_VECTOR_INDEX_ARTIFACT_BYTES
