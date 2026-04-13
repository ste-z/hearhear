DEFAULT_TFIDF_PARAMS = {
    "analyzer": "word",
    "token_pattern": r"(?u)\b[^\W\d_]+(?:[-'][^\W\d_]+)*\b",
    "lowercase": True,
    "strip_accents": "unicode",
    "stop_words": "english",
    "min_df": 2,
}

# Keep vector-index artifacts comfortably below the common 100 MB git hosting limit.
MAX_VECTOR_INDEX_ARTIFACT_BYTES = 95 * 1024 * 1024
MAX_TERM_DOC_MATRIX_CHUNK_BYTES = MAX_VECTOR_INDEX_ARTIFACT_BYTES
