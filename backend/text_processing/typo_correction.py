import math
import re

try:
    from rapidfuzz.distance.Levenshtein import distance as levenshtein_distance
except ModuleNotFoundError:
    levenshtein_distance = None

from backend.text_processing.indexing.settings import (
    DEFAULT_STOP_WORDS,
    DEFAULT_TOKEN_PATTERN,
)
from backend.text_processing.text_normalization import normalize_text_for_vectorization


_TOKEN_RE = re.compile(DEFAULT_TOKEN_PATTERN)
_MAX_DISTANCE = 2
_MAX_OPTIONS = 3


def _query_tokens(query):
    normalized_query = normalize_text_for_vectorization(query).lower()
    return [
        match.group(0)
        for match in _TOKEN_RE.finditer(normalized_query)
        if match.group(0).strip()
    ]


def _term_df_lookup(processor):
    terms = [str(term).lower() for term in list(getattr(processor, "terms", []) or [])]
    if not terms:
        return {}

    postings_indptr = getattr(processor, "postings_indptr", None)
    if postings_indptr is not None and len(postings_indptr) == len(terms) + 1:
        return {
            term: max(0, int(postings_indptr[index + 1]) - int(postings_indptr[index]))
            for index, term in enumerate(terms)
        }

    vectorizer = getattr(processor, "vectorizer", None)
    idf_values = getattr(vectorizer, "idf_", None)
    n_docs = int(getattr(processor, "n_docs", 0) or 0)
    if idf_values is not None and len(idf_values) == len(terms) and n_docs > 0:
        smooth_idf = bool(getattr(vectorizer, "smooth_idf", True))

        def df_from_idf(idf_value):
            idf_scale = math.exp(float(idf_value) - 1)
            if smooth_idf:
                return ((1 + n_docs) / idf_scale) - 1
            return n_docs / idf_scale

        return {
            term: max(0, int(round(df_from_idf(idf_values[index]))))
            for index, term in enumerate(terms)
        }

    return {term: 0 for term in terms}


def _replacement_pool(processor):
    n_docs = int(getattr(processor, "n_docs", 0) or 0)
    replacement_df = _term_df_lookup(processor)
    for stop_word in DEFAULT_STOP_WORDS:
        normalized = normalize_text_for_vectorization(stop_word).lower()
        if not normalized:
            continue
        replacement_df[normalized] = max(replacement_df.get(normalized, 0), n_docs)
    return replacement_df


def _candidate_replacements(term, replacement_df):
    if levenshtein_distance is None:
        return []

    candidates = []
    for candidate, doc_frequency in replacement_df.items():
        if candidate == term:
            continue
        if abs(len(term) - len(candidate)) >= 2:
            continue
        edit_distance = levenshtein_distance(
            term,
            candidate,
            score_cutoff=_MAX_DISTANCE,
        )
        if edit_distance > _MAX_DISTANCE:
            continue
        candidates.append(
            {
                "term": candidate,
                "distance": int(edit_distance),
                "df": int(doc_frequency),
            }
        )

    candidates.sort(key=lambda item: (item["distance"], -item["df"], item["term"]))
    return candidates[:_MAX_OPTIONS]


def _replace_query_terms(query, replacements):
    if not replacements:
        return str(query or "").strip()

    def replace_match(match):
        token = normalize_text_for_vectorization(match.group(0)).lower()
        return replacements.get(token, match.group(0))

    return _TOKEN_RE.sub(replace_match, str(query or "")).strip()


def build_query_typo_suggestion(query, processor):
    """
    Suggest one per-token typo correction at a time.

    The processor's feature list is already filtered by the index min_df setting,
    so exact/candidate checks use the same corpus vocabulary as retrieval. Exact
    hits suppress correction for that token only, allowing other query words to
    surface their own suggestions one at a time.
    """
    original_query = str(query or "").strip()
    if not original_query or processor is None:
        return None

    tokens = _query_tokens(original_query)
    if not tokens:
        return None

    replacement_df = _replacement_pool(processor)
    if not replacement_df:
        return None

    seen_terms = set()
    stop_words = set(DEFAULT_STOP_WORDS)
    for token in tokens:
        if token in seen_terms:
            continue
        seen_terms.add(token)
        if token in replacement_df or token in stop_words:
            continue

        options = _candidate_replacements(token, replacement_df)
        if not options:
            continue

        query_options = [
            {
                "query": _replace_query_terms(original_query, {token: option["term"]}),
                "label": option["term"],
                "replacements": {token: option["term"]},
                "distance": int(option["distance"]),
                "df": int(option["df"]),
            }
            for option in options[:_MAX_OPTIONS]
        ]
        return {
            "query": original_query,
            "highlighted_terms": [token],
            "options": query_options,
            "corrections": [
                {
                    "term": token,
                    "options": options,
                }
            ],
        }

    return None
