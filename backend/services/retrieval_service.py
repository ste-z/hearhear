import re

import numpy as np
from sqlalchemy import func, or_

from backend.db.models import GuardianArticle
from backend.runtime.runtime_debug import log_runtime_event
from backend.services.rocchio_feedback import (
    build_rocchio_processor_searcher,
    normalize_article_id_list,
)
from backend.text_processing.search_helpers import (
    attach_query_svd_chart_dimensions as _attach_query_svd_chart_dimensions,
    DEFAULT_RETRIEVAL_MODEL,
    SUPPORTED_RETRIEVAL_MODELS as _SUPPORTED_RETRIEVAL_MODELS,
    build_matches,
    build_retrieval_processor,
    normalize_retrieval_model,
    query_svd_corpus_chart_dimensions,
    query_svd_dimensions,
    serialize_article,
)
from backend.text_processing.typo_correction import build_query_typo_suggestion


SUPPORTED_RETRIEVAL_MODELS = _SUPPORTED_RETRIEVAL_MODELS
SUPPORTED_RERANK_SELECTION_MODES = ("manual", "automatic")
DEFAULT_RERANK_SELECTION_MODE = "automatic"
DEFAULT_AUTO_RERANK_THRESHOLDS = {
    "tfidf": 0.3,
    "svd": 0.6,
}
MAX_AUTO_RERANK_CANDIDATES = 100
WORD_COUNT_PATTERN = re.compile(r"\b[\w'-]+\b")
READING_TIME_WORDS_PER_MINUTE = 250


def normalize_avoid_words(value):
    if value is None:
        return []

    if isinstance(value, str):
        raw_values = [value]
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]

    normalized = []
    seen = set()
    for raw_value in raw_values:
        for token in WORD_COUNT_PATTERN.findall(str(raw_value or "")):
            word = token.strip("_-'").casefold()
            if not word or word in seen:
                continue
            normalized.append(word)
            seen.add(word)
    return normalized


def _coerce_year(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_length_count(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _article_character_count_expression():
    return GuardianArticle.body_character_count


def _article_word_count_expression():
    return GuardianArticle.body_word_count


def _reading_minutes_from_word_count(word_count):
    resolved_count = _coerce_length_count(word_count)
    if resolved_count is None or resolved_count <= 0:
        return 0
    return max(1, ((resolved_count - 1) // READING_TIME_WORDS_PER_MINUTE) + 1)


def _reading_time_min_word_count(reading_minutes):
    resolved_minutes = _coerce_length_count(reading_minutes)
    if resolved_minutes is None:
        return None
    if resolved_minutes <= 1:
        return 1
    return ((resolved_minutes - 1) * READING_TIME_WORDS_PER_MINUTE) + 1


def _reading_time_max_word_count(reading_minutes):
    resolved_minutes = _coerce_length_count(reading_minutes)
    if resolved_minutes is None:
        return None
    return max(0, resolved_minutes * READING_TIME_WORDS_PER_MINUTE)


def available_article_year_range():
    min_year, max_year = (
        GuardianArticle.query.with_entities(
            func.min(GuardianArticle.year),
            func.max(GuardianArticle.year),
        )
        .first()
        or (None, None)
    )
    if min_year is None or max_year is None:
        return None, None
    return int(min_year), int(max_year)


def available_article_character_range():
    character_count = _article_character_count_expression()
    min_characters, max_characters = (
        GuardianArticle.query.with_entities(
            func.min(character_count),
            func.max(character_count),
        )
        .first()
        or (None, None)
    )
    if min_characters is None or max_characters is None:
        return None, None
    return int(min_characters), int(max_characters)


def available_article_word_range():
    word_count = _article_word_count_expression()
    min_words, max_words = (
        GuardianArticle.query.with_entities(
            func.min(word_count),
            func.max(word_count),
        )
        .first()
        or (None, None)
    )
    if min_words is None or max_words is None:
        return None, None
    return int(min_words), int(max_words)


def available_article_reading_time_range():
    min_words, max_words = available_article_word_range()
    if min_words is None or max_words is None:
        return None, None
    return (
        _reading_minutes_from_word_count(min_words),
        _reading_minutes_from_word_count(max_words),
    )


def normalize_rerank_selection_mode(value, default=DEFAULT_RERANK_SELECTION_MODE):
    if value is None:
        return default

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return default

    if normalized in {"manual", "top_k", "topk"}:
        return "manual"
    if normalized in {"automatic", "auto", "threshold", "thresholded"}:
        return "automatic"

    supported = ", ".join(SUPPORTED_RERANK_SELECTION_MODES)
    raise ValueError(
        f"Unsupported rerank_selection_mode {value!r}. Supported modes: {supported}."
    )


def default_auto_rerank_threshold(retrieval_model=DEFAULT_RETRIEVAL_MODEL):
    resolved_model = normalize_retrieval_model(retrieval_model)
    return float(DEFAULT_AUTO_RERANK_THRESHOLDS[resolved_model])


def resolve_auto_rerank_threshold(value, retrieval_model=DEFAULT_RETRIEVAL_MODEL):
    default_threshold = default_auto_rerank_threshold(retrieval_model=retrieval_model)
    resolved = _coerce_float(value, default=default_threshold)
    if resolved is None:
        return default_threshold
    return max(0.0, min(1.0, float(resolved)))


def normalize_article_year_range(year_start=None, year_end=None):
    available_start, available_end = available_article_year_range()
    if available_start is None or available_end is None:
        return None, None
    if year_start is None and year_end is None:
        return None, None

    resolved_start = _coerce_year(year_start)
    resolved_end = _coerce_year(year_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError("Start year must be less than or equal to end year.")

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def normalize_article_character_range(character_start=None, character_end=None):
    available_start, available_end = available_article_character_range()
    if available_start is None or available_end is None:
        return None, None
    if character_start is None and character_end is None:
        return None, None

    resolved_start = _coerce_length_count(character_start)
    resolved_end = _coerce_length_count(character_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError(
            "Minimum article length must be less than or equal to maximum article length."
        )

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def normalize_article_word_range(word_start=None, word_end=None):
    available_start, available_end = available_article_word_range()
    if available_start is None or available_end is None:
        return None, None
    if word_start is None and word_end is None:
        return None, None

    resolved_start = _coerce_length_count(word_start)
    resolved_end = _coerce_length_count(word_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError(
            "Minimum article word count must be less than or equal to maximum article word count."
        )

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def normalize_article_reading_time_range(reading_time_start=None, reading_time_end=None):
    available_start, available_end = available_article_reading_time_range()
    if available_start is None or available_end is None:
        return None, None
    if reading_time_start is None and reading_time_end is None:
        return None, None

    resolved_start = _coerce_length_count(reading_time_start)
    resolved_end = _coerce_length_count(reading_time_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError(
            "Minimum article reading time must be less than or equal to maximum article reading time."
        )

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def _word_range_for_reading_time_range(reading_time_start=None, reading_time_end=None):
    if reading_time_start is None and reading_time_end is None:
        return None, None
    return (
        _reading_time_min_word_count(reading_time_start),
        _reading_time_max_word_count(reading_time_end),
    )


def _ranked_article_year(article, year_lookup):
    if isinstance(article, str):
        return _coerce_year(year_lookup.get(article))
    if isinstance(article, dict):
        return _coerce_year(article.get("year"))
    return _coerce_year(getattr(article, "year", None))


def _ranked_article_id(article):
    if isinstance(article, str):
        return article.strip()
    if isinstance(article, dict):
        value = article.get("id")
    else:
        value = getattr(article, "id", None)
    article_id = str(value or "").strip()
    return article_id or None


def _body_text_character_count(value):
    if value is None:
        return None
    return len(str(value))


def _body_text_word_count(value):
    if value is None:
        return None
    return len(WORD_COUNT_PATTERN.findall(str(value)))


def _word_token_variants(value):
    tokens = set()
    if value is None:
        return tokens

    for token in WORD_COUNT_PATTERN.findall(str(value)):
        normalized = token.strip("_-'").casefold()
        if not normalized:
            continue
        tokens.add(normalized)
        for part in re.split(r"[-']", normalized):
            part = part.strip("_")
            if part:
                tokens.add(part)
    return tokens


def _article_searchable_token_set(article):
    if isinstance(article, dict):
        fields = [
            article.get("title"),
            article.get("summary"),
            article.get("body_text"),
            article.get("central_claim_summary"),
        ]
        keywords = article.get("keywords")
    else:
        fields = [
            getattr(article, "title", None),
            getattr(article, "summary", None),
            getattr(article, "body_text", None),
        ]
        keywords = getattr(article, "keywords", None)

    if isinstance(keywords, (list, tuple, set)):
        fields.extend(keywords)
    elif keywords:
        fields.append(keywords)

    tokens = set()
    for field in fields:
        tokens.update(_word_token_variants(field))
    return tokens


def _ranked_article_character_count(article, character_lookup):
    if isinstance(article, str):
        return _coerce_length_count(character_lookup.get(article))
    if isinstance(article, dict):
        for count_key in (
            "character_count",
            "body_character_count",
            "article_character_count",
        ):
            explicit_count = _coerce_length_count(article.get(count_key))
            if explicit_count is not None:
                return explicit_count
        if "body_text" in article:
            return _body_text_character_count(article.get("body_text"))
        article_id = _ranked_article_id(article)
        return _coerce_length_count(character_lookup.get(article_id))

    for count_attr in ("character_count", "body_character_count", "article_character_count"):
        explicit_count = _coerce_length_count(getattr(article, count_attr, None))
        if explicit_count is not None:
            return explicit_count
    if hasattr(article, "body_text"):
        return _body_text_character_count(getattr(article, "body_text", None))
    article_id = _ranked_article_id(article)
    return _coerce_length_count(character_lookup.get(article_id))


def _ranked_article_word_count(article, word_lookup):
    if isinstance(article, str):
        return _coerce_length_count(word_lookup.get(article))
    if isinstance(article, dict):
        for count_key in (
            "word_count",
            "body_word_count",
            "article_word_count",
        ):
            explicit_count = _coerce_length_count(article.get(count_key))
            if explicit_count is not None:
                return explicit_count
        if "body_text" in article:
            return _body_text_word_count(article.get("body_text"))
        article_id = _ranked_article_id(article)
        return _coerce_length_count(word_lookup.get(article_id))

    explicit_count = _coerce_length_count(getattr(article, "body_word_count", None))
    if explicit_count is not None:
        return explicit_count
    if hasattr(article, "body_text"):
        return _body_text_word_count(getattr(article, "body_text", None))
    article_id = _ranked_article_id(article)
    return _coerce_length_count(word_lookup.get(article_id))


def _filter_ranked_articles_by_year_range(ranked_articles, year_start=None, year_end=None):
    if year_start is None and year_end is None:
        return list(ranked_articles)

    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    year_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(GuardianArticle.id, GuardianArticle.year)
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        year_lookup = {
            article_id: _coerce_year(article_year)
            for article_id, article_year in rows
        }

    filtered = []
    for article, score in ranked_articles:
        article_year = _ranked_article_year(article, year_lookup)
        if article_year is None:
            continue
        if year_start is not None and article_year < year_start:
            continue
        if year_end is not None and article_year > year_end:
            continue
        filtered.append((article, score))

    return filtered


def _filter_ranked_articles_by_character_range(
    ranked_articles,
    character_start=None,
    character_end=None,
):
    if character_start is None and character_end is None:
        return list(ranked_articles)

    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    character_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(
                GuardianArticle.id,
                _article_character_count_expression(),
            )
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        character_lookup = {
            article_id: _coerce_length_count(character_count)
            for article_id, character_count in rows
        }

    filtered = []
    for article, score in ranked_articles:
        character_count = _ranked_article_character_count(article, character_lookup)
        if character_count is None:
            continue
        if character_start is not None and character_count < character_start:
            continue
        if character_end is not None and character_count > character_end:
            continue
        filtered.append((article, score))

    return filtered


def _filter_ranked_articles_by_word_range(ranked_articles, word_start=None, word_end=None):
    if word_start is None and word_end is None:
        return list(ranked_articles)

    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    word_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(
                GuardianArticle.id,
                _article_word_count_expression(),
            )
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        word_lookup = {
            article_id: _coerce_length_count(word_count)
            for article_id, word_count in rows
        }

    filtered = []
    for article, score in ranked_articles:
        word_count = _ranked_article_word_count(article, word_lookup)
        if word_count is None:
            continue
        if word_start is not None and word_count < word_start:
            continue
        if word_end is not None and word_count > word_end:
            continue
        filtered.append((article, score))

    return filtered


def _filter_ranked_articles_by_avoid_words(ranked_articles, words_to_avoid=None):
    resolved_avoid_words = normalize_avoid_words(words_to_avoid)
    if not resolved_avoid_words:
        return list(ranked_articles)

    avoid_word_set = set(resolved_avoid_words)
    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    token_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(
                GuardianArticle.id,
                GuardianArticle.title,
                GuardianArticle.summary,
                GuardianArticle.body_text,
                GuardianArticle.keywords,
            )
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        token_lookup = {
            article_id: _article_searchable_token_set(
                {
                    "title": title,
                    "summary": summary,
                    "body_text": body_text,
                    "keywords": keywords,
                }
            )
            for article_id, title, summary, body_text, keywords in rows
        }

    filtered = []
    for article, score in ranked_articles:
        if isinstance(article, str):
            article_tokens = token_lookup.get(article.strip())
            if article_tokens is None:
                filtered.append((article, score))
                continue
        else:
            article_tokens = _article_searchable_token_set(article)

        if article_tokens.isdisjoint(avoid_word_set):
            filtered.append((article, score))

    return filtered


def _filter_ranked_articles_by_excluded_ids(ranked_articles, excluded_article_ids):
    excluded_ids = set(normalize_article_id_list(excluded_article_ids))
    if not excluded_ids:
        return list(ranked_articles)

    return [
        (article, score)
        for article, score in ranked_articles
        if _ranked_article_id(article) not in excluded_ids
    ]


def _filter_matches_by_topic_threshold(article_matches, threshold):
    resolved_threshold = float(threshold)
    return [
        match
        for match in article_matches
        if _coerce_float(match.get("score"), default=0.0) >= resolved_threshold
    ]


def select_rerank_candidates(
    query,
    top_n=100,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
    words_to_avoid=None,
    rerank_selection_mode=DEFAULT_RERANK_SELECTION_MODE,
    rerank_threshold=None,
    topic_feedback_irrelevant_article_ids=None,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    resolved_selection_mode = normalize_rerank_selection_mode(rerank_selection_mode)
    resolved_top_n = max(1, int(top_n))

    if resolved_selection_mode == "manual":
        matches = retrieval_search(
            query,
            top_n=resolved_top_n,
            retrieval_model=resolved_model,
            year_start=year_start,
            year_end=year_end,
            character_start=character_start,
            character_end=character_end,
            word_start=word_start,
            word_end=word_end,
            reading_time_start=reading_time_start,
            reading_time_end=reading_time_end,
            words_to_avoid=words_to_avoid,
            topic_feedback_irrelevant_article_ids=topic_feedback_irrelevant_article_ids,
        )
        log_runtime_event(
            "rerank_candidates.manual_done",
            retrieval_model=resolved_model,
            requested_top_n=resolved_top_n,
            selected_count=len(matches),
        )
        return {
            "matches": matches,
            "selection_mode": resolved_selection_mode,
            "candidate_count": len(matches),
            "rerank_threshold": None,
            "empty_results_message": None,
        }

    resolved_threshold = resolve_auto_rerank_threshold(
        rerank_threshold,
        retrieval_model=resolved_model,
    )
    matches = retrieval_search(
        query,
        top_n=MAX_AUTO_RERANK_CANDIDATES,
        retrieval_model=resolved_model,
        year_start=year_start,
        year_end=year_end,
        character_start=character_start,
        character_end=character_end,
        word_start=word_start,
        word_end=word_end,
        reading_time_start=reading_time_start,
        reading_time_end=reading_time_end,
        words_to_avoid=words_to_avoid,
        topic_feedback_irrelevant_article_ids=topic_feedback_irrelevant_article_ids,
    )
    selected_matches = _filter_matches_by_topic_threshold(
        matches,
        threshold=resolved_threshold,
    )[:MAX_AUTO_RERANK_CANDIDATES]
    empty_results_message = None
    if not selected_matches:
        retrieval_label = "SVD" if resolved_model == "svd" else "TF-IDF"
        empty_results_message = (
            f"No relevant articles found above the {resolved_threshold:.2f} "
            f"topic relevance threshold for {retrieval_label}."
        )

    log_runtime_event(
        "rerank_candidates.automatic_done",
        retrieval_model=resolved_model,
        candidate_limit=MAX_AUTO_RERANK_CANDIDATES,
        threshold=resolved_threshold,
        retrieved_count=len(matches),
        selected_count=len(selected_matches),
    )
    return {
        "matches": selected_matches,
        "selection_mode": resolved_selection_mode,
        "candidate_count": len(selected_matches),
        "rerank_threshold": resolved_threshold,
        "empty_results_message": empty_results_message,
    }


def keyword_search(
    query,
    top_n=100,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
    words_to_avoid=None,
    exclude_article_ids=None,
):
    if not query or not query.strip():
        return []

    resolved_query = query.strip()
    if len(resolved_query) < 3:
        return []
    resolved_top_n = max(1, int(top_n))
    resolved_year_start, resolved_year_end = normalize_article_year_range(
        year_start,
        year_end,
    )
    resolved_character_start, resolved_character_end = normalize_article_character_range(
        character_start,
        character_end,
    )
    resolved_word_start, resolved_word_end = normalize_article_word_range(
        word_start,
        word_end,
    )
    resolved_reading_time_start, resolved_reading_time_end = normalize_article_reading_time_range(
        reading_time_start,
        reading_time_end,
    )
    reading_time_word_start, reading_time_word_end = _word_range_for_reading_time_range(
        resolved_reading_time_start,
        resolved_reading_time_end,
    )
    resolved_avoid_words = normalize_avoid_words(words_to_avoid)

    results_query = GuardianArticle.query.filter(
        or_(
            GuardianArticle.title.ilike(f"%{resolved_query}%"),
            GuardianArticle.summary.ilike(f"%{resolved_query}%"),
        )
    )
    if resolved_year_start is not None:
        results_query = results_query.filter(GuardianArticle.year >= resolved_year_start)
    if resolved_year_end is not None:
        results_query = results_query.filter(GuardianArticle.year <= resolved_year_end)
    character_count = _article_character_count_expression()
    if resolved_character_start is not None:
        results_query = results_query.filter(character_count >= resolved_character_start)
    if resolved_character_end is not None:
        results_query = results_query.filter(character_count <= resolved_character_end)
    word_count = _article_word_count_expression()
    if resolved_word_start is not None:
        results_query = results_query.filter(word_count >= resolved_word_start)
    if resolved_word_end is not None:
        results_query = results_query.filter(word_count <= resolved_word_end)
    if reading_time_word_start is not None:
        results_query = results_query.filter(word_count >= reading_time_word_start)
    if reading_time_word_end is not None:
        results_query = results_query.filter(word_count <= reading_time_word_end)
    excluded_ids = normalize_article_id_list(exclude_article_ids)
    if excluded_ids:
        results_query = results_query.filter(~GuardianArticle.id.in_(excluded_ids))

    ordered_query = results_query.order_by(GuardianArticle.date.desc())
    if not resolved_avoid_words:
        results = ordered_query.limit(resolved_top_n).all()
    else:
        results = []
        offset = 0
        batch_size = max(resolved_top_n * 4, resolved_top_n + 20)
        while len(results) < resolved_top_n:
            batch = ordered_query.offset(offset).limit(batch_size).all()
            if not batch:
                break
            filtered_batch = _filter_ranked_articles_by_avoid_words(
                [(article, 0) for article in batch],
                resolved_avoid_words,
            )
            results.extend(article for article, _score in filtered_batch)
            offset += len(batch)

    return [serialize_article(article) for article in results[:resolved_top_n]]


def retrieval_search(
    query,
    top_n=100,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
    words_to_avoid=None,
    topic_feedback_irrelevant_article_ids=None,
):
    if not query or not query.strip():
        return []

    resolved_query = query.strip()
    if len(resolved_query) < 3:
        return []
    resolved_model = normalize_retrieval_model(retrieval_model)
    resolved_top_n = max(1, int(top_n))
    feedback_article_ids = normalize_article_id_list(
        topic_feedback_irrelevant_article_ids
    )
    resolved_year_start, resolved_year_end = normalize_article_year_range(
        year_start,
        year_end,
    )
    resolved_character_start, resolved_character_end = normalize_article_character_range(
        character_start,
        character_end,
    )
    resolved_word_start, resolved_word_end = normalize_article_word_range(
        word_start,
        word_end,
    )
    resolved_reading_time_start, resolved_reading_time_end = normalize_article_reading_time_range(
        reading_time_start,
        reading_time_end,
    )
    reading_time_word_start, reading_time_word_end = _word_range_for_reading_time_range(
        resolved_reading_time_start,
        resolved_reading_time_end,
    )
    resolved_avoid_words = (
        normalize_avoid_words(words_to_avoid)
        if resolved_model == "tfidf"
        else []
    )

    log_runtime_event(
        "retrieval_search.start",
        retrieval_model=resolved_model,
        query_chars=len(resolved_query),
        top_n=resolved_top_n,
        year_start=resolved_year_start,
        year_end=resolved_year_end,
        character_start=resolved_character_start,
        character_end=resolved_character_end,
        word_start=resolved_word_start,
        word_end=resolved_word_end,
        reading_time_start=resolved_reading_time_start,
        reading_time_end=resolved_reading_time_end,
        avoid_word_count=len(resolved_avoid_words),
        rocchio_irrelevant_count=len(feedback_article_ids),
    )
    try:
        processor = build_retrieval_processor(retrieval_model=resolved_model)
    except RuntimeError as exc:
        log_runtime_event(
            "retrieval_search.keyword_fallback",
            retrieval_model=resolved_model,
            reason=str(exc),
            year_start=resolved_year_start,
            year_end=resolved_year_end,
            character_start=resolved_character_start,
            character_end=resolved_character_end,
            word_start=resolved_word_start,
            word_end=resolved_word_end,
            reading_time_start=resolved_reading_time_start,
            reading_time_end=resolved_reading_time_end,
            avoid_word_count=len(resolved_avoid_words),
        )
        return keyword_search(
            resolved_query,
            top_n=resolved_top_n,
            year_start=resolved_year_start,
            year_end=resolved_year_end,
            character_start=resolved_character_start,
            character_end=resolved_character_end,
            word_start=resolved_word_start,
            word_end=resolved_word_end,
            reading_time_start=resolved_reading_time_start,
            reading_time_end=resolved_reading_time_end,
            words_to_avoid=resolved_avoid_words,
            exclude_article_ids=feedback_article_ids,
        )

    if processor is None:
        log_runtime_event(
            "retrieval_search.no_processor",
            retrieval_model=resolved_model,
        )
        return []

    processor_search = build_rocchio_processor_searcher(
        query=resolved_query,
        processor=processor,
        retrieval_model=resolved_model,
        irrelevant_article_ids=feedback_article_ids,
    )
    search_padding = len(feedback_article_ids)

    has_range_filter = bool(resolved_avoid_words) or any(
        value is not None
        for value in (
            resolved_year_start,
            resolved_year_end,
            resolved_character_start,
            resolved_character_end,
            resolved_word_start,
            resolved_word_end,
            resolved_reading_time_start,
            resolved_reading_time_end,
        )
    )

    if not has_range_filter:
        ranked = processor_search(top_n=resolved_top_n + search_padding)
        ranked = _filter_ranked_articles_by_excluded_ids(
            ranked,
            feedback_article_ids,
        )[:resolved_top_n]
    else:
        max_candidates = max(
            resolved_top_n,
            int(getattr(processor, "n_docs", resolved_top_n)),
        )
        search_limit = min(
            max_candidates,
            max((resolved_top_n + search_padding) * 4, resolved_top_n + 20),
        )
        filtered_ranked = []

        while True:
            log_runtime_event(
                "retrieval_search.range_filter_scan",
                retrieval_model=resolved_model,
                year_start=resolved_year_start,
                year_end=resolved_year_end,
                character_start=resolved_character_start,
                character_end=resolved_character_end,
                word_start=resolved_word_start,
                word_end=resolved_word_end,
                reading_time_start=resolved_reading_time_start,
                reading_time_end=resolved_reading_time_end,
                avoid_word_count=len(resolved_avoid_words),
                search_limit=search_limit,
            )
            ranked_batch = processor_search(top_n=search_limit)
            filtered_ranked = _filter_ranked_articles_by_year_range(
                ranked_batch,
                year_start=resolved_year_start,
                year_end=resolved_year_end,
            )
            filtered_ranked = _filter_ranked_articles_by_character_range(
                filtered_ranked,
                character_start=resolved_character_start,
                character_end=resolved_character_end,
            )
            filtered_ranked = _filter_ranked_articles_by_word_range(
                filtered_ranked,
                word_start=resolved_word_start,
                word_end=resolved_word_end,
            )
            filtered_ranked = _filter_ranked_articles_by_word_range(
                filtered_ranked,
                word_start=reading_time_word_start,
                word_end=reading_time_word_end,
            )
            filtered_ranked = _filter_ranked_articles_by_avoid_words(
                filtered_ranked,
                resolved_avoid_words,
            )
            filtered_ranked = _filter_ranked_articles_by_excluded_ids(
                filtered_ranked,
                feedback_article_ids,
            )
            if (
                len(filtered_ranked) >= resolved_top_n
                or search_limit >= max_candidates
                or len(ranked_batch) < search_limit
            ):
                break

            next_limit = min(max_candidates, max(search_limit * 2, resolved_top_n))
            if next_limit == search_limit:
                break
            search_limit = next_limit

        ranked = filtered_ranked[:resolved_top_n]
        log_runtime_event(
            "retrieval_search.range_filter_done",
            retrieval_model=resolved_model,
            year_start=resolved_year_start,
            year_end=resolved_year_end,
            character_start=resolved_character_start,
            character_end=resolved_character_end,
            word_start=resolved_word_start,
            word_end=resolved_word_end,
            reading_time_start=resolved_reading_time_start,
            reading_time_end=resolved_reading_time_end,
            avoid_word_count=len(resolved_avoid_words),
            filtered_count=len(ranked),
        )

    log_runtime_event(
        "retrieval_search.done",
        retrieval_model=resolved_model,
        result_count=len(ranked),
    )
    return build_matches(
        ranked,
        retrieval_model=resolved_model,
        processor=processor,
    )


def tfidf_cos_search(
    query,
    top_n=100,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
    words_to_avoid=None,
):
    return retrieval_search(
        query,
        top_n=top_n,
        retrieval_model="tfidf",
        year_start=year_start,
        year_end=year_end,
        character_start=character_start,
        character_end=character_end,
        word_start=word_start,
        word_end=word_end,
        reading_time_start=reading_time_start,
        reading_time_end=reading_time_end,
        words_to_avoid=words_to_avoid,
    )


def svd_search(
    query,
    top_n=100,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
    words_to_avoid=None,
):
    return retrieval_search(
        query,
        top_n=top_n,
        retrieval_model="svd",
        year_start=year_start,
        year_end=year_end,
        character_start=character_start,
        character_end=character_end,
        word_start=word_start,
        word_end=word_end,
        reading_time_start=reading_time_start,
        reading_time_end=reading_time_end,
        words_to_avoid=words_to_avoid,
    )


def similar_svd_articles(
    article_id,
    limit=5,
    offset=0,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
):
    source_id = str(article_id or "").strip()
    if not source_id:
        raise ValueError("An article_id is required.")

    resolved_limit = max(1, min(25, int(limit)))
    resolved_offset = max(0, int(offset))
    resolved_year_start, resolved_year_end = normalize_article_year_range(
        year_start,
        year_end,
    )
    resolved_character_start, resolved_character_end = normalize_article_character_range(
        character_start,
        character_end,
    )
    resolved_word_start, resolved_word_end = normalize_article_word_range(
        word_start,
        word_end,
    )
    resolved_reading_time_start, resolved_reading_time_end = normalize_article_reading_time_range(
        reading_time_start,
        reading_time_end,
    )
    reading_time_word_start, reading_time_word_end = _word_range_for_reading_time_range(
        resolved_reading_time_start,
        resolved_reading_time_end,
    )

    try:
        processor = build_retrieval_processor(retrieval_model="svd")
    except RuntimeError as exc:
        raise ValueError("SVD similar-article search is not available yet.") from exc

    try:
        source_idx = processor.get_doc_idx_by_id(source_id)
        source_vector = processor.get_doc_vector(source_id, normalize=True)
    except Exception as exc:
        raise ValueError("That article is not available in the SVD index.") from exc

    scores = np.asarray(
        processor.normalized_doc_embeddings @ source_vector,
        dtype=np.float32,
    ).reshape(-1)
    ranked_indices = np.argsort(scores)[::-1]

    ranked = []
    for raw_idx in ranked_indices:
        idx = int(raw_idx)
        if idx == source_idx:
            continue
        score = float(scores[idx])
        if score <= 0:
            continue
        ranked.append((processor.doc_ids[idx], score))

    if resolved_year_start is not None or resolved_year_end is not None:
        ranked = _filter_ranked_articles_by_year_range(
            ranked,
            year_start=resolved_year_start,
            year_end=resolved_year_end,
        )
    if resolved_character_start is not None or resolved_character_end is not None:
        ranked = _filter_ranked_articles_by_character_range(
            ranked,
            character_start=resolved_character_start,
            character_end=resolved_character_end,
        )
    if resolved_word_start is not None or resolved_word_end is not None:
        ranked = _filter_ranked_articles_by_word_range(
            ranked,
            word_start=resolved_word_start,
            word_end=resolved_word_end,
        )
    if reading_time_word_start is not None or reading_time_word_end is not None:
        ranked = _filter_ranked_articles_by_word_range(
            ranked,
            word_start=reading_time_word_start,
            word_end=reading_time_word_end,
        )

    page_ranked = ranked[resolved_offset:resolved_offset + resolved_limit + 1]
    has_more = len(page_ranked) > resolved_limit
    page_ranked = page_ranked[:resolved_limit]
    results = build_matches(
        page_ranked,
        retrieval_model="svd",
        processor=processor,
    )

    return {
        "source_article_id": source_id,
        "results": results,
        "next_offset": resolved_offset + len(results),
        "has_more": has_more,
    }


def retrieval_query_svd_dimensions(query, retrieval_model=DEFAULT_RETRIEVAL_MODEL):
    resolved_model = normalize_retrieval_model(retrieval_model)
    if resolved_model != "svd":
        return []

    resolved_query = str(query or "").strip()
    if len(resolved_query) < 3:
        return []

    try:
        processor = build_retrieval_processor(retrieval_model=resolved_model)
    except RuntimeError:
        return []

    return query_svd_dimensions(
        resolved_query,
        retrieval_model=resolved_model,
        processor=processor,
    )


def retrieval_query_svd_corpus_chart_dimensions(
    query,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    if resolved_model != "svd":
        return []

    resolved_query = str(query or "").strip()
    if len(resolved_query) < 3:
        return []

    try:
        processor = build_retrieval_processor(retrieval_model=resolved_model)
    except RuntimeError:
        return []

    return query_svd_corpus_chart_dimensions(
        resolved_query,
        retrieval_model=resolved_model,
        processor=processor,
    )


def retrieval_query_typo_suggestion(query, retrieval_model=DEFAULT_RETRIEVAL_MODEL):
    resolved_query = str(query or "").strip()
    if len(resolved_query) < 2:
        return None

    resolved_model = normalize_retrieval_model(retrieval_model)
    try:
        processor = build_retrieval_processor(retrieval_model=resolved_model)
    except RuntimeError:
        return None

    return build_query_typo_suggestion(resolved_query, processor)


def attach_query_svd_chart_dimensions(
    article_matches,
    query_dimensions,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    query=None,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    if resolved_model != "svd":
        return article_matches
    if not article_matches or not query_dimensions:
        return article_matches

    try:
        processor = build_retrieval_processor(retrieval_model=resolved_model)
    except RuntimeError:
        return article_matches

    return _attach_query_svd_chart_dimensions(
        article_matches,
        query_dimensions=query_dimensions,
        retrieval_model=resolved_model,
        processor=processor,
        query=query,
    )


def json_search(
    query,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    top_n=100,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
    words_to_avoid=None,
    topic_feedback_irrelevant_article_ids=None,
):
    """
    Primary search used by /api/articles and optional LLM retrieval.
    Defaults to TF-IDF cosine search, falls back to keyword SQL search.
    """
    resolved_model = normalize_retrieval_model(retrieval_model)
    try:
        log_runtime_event(
            "json_search.try_retrieval",
            retrieval_model=resolved_model,
            query_chars=len(str(query or "").strip()),
        )
        return retrieval_search(
            query,
            top_n=top_n,
            retrieval_model=resolved_model,
            year_start=year_start,
            year_end=year_end,
            character_start=character_start,
            character_end=character_end,
            word_start=word_start,
            word_end=word_end,
            reading_time_start=reading_time_start,
            reading_time_end=reading_time_end,
            words_to_avoid=words_to_avoid,
            topic_feedback_irrelevant_article_ids=topic_feedback_irrelevant_article_ids,
        )
    except Exception:
        log_runtime_event(
            "json_search.fallback_keyword",
            retrieval_model=resolved_model,
        )
        return keyword_search(
            query,
            top_n=top_n,
            year_start=year_start,
            year_end=year_end,
            character_start=character_start,
            character_end=character_end,
            word_start=word_start,
            word_end=word_end,
            reading_time_start=reading_time_start,
            reading_time_end=reading_time_end,
            words_to_avoid=words_to_avoid if resolved_model == "tfidf" else None,
            exclude_article_ids=topic_feedback_irrelevant_article_ids,
        )


def stance_search(
    topic,
    opinion,
    topic_weight=0.4,
    stance_weight=0.4,
    recency_weight=0.2,
    top_n=20,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_start=None,
    reading_time_end=None,
    words_to_avoid=None,
    normalize_topic_scores=False,
    rerank_selection_mode=DEFAULT_RERANK_SELECTION_MODE,
    rerank_threshold=None,
    topic_feedback_irrelevant_article_ids=None,
    stance_method="nli",
    use_chunking=False,
    chunking_mode="none",
):
    from backend.stance_processing.stance_rerank import rerank_article_matches

    topic_text = str(topic or "").strip()
    opinion_text = str(opinion or "").strip()
    if len(topic_text) < 2 or len(opinion_text) < 2:
        return {
            "results": [],
            "empty_results_message": None,
        }
    resolved_model = normalize_retrieval_model(retrieval_model)
    resolved_selection_mode = normalize_rerank_selection_mode(rerank_selection_mode)

    candidate_payload = select_rerank_candidates(
        query=topic_text,
        top_n=top_n,
        retrieval_model=resolved_model,
        year_start=year_start,
        year_end=year_end,
        character_start=character_start,
        character_end=character_end,
        word_start=word_start,
        word_end=word_end,
        reading_time_start=reading_time_start,
        reading_time_end=reading_time_end,
        words_to_avoid=words_to_avoid,
        rerank_selection_mode=resolved_selection_mode,
        rerank_threshold=rerank_threshold,
        topic_feedback_irrelevant_article_ids=topic_feedback_irrelevant_article_ids,
    )
    topic_matches = candidate_payload["matches"]
    if not topic_matches:
        log_runtime_event(
            "stance_search.no_topic_matches",
            retrieval_model=resolved_model,
            rerank_selection_mode=resolved_selection_mode,
            rerank_threshold=candidate_payload.get("rerank_threshold"),
            empty_results_message=candidate_payload.get("empty_results_message"),
        )
        return {
            "results": [],
            "empty_results_message": candidate_payload.get("empty_results_message"),
        }

    log_runtime_event(
        "stance_search.rerank_start",
        retrieval_model=resolved_model,
        topic_chars=len(topic_text),
        opinion_chars=len(opinion_text),
        top_n=len(topic_matches),
        normalize_topic_scores=bool(normalize_topic_scores),
        rerank_selection_mode=resolved_selection_mode,
        rerank_threshold=candidate_payload.get("rerank_threshold"),
        rocchio_irrelevant_count=len(normalize_article_id_list(topic_feedback_irrelevant_article_ids)),
        stance_method=stance_method,
        use_chunking=bool(use_chunking),
        chunking_mode=chunking_mode,
    )
    reranked = rerank_article_matches(
        article_matches=topic_matches,
        topic=topic_text,
        opinion=opinion_text,
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        recency_weight=recency_weight,
        top_n=len(topic_matches),
        normalize_topic_scores=normalize_topic_scores,
        stance_method=stance_method,
        use_chunking=use_chunking,
        chunking_mode=chunking_mode,
    )
    return {
        "results": reranked,
        "empty_results_message": candidate_payload.get("empty_results_message"),
    }
