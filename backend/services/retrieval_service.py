import numpy as np
from sqlalchemy import or_

from backend.db.models import GuardianArticle
from backend.runtime.runtime_debug import log_runtime_event
from backend.services.chunk_retrieval_service import (
    DEFAULT_CHUNK_ARTICLE_TOP_K,
    DEFAULT_CHUNK_AUTO_THRESHOLDS,
    DEFAULT_CHUNK_CANDIDATE_TOP_K,
    MAX_CHUNK_CANDIDATE_TOP_K,
    chunk_retrieval_search,
    default_chunk_auto_threshold,
    normalize_chunk_article_top_k,
    normalize_chunk_candidate_top_k,
)
from backend.services.filters.article_filters import (
    available_article_character_range,
    available_article_reading_time_range,
    available_article_word_range,
    available_article_year_range,
    filter_query_by_article_ranges,
    filter_ranked_articles_by_excluded_ids as _filter_ranked_articles_by_excluded_ids,
    filter_ranked_articles_by_metadata_ranges as _filter_ranked_articles_by_metadata_ranges,
    normalize_article_character_range,
    normalize_article_reading_time_range,
    normalize_article_word_range,
    normalize_article_year_range,
    word_range_for_reading_time_range as _word_range_for_reading_time_range,
)
from backend.services.filters.text_filters import (
    filter_ranked_articles_by_avoid_words as _filter_ranked_articles_by_avoid_words,
    normalize_avoid_words,
)
from backend.services.rocchio_feedback import (
    build_rocchio_processor_searcher,
    normalize_article_id_list,
)
from backend.stance_processing.stance_rerank import DEFAULT_STANCE_METHOD
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
from backend.text_processing.indexing.dense_search import top_positive_dot_candidates
from backend.text_processing.typo_correction import build_query_typo_suggestion


SUPPORTED_RETRIEVAL_MODELS = _SUPPORTED_RETRIEVAL_MODELS
SUPPORTED_RERANK_SELECTION_MODES = ("manual", "automatic")
DEFAULT_RERANK_SELECTION_MODE = "automatic"
DEFAULT_AUTO_RERANK_THRESHOLDS = {
    "tfidf": 0.3,
    "svd": 0.6,
    "minilm": 0.4,
}
MAX_AUTO_RERANK_CANDIDATES = 100


def _coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    use_chunking=False,
    chunking_mode="none",
    chunk_candidate_top_k=DEFAULT_CHUNK_CANDIDATE_TOP_K,
    chunk_article_top_k=DEFAULT_CHUNK_ARTICLE_TOP_K,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    resolved_selection_mode = normalize_rerank_selection_mode(rerank_selection_mode)
    resolved_top_n = max(1, int(top_n))
    resolved_use_chunking = bool(use_chunking) and str(chunking_mode or "none") != "none"
    if resolved_use_chunking and resolved_model == "tfidf":
        log_runtime_event(
            "rerank_candidates.chunk_force_svd",
            requested_retrieval_model=resolved_model,
        )
        resolved_model = "svd"

    if resolved_use_chunking:
        resolved_reading_time_start, resolved_reading_time_end = normalize_article_reading_time_range(
            reading_time_start,
            reading_time_end,
        )
        reading_time_word_start, reading_time_word_end = _word_range_for_reading_time_range(
            resolved_reading_time_start,
            resolved_reading_time_end,
        )
        chunk_threshold = (
            resolve_auto_rerank_threshold(
                rerank_threshold,
                retrieval_model=resolved_model,
            )
            if resolved_selection_mode == "automatic" and rerank_threshold is not None
            else (
                default_chunk_auto_threshold(resolved_model)
                if resolved_selection_mode == "automatic"
                else None
            )
        )
        payload = chunk_retrieval_search(
            query=query,
            top_n=resolved_top_n,
            retrieval_model=resolved_model,
            chunking_mode=chunking_mode,
            rerank_selection_mode=resolved_selection_mode,
            rerank_threshold=chunk_threshold,
            chunk_candidate_top_k=chunk_candidate_top_k,
            chunk_article_top_k=chunk_article_top_k,
            year_start=year_start,
            year_end=year_end,
            character_start=character_start,
            character_end=character_end,
            word_start=word_start,
            word_end=word_end,
            reading_time_word_start=reading_time_word_start,
            reading_time_word_end=reading_time_word_end,
            words_to_avoid=words_to_avoid,
            topic_feedback_irrelevant_article_ids=topic_feedback_irrelevant_article_ids,
        )
        log_runtime_event(
            "rerank_candidates.chunk_done",
            retrieval_model=resolved_model,
            selection_mode=resolved_selection_mode,
            chunking_mode=chunking_mode,
            selected_count=len(payload.get("matches") or []),
            chunk_candidate_count=payload.get("chunk_candidate_count"),
            chunk_candidate_top_k=normalize_chunk_candidate_top_k(chunk_candidate_top_k),
            chunk_article_top_k=normalize_chunk_article_top_k(chunk_article_top_k),
            threshold=payload.get("rerank_threshold"),
        )
        return payload

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
    auto_candidate_limit = MAX_AUTO_RERANK_CANDIDATES
    auto_selected_limit = min(MAX_AUTO_RERANK_CANDIDATES, resolved_top_n)
    matches = retrieval_search(
        query,
        top_n=auto_candidate_limit,
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
    )[:auto_selected_limit]
    empty_results_message = None
    if not selected_matches:
        retrieval_label = (
            "SVD" if resolved_model == "svd"
            else ("Enhanced Semantic" if resolved_model == "minilm" else "TF-IDF")
        )
        empty_results_message = (
            f"No relevant articles found above the {resolved_threshold:.2f} "
            f"topic relevance threshold for {retrieval_label}."
        )

    log_runtime_event(
        "rerank_candidates.automatic_done",
        retrieval_model=resolved_model,
        candidate_limit=auto_candidate_limit,
        selected_limit=auto_selected_limit,
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
    results_query = filter_query_by_article_ranges(
        results_query,
        year_start=resolved_year_start,
        year_end=resolved_year_end,
        character_start=resolved_character_start,
        character_end=resolved_character_end,
        word_start=resolved_word_start,
        word_end=resolved_word_end,
        reading_time_word_start=reading_time_word_start,
        reading_time_word_end=reading_time_word_end,
    )
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
            filtered_ranked = _filter_ranked_articles_by_metadata_ranges(
                ranked_batch,
                year_start=resolved_year_start,
                year_end=resolved_year_end,
                character_start=resolved_character_start,
                character_end=resolved_character_end,
                word_start=resolved_word_start,
                word_end=resolved_word_end,
                reading_time_word_start=reading_time_word_start,
                reading_time_word_end=reading_time_word_end,
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


def similar_articles(
    article_id,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    limit=10,
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
    """Find articles similar to ``article_id`` using the cosine-similarity
    notion of the requested ``retrieval_model``.

    - ``svd`` / ``minilm``: dense cosine over the model's article embeddings
      (using ``processor.normalized_doc_embeddings`` + ``get_doc_vector``).
    - ``tfidf``: sparse cosine via re-vectorizing the source article's body
      text and running the standard postings-index search.
    """
    resolved_model = normalize_retrieval_model(retrieval_model)
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

    eligible_ids = _collect_similar_eligibility(
        year_start=resolved_year_start,
        year_end=resolved_year_end,
        character_start=resolved_character_start,
        character_end=resolved_character_end,
        word_start=resolved_word_start,
        word_end=resolved_word_end,
        reading_time_word_start=reading_time_word_start,
        reading_time_word_end=reading_time_word_end,
    )

    if resolved_model == "tfidf":
        return _similar_articles_tfidf(
            source_id=source_id,
            eligible_ids=eligible_ids,
            limit=resolved_limit,
            offset=resolved_offset,
        )
    return _similar_articles_dense(
        source_id=source_id,
        retrieval_model=resolved_model,
        eligible_ids=eligible_ids,
        limit=resolved_limit,
        offset=resolved_offset,
    )


def similar_svd_articles(*args, **kwargs):
    """Back-compat wrapper — always uses the SVD index."""
    kwargs.pop("retrieval_model", None)
    return similar_articles(*args, retrieval_model="svd", **kwargs)


def _collect_similar_eligibility(
    *,
    year_start,
    year_end,
    character_start,
    character_end,
    word_start,
    word_end,
    reading_time_word_start,
    reading_time_word_end,
):
    has_metadata_filters = any(
        value is not None
        for value in (
            year_start,
            year_end,
            character_start,
            character_end,
            word_start,
            word_end,
            reading_time_word_start,
            reading_time_word_end,
        )
    )
    if not has_metadata_filters:
        return None
    eligible_query = GuardianArticle.query.with_entities(GuardianArticle.id)
    if year_start is not None:
        eligible_query = eligible_query.filter(GuardianArticle.year >= year_start)
    if year_end is not None:
        eligible_query = eligible_query.filter(GuardianArticle.year <= year_end)
    if character_start is not None:
        eligible_query = eligible_query.filter(
            GuardianArticle.body_character_count >= character_start
        )
    if character_end is not None:
        eligible_query = eligible_query.filter(
            GuardianArticle.body_character_count <= character_end
        )
    if word_start is not None:
        eligible_query = eligible_query.filter(
            GuardianArticle.body_word_count >= word_start
        )
    if word_end is not None:
        eligible_query = eligible_query.filter(
            GuardianArticle.body_word_count <= word_end
        )
    if reading_time_word_start is not None:
        eligible_query = eligible_query.filter(
            GuardianArticle.body_word_count >= reading_time_word_start
        )
    if reading_time_word_end is not None:
        eligible_query = eligible_query.filter(
            GuardianArticle.body_word_count <= reading_time_word_end
        )
    return {str(row[0]) for row in eligible_query.all()}


def _similar_articles_dense(
    *,
    source_id,
    retrieval_model,
    eligible_ids,
    limit,
    offset,
):
    try:
        processor = build_retrieval_processor(retrieval_model=retrieval_model)
    except RuntimeError as exc:
        raise ValueError(
            f"{retrieval_model.upper()} similar-article search is not available yet."
        ) from exc

    try:
        source_idx = processor.get_doc_idx_by_id(source_id)
        source_vector = processor.get_doc_vector(source_id, normalize=True)
    except Exception as exc:
        raise ValueError(
            f"That article is not available in the {retrieval_model.upper()} index."
        ) from exc

    def keep_candidate_indices(indices):
        if eligible_ids is None:
            return indices != source_idx
        return np.asarray(
            [
                int(idx) != source_idx
                and str(processor.doc_ids[int(idx)]) in eligible_ids
                for idx in indices
            ],
            dtype=bool,
        )

    candidate_indices, candidate_scores = top_positive_dot_candidates(
        processor.normalized_doc_embeddings,
        source_vector,
        top_n=offset + limit + 1,
        candidate_filter=keep_candidate_indices,
    )
    page_slice = slice(offset, offset + limit + 1)
    page_indices = candidate_indices[page_slice]
    page_scores = candidate_scores[page_slice]
    has_more = len(page_indices) > limit
    page_indices = page_indices[:limit]
    page_scores = page_scores[:limit]
    page_ranked = [
        (processor.doc_ids[int(idx)], float(score))
        for idx, score in zip(page_indices, page_scores)
    ]
    results = build_matches(
        page_ranked,
        retrieval_model=retrieval_model,
        processor=processor,
    )

    return {
        "source_article_id": source_id,
        "results": results,
        "next_offset": offset + len(results),
        "has_more": has_more,
    }


def _similar_articles_tfidf(*, source_id, eligible_ids, limit, offset):
    """TF-IDF doc-to-doc cosine: re-vectorize the source article's body text
    and run the standard postings-index search. Slightly different semantics
    from a "true" doc/doc cosine because the postings score is the standard
    TF-IDF query/doc dot product, but for this corpus and use case (one
    Guardian article vs. all others) the ranking is materially identical.
    """
    try:
        processor = build_retrieval_processor(retrieval_model="tfidf")
    except RuntimeError as exc:
        raise ValueError("TF-IDF similar-article search is not available yet.") from exc

    source_article = GuardianArticle.query.filter_by(id=source_id).first()
    if source_article is None or not (source_article.body_text or "").strip():
        raise ValueError("That article is not available in the TF-IDF index.")

    raw_results = processor.search(
        source_article.body_text,
        top_n=offset + limit + 32,
        return_articles=False,
    )
    if not raw_results:
        return {
            "source_article_id": source_id,
            "results": [],
            "next_offset": offset,
            "has_more": False,
        }

    filtered_pairs = []
    for entry in raw_results:
        if isinstance(entry, dict):
            doc_id = str(entry.get("doc_id") or entry.get("id") or "")
            score = float(entry.get("score") or 0.0)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            doc_id = str(entry[0])
            score = float(entry[1])
        else:
            continue
        if not doc_id or doc_id == source_id:
            continue
        if eligible_ids is not None and doc_id not in eligible_ids:
            continue
        filtered_pairs.append((doc_id, score))

    page_slice = slice(offset, offset + limit + 1)
    page = filtered_pairs[page_slice]
    has_more = len(page) > limit
    page = page[:limit]

    results = build_matches(
        page,
        retrieval_model="tfidf",
        processor=processor,
    )
    return {
        "source_article_id": source_id,
        "results": results,
        "next_offset": offset + len(results),
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
    stance_method=DEFAULT_STANCE_METHOD,
    use_chunking=False,
    chunking_mode="none",
    chunk_candidate_top_k=DEFAULT_CHUNK_CANDIDATE_TOP_K,
    chunk_article_top_k=DEFAULT_CHUNK_ARTICLE_TOP_K,
    llm_label_irrelevant=True,
    progress_callback=None,
    on_topic_done=None,
):
    """on_topic_done is invoked with the topic-relevance candidate list as
    soon as that step finishes (before the slower stance-scoring/rerank).
    Lets the API layer publish those candidates via SSE so the frontend can
    paint stage 2 with scattered cards while reranking is still running."""
    from backend.stance_processing.stance_rerank import rerank_article_matches

    topic_text = str(topic or "").strip()
    opinion_text = str(opinion or "").strip()
    if len(topic_text) < 2 or len(opinion_text) < 2:
        if progress_callback:
            progress_callback("complete", "Search complete", 1.0, result_count=0)
        return {
            "results": [],
            "empty_results_message": None,
        }
    resolved_model = normalize_retrieval_model(retrieval_model)
    if use_chunking and str(chunking_mode or "none") != "none" and resolved_model == "tfidf":
        resolved_model = "svd"
    resolved_selection_mode = normalize_rerank_selection_mode(rerank_selection_mode)

    if progress_callback:
        progress_callback("topic", "Scoring topic relevance", 0.08)
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
        use_chunking=use_chunking,
        chunking_mode=chunking_mode,
        chunk_candidate_top_k=chunk_candidate_top_k,
        chunk_article_top_k=chunk_article_top_k,
    )
    topic_matches = candidate_payload["matches"]
    if progress_callback:
        progress_callback(
            "topic",
            "Topic relevance scored",
            0.38,
            candidate_count=len(topic_matches),
        )
    # Hand the topic-only candidates to the caller before the heavy
    # rerank starts — lets the UI paint stage 2 immediately.
    if on_topic_done is not None and topic_matches:
        try:
            on_topic_done(list(topic_matches))
        except Exception:  # noqa: BLE001 — best-effort, never block the search
            pass
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
    if progress_callback:
        scorer_label = "LLM" if stance_method == "llm" or use_chunking else "NLI model"
        progress_callback(
            "agreement",
            f"Scoring stance agreement with {scorer_label}",
            0.45,
            candidate_count=len(topic_matches),
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
        llm_label_irrelevant=llm_label_irrelevant,
        progress_callback=progress_callback,
    )
    if progress_callback:
        progress_callback("ranking", "Finalizing ranking", 0.86, result_count=len(reranked))
    return {
        "results": reranked,
        "empty_results_message": candidate_payload.get("empty_results_message"),
    }
