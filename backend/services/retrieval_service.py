from sqlalchemy import func, or_

from backend.db.models import GuardianArticle
from backend.runtime.runtime_debug import log_runtime_event
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


SUPPORTED_RETRIEVAL_MODELS = _SUPPORTED_RETRIEVAL_MODELS
SUPPORTED_RERANK_SELECTION_MODES = ("manual", "automatic")
DEFAULT_RERANK_SELECTION_MODE = "automatic"
DEFAULT_AUTO_RERANK_THRESHOLDS = {
    "tfidf": 0.3,
    "svd": 0.6,
}
MAX_AUTO_RERANK_CANDIDATES = 100


def _coerce_year(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _ranked_article_year(article, year_lookup):
    if isinstance(article, str):
        return _coerce_year(year_lookup.get(article))
    if isinstance(article, dict):
        return _coerce_year(article.get("year"))
    return _coerce_year(getattr(article, "year", None))


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
    rerank_selection_mode=DEFAULT_RERANK_SELECTION_MODE,
    rerank_threshold=None,
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


def keyword_search(query, top_n=100, year_start=None, year_end=None):
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

    results = (
        results_query
        .order_by(GuardianArticle.date.desc())
        .limit(resolved_top_n)
        .all()
    )

    return [serialize_article(article) for article in results]


def retrieval_search(
    query,
    top_n=100,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    year_start=None,
    year_end=None,
):
    if not query or not query.strip():
        return []

    resolved_query = query.strip()
    if len(resolved_query) < 3:
        return []
    resolved_model = normalize_retrieval_model(retrieval_model)
    resolved_top_n = max(1, int(top_n))
    resolved_year_start, resolved_year_end = normalize_article_year_range(
        year_start,
        year_end,
    )

    log_runtime_event(
        "retrieval_search.start",
        retrieval_model=resolved_model,
        query_chars=len(resolved_query),
        top_n=resolved_top_n,
        year_start=resolved_year_start,
        year_end=resolved_year_end,
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
        )
        return keyword_search(
            resolved_query,
            top_n=resolved_top_n,
            year_start=resolved_year_start,
            year_end=resolved_year_end,
        )

    if processor is None:
        log_runtime_event(
            "retrieval_search.no_processor",
            retrieval_model=resolved_model,
        )
        return []

    if resolved_year_start is None and resolved_year_end is None:
        ranked = processor.search(resolved_query, top_n=resolved_top_n)
    else:
        max_candidates = max(
            resolved_top_n,
            int(getattr(processor, "n_docs", resolved_top_n)),
        )
        search_limit = min(max_candidates, max(resolved_top_n * 4, resolved_top_n + 20))
        filtered_ranked = []

        while True:
            log_runtime_event(
                "retrieval_search.year_filter_scan",
                retrieval_model=resolved_model,
                year_start=resolved_year_start,
                year_end=resolved_year_end,
                search_limit=search_limit,
            )
            ranked_batch = processor.search(resolved_query, top_n=search_limit)
            filtered_ranked = _filter_ranked_articles_by_year_range(
                ranked_batch,
                year_start=resolved_year_start,
                year_end=resolved_year_end,
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
            "retrieval_search.year_filter_done",
            retrieval_model=resolved_model,
            year_start=resolved_year_start,
            year_end=resolved_year_end,
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


def tfidf_cos_search(query, top_n=100, year_start=None, year_end=None):
    return retrieval_search(
        query,
        top_n=top_n,
        retrieval_model="tfidf",
        year_start=year_start,
        year_end=year_end,
    )


def svd_search(query, top_n=100, year_start=None, year_end=None):
    return retrieval_search(
        query,
        top_n=top_n,
        retrieval_model="svd",
        year_start=year_start,
        year_end=year_end,
    )


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


def attach_query_svd_chart_dimensions(
    article_matches,
    query_dimensions,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
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
    )


def json_search(
    query,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    top_n=100,
    year_start=None,
    year_end=None,
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
    normalize_topic_scores=False,
    rerank_selection_mode=DEFAULT_RERANK_SELECTION_MODE,
    rerank_threshold=None,
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
        rerank_selection_mode=resolved_selection_mode,
        rerank_threshold=rerank_threshold,
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
    )
    return {
        "results": reranked,
        "empty_results_message": candidate_payload.get("empty_results_message"),
    }
