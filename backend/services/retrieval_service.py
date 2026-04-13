from sqlalchemy import or_

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


def keyword_search(query):
    if not query or not query.strip():
        return []

    resolved_query = query.strip()
    if len(resolved_query) < 3:
        return []

    results = (
        GuardianArticle.query.filter(
            or_(
                GuardianArticle.title.ilike(f"%{resolved_query}%"),
                GuardianArticle.summary.ilike(f"%{resolved_query}%"),
            )
        )
        .order_by(GuardianArticle.date.desc())
        .limit(100)
        .all()
    )

    return [serialize_article(article) for article in results]


def retrieval_search(query, top_n=100, retrieval_model=DEFAULT_RETRIEVAL_MODEL):
    if not query or not query.strip():
        return []

    resolved_query = query.strip()
    if len(resolved_query) < 3:
        return []
    resolved_model = normalize_retrieval_model(retrieval_model)

    log_runtime_event(
        "retrieval_search.start",
        retrieval_model=resolved_model,
        query_chars=len(resolved_query),
        top_n=int(top_n),
    )
    try:
        processor = build_retrieval_processor(retrieval_model=resolved_model)
    except RuntimeError as exc:
        log_runtime_event(
            "retrieval_search.keyword_fallback",
            retrieval_model=resolved_model,
            reason=str(exc),
        )
        return keyword_search(resolved_query)

    if processor is None:
        log_runtime_event(
            "retrieval_search.no_processor",
            retrieval_model=resolved_model,
        )
        return []

    ranked = processor.search(resolved_query, top_n=top_n)
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


def tfidf_cos_search(query, top_n=100):
    return retrieval_search(query, top_n=top_n, retrieval_model="tfidf")


def svd_search(query, top_n=100):
    return retrieval_search(query, top_n=top_n, retrieval_model="svd")


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


def json_search(query, retrieval_model=DEFAULT_RETRIEVAL_MODEL, top_n=100):
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
        )
    except Exception:
        log_runtime_event(
            "json_search.fallback_keyword",
            retrieval_model=resolved_model,
        )
        return keyword_search(query)


def stance_search(
    topic,
    opinion,
    topic_weight=0.5,
    stance_weight=0.5,
    top_n=20,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
):
    from backend.stance_processing.stance_rerank import rerank_article_matches

    topic_text = str(topic or "").strip()
    opinion_text = str(opinion or "").strip()
    if len(topic_text) < 2 or len(opinion_text) < 2:
        return []
    resolved_model = normalize_retrieval_model(retrieval_model)

    topic_matches = retrieval_search(
        topic_text,
        top_n=top_n,
        retrieval_model=resolved_model,
    )
    if not topic_matches:
        log_runtime_event(
            "stance_search.no_topic_matches",
            retrieval_model=resolved_model,
        )
        return []

    log_runtime_event(
        "stance_search.rerank_start",
        retrieval_model=resolved_model,
        topic_chars=len(topic_text),
        opinion_chars=len(opinion_text),
        top_n=int(top_n),
    )
    return rerank_article_matches(
        article_matches=topic_matches,
        topic=topic_text,
        opinion=opinion_text,
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        top_n=top_n,
    )
