from sqlalchemy import or_

from backend.db.models import GuardianArticle
from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.search_helpers import (
    build_matches,
    build_vector_processor,
    serialize_article,
)


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


def tfidf_cos_search(query, top_n=100):
    if not query or not query.strip():
        return []

    resolved_query = query.strip()
    if len(resolved_query) < 3:
        return []

    log_runtime_event(
        "tfidf_cos_search.start",
        query_chars=len(resolved_query),
        top_n=int(top_n),
    )
    try:
        processor = build_vector_processor()
    except RuntimeError as exc:
        log_runtime_event(
            "tfidf_cos_search.keyword_fallback",
            reason=str(exc),
        )
        return keyword_search(resolved_query)

    if processor is None:
        log_runtime_event("tfidf_cos_search.no_processor")
        return []

    ranked = processor.search(resolved_query, top_n=top_n)
    log_runtime_event("tfidf_cos_search.done", result_count=len(ranked))
    return build_matches(ranked)


def json_search(query):
    """
    Primary search used by /api/articles and optional LLM retrieval.
    Defaults to TF-IDF cosine search, falls back to keyword SQL search.
    """
    try:
        log_runtime_event(
            "json_search.try_tfidf",
            query_chars=len(str(query or "").strip()),
        )
        return tfidf_cos_search(query)
    except Exception:
        log_runtime_event("json_search.fallback_keyword")
        return keyword_search(query)


def stance_search(topic, opinion, topic_weight=0.5, stance_weight=0.5, top_n=20):
    from backend.stance_processing.stance_rerank import rerank_article_matches

    topic_text = str(topic or "").strip()
    opinion_text = str(opinion or "").strip()
    if len(topic_text) < 2 or len(opinion_text) < 2:
        return []

    topic_matches = tfidf_cos_search(topic_text, top_n=top_n)
    if not topic_matches:
        log_runtime_event("stance_search.no_topic_matches")
        return []

    log_runtime_event(
        "stance_search.rerank_start",
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
