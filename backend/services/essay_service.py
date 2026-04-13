from backend.runtime.runtime_debug import log_runtime_event
from backend.services.retrieval_service import (
    DEFAULT_RETRIEVAL_MODEL,
    normalize_retrieval_model,
    retrieval_search,
)
from backend.stance_processing.nli_processor import score_claim_sentences
from backend.stance_processing.stance_rerank import (
    rerank_article_matches_by_statement,
)
from backend.text_processing.sentence_splitter import sentence_rows_from_text


def essay_claim_candidates(essay_text, top_n=5):
    resolved_text = str(essay_text or "").strip()
    if len(resolved_text) < 3:
        return {
            "essay_text": resolved_text,
            "sentence_count": 0,
            "candidates": [],
        }

    sentence_rows = sentence_rows_from_text(resolved_text, prefix="essay_s")
    candidates = score_claim_sentences(sentence_rows, top_n=top_n)
    return {
        "essay_text": resolved_text,
        "sentence_count": len(sentence_rows),
        "candidates": candidates,
    }


def essay_search(
    essay_text,
    selected_thesis_sentence,
    selected_thesis_id=None,
    topic_weight=0.4,
    stance_weight=0.4,
    recency_weight=0.2,
    top_n=20,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    year_start=None,
    year_end=None,
    normalize_topic_scores=False,
):
    resolved_essay = str(essay_text or "").strip()
    resolved_thesis = str(selected_thesis_sentence or "").strip()
    if len(resolved_essay) < 3:
        return []
    resolved_model = normalize_retrieval_model(retrieval_model)

    topic_matches = retrieval_search(
        resolved_essay,
        top_n=top_n,
        retrieval_model=resolved_model,
        year_start=year_start,
        year_end=year_end,
    )
    if not topic_matches:
        log_runtime_event(
            "essay_search.no_topic_matches",
            retrieval_model=resolved_model,
        )
        return []

    if not resolved_thesis:
        log_runtime_event(
            "essay_search.return_topic_matches",
            retrieval_model=resolved_model,
            result_count=len(topic_matches),
        )
        return topic_matches

    log_runtime_event(
        "essay_search.rerank_start",
        retrieval_model=resolved_model,
        essay_chars=len(resolved_essay),
        thesis_chars=len(resolved_thesis),
        top_n=int(top_n),
        normalize_topic_scores=bool(normalize_topic_scores),
    )
    reranked = rerank_article_matches_by_statement(
        article_matches=topic_matches,
        statement=resolved_thesis,
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        recency_weight=recency_weight,
        top_n=top_n,
        normalize_topic_scores=normalize_topic_scores,
    )
    for match in reranked:
        match["selected_thesis_sentence"] = resolved_thesis
        match["selected_thesis_id"] = selected_thesis_id
        match["essay_query_text"] = resolved_essay
    return reranked
