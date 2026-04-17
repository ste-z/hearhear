from backend.runtime.runtime_debug import log_runtime_event
from backend.services.retrieval_service import (
    DEFAULT_RETRIEVAL_MODEL,
    normalize_retrieval_model,
    DEFAULT_RERANK_SELECTION_MODE,
    normalize_rerank_selection_mode,
    select_rerank_candidates,
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
    rerank_selection_mode=DEFAULT_RERANK_SELECTION_MODE,
    rerank_threshold=None,
    stance_method="nli",
    use_chunking=False,
):
    resolved_essay = str(essay_text or "").strip()
    resolved_thesis = str(selected_thesis_sentence or "").strip()
    if len(resolved_essay) < 3:
        return {
            "results": [],
            "empty_results_message": None,
        }
    resolved_model = normalize_retrieval_model(retrieval_model)
    resolved_selection_mode = normalize_rerank_selection_mode(rerank_selection_mode)

    candidate_payload = select_rerank_candidates(
        query=resolved_essay,
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
            "essay_search.no_topic_matches",
            retrieval_model=resolved_model,
            rerank_selection_mode=resolved_selection_mode,
            rerank_threshold=candidate_payload.get("rerank_threshold"),
            empty_results_message=candidate_payload.get("empty_results_message"),
        )
        return {
            "results": [],
            "empty_results_message": candidate_payload.get("empty_results_message"),
        }

    if not resolved_thesis:
        log_runtime_event(
            "essay_search.return_topic_matches",
            retrieval_model=resolved_model,
            result_count=len(topic_matches),
            rerank_selection_mode=resolved_selection_mode,
            rerank_threshold=candidate_payload.get("rerank_threshold"),
        )
        return {
            "results": topic_matches,
            "empty_results_message": candidate_payload.get("empty_results_message"),
        }

    log_runtime_event(
        "essay_search.rerank_start",
        retrieval_model=resolved_model,
        essay_chars=len(resolved_essay),
        thesis_chars=len(resolved_thesis),
        top_n=len(topic_matches),
        normalize_topic_scores=bool(normalize_topic_scores),
        rerank_selection_mode=resolved_selection_mode,
        rerank_threshold=candidate_payload.get("rerank_threshold"),
        stance_method=stance_method,
        use_chunking=bool(use_chunking),
    )
    reranked = rerank_article_matches_by_statement(
        article_matches=topic_matches,
        statement=resolved_thesis,
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        recency_weight=recency_weight,
        top_n=len(topic_matches),
        normalize_topic_scores=normalize_topic_scores,
        stance_method=stance_method,
        use_chunking=use_chunking,
    )
    for match in reranked:
        match["selected_thesis_sentence"] = resolved_thesis
        match["selected_thesis_id"] = selected_thesis_id
        match["essay_query_text"] = resolved_essay
    return {
        "results": reranked,
        "empty_results_message": candidate_payload.get("empty_results_message"),
    }
