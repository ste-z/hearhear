from backend.runtime.runtime_debug import log_runtime_event
from backend.services.retrieval_service import (
    DEFAULT_CHUNK_ARTICLE_TOP_K,
    DEFAULT_CHUNK_CANDIDATE_TOP_K,
    DEFAULT_RETRIEVAL_MODEL,
    normalize_retrieval_model,
    DEFAULT_RERANK_SELECTION_MODE,
    normalize_rerank_selection_mode,
    select_rerank_candidates,
)
from backend.stance_processing.nli_processor import score_claim_sentences
from backend.stance_processing.stance_rerank import (
    normalize_chunking_mode,
    normalize_stance_method,
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
    chunk_candidate_top_k=DEFAULT_CHUNK_CANDIDATE_TOP_K,
    chunk_article_top_k=DEFAULT_CHUNK_ARTICLE_TOP_K,
):
    resolved_essay = str(essay_text or "").strip()
    resolved_thesis = str(selected_thesis_sentence or "").strip()
    if len(resolved_essay) < 3:
        return {
            "results": [],
            "empty_results_message": None,
        }
    resolved_model = normalize_retrieval_model(retrieval_model)
    if use_chunking and str(chunking_mode or "none") != "none" and resolved_model == "tfidf":
        resolved_model = "svd"
    resolved_selection_mode = normalize_rerank_selection_mode(rerank_selection_mode)

    candidate_payload = select_rerank_candidates(
        query=resolved_essay,
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

    resolved_chunking_mode = normalize_chunking_mode(chunking_mode)
    if use_chunking and resolved_chunking_mode == "none":
        resolved_chunking_mode = "paragraph"
    resolved_use_chunking = resolved_chunking_mode != "none"
    resolved_stance_method = (
        "llm" if resolved_use_chunking else normalize_stance_method(stance_method)
    )
    agreement_statement = (
        resolved_essay if resolved_stance_method == "llm" else resolved_thesis
    )
    agreement_statement_source = (
        "essay" if resolved_stance_method == "llm" else "selected_thesis_sentence"
    )

    if not agreement_statement:
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
        agreement_statement_chars=len(agreement_statement),
        agreement_statement_source=agreement_statement_source,
        top_n=len(topic_matches),
        normalize_topic_scores=bool(normalize_topic_scores),
        rerank_selection_mode=resolved_selection_mode,
        rerank_threshold=candidate_payload.get("rerank_threshold"),
        stance_method=resolved_stance_method,
        use_chunking=bool(resolved_use_chunking),
        chunking_mode=resolved_chunking_mode,
    )
    reranked = rerank_article_matches_by_statement(
        article_matches=topic_matches,
        statement=agreement_statement,
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        recency_weight=recency_weight,
        top_n=len(topic_matches),
        normalize_topic_scores=normalize_topic_scores,
        stance_method=resolved_stance_method,
        use_chunking=resolved_use_chunking,
        chunking_mode=resolved_chunking_mode,
    )
    for match in reranked:
        match["selected_thesis_sentence"] = resolved_thesis
        match["selected_thesis_id"] = selected_thesis_id
        match["essay_query_text"] = resolved_essay
    return {
        "results": reranked,
        "empty_results_message": candidate_payload.get("empty_results_message"),
    }
