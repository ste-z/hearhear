from backend.claim_store import get_claim_records
from backend.nli_processor import (
    normalize_stance_score,
    score_nli_pairs,
    stance_label_from_probs,
)
from backend.runtime_debug import log_runtime_event


DEFAULT_TOPIC_WEIGHT = 0.5
DEFAULT_STANCE_WEIGHT = 0.5
DEFAULT_RERANK_TOP_N = 20
MAX_RERANK_TOP_N = 100


def build_stance_statement(topic, opinion):
    topic_text = str(topic or "").strip()
    opinion_text = str(opinion or "").strip()
    if not topic_text and not opinion_text:
        return ""
    if not topic_text:
        return opinion_text
    if not opinion_text:
        return f"Regarding {topic_text}"
    return f"Regarding {topic_text}, I believe {opinion_text}"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resolve_weight_pair(topic_weight, stance_weight):
    resolved_topic = max(0.0, _safe_float(topic_weight, DEFAULT_TOPIC_WEIGHT))
    resolved_stance = max(0.0, _safe_float(stance_weight, DEFAULT_STANCE_WEIGHT))
    if resolved_topic == 0.0 and resolved_stance == 0.0:
        return DEFAULT_TOPIC_WEIGHT, DEFAULT_STANCE_WEIGHT
    return resolved_topic, resolved_stance


def _resolve_top_n(top_n):
    try:
        resolved = int(top_n)
    except (TypeError, ValueError):
        resolved = DEFAULT_RERANK_TOP_N
    return max(1, min(MAX_RERANK_TOP_N, resolved))


def _normalize_topic_scores(matches):
    max_score = max((_safe_float(match.get("score")) for match in matches), default=0.0)
    normalized = []
    for match in matches:
        topic_score = _safe_float(match.get("score"))
        normalized_score = topic_score / max_score if max_score > 0 else 0.0
        normalized.append((topic_score, normalized_score))
    return normalized


def _claim_payload(claim_record):
    if not claim_record:
        return {
            "central_claim_summary": None,
            "has_clear_central_thesis": None,
            "thesis_sentence_id": None,
            "thesis_sentence": None,
            "support_sentence_ids": [],
            "support_sentences": [],
            "secondary_claim_ids": [],
            "secondary_claim_sentences": [],
            "claim_source_path": None,
            "claim_available": False,
        }

    return {
        "central_claim_summary": claim_record.get("central_claim_summary"),
        "has_clear_central_thesis": claim_record.get("has_clear_central_thesis"),
        "thesis_sentence_id": claim_record.get("thesis_sentence_id"),
        "thesis_sentence": claim_record.get("thesis_sentence"),
        "support_sentence_ids": claim_record.get("support_sentence_ids") or [],
        "support_sentences": claim_record.get("support_sentences") or [],
        "secondary_claim_ids": claim_record.get("secondary_claim_ids") or [],
        "secondary_claim_sentences": claim_record.get("secondary_claim_sentences") or [],
        "claim_source_path": claim_record.get("_source_path"),
        "claim_available": True,
    }


def _combined_score(topic_score, stance_score, topic_weight, stance_weight):
    effective_topic_weight = float(topic_weight)
    effective_stance_weight = float(stance_weight if stance_score is not None else 0.0)
    if effective_topic_weight == 0.0 and effective_stance_weight == 0.0:
        return float(topic_score)
    total_weight = effective_topic_weight + effective_stance_weight
    return (
        (float(topic_score) * effective_topic_weight)
        + (float(stance_score or 0.0) * effective_stance_weight)
    ) / total_weight


def rerank_article_matches_by_statement(
    article_matches,
    statement,
    topic_weight=DEFAULT_TOPIC_WEIGHT,
    stance_weight=DEFAULT_STANCE_WEIGHT,
    top_n=DEFAULT_RERANK_TOP_N,
):
    resolved_top_n = _resolve_top_n(top_n)
    matches = [dict(match) for match in list(article_matches)[:resolved_top_n]]
    if not matches:
        return []

    log_runtime_event(
        "stance_rerank.start",
        match_count=len(matches),
        statement_chars=len(str(statement or "").strip()),
        top_n=resolved_top_n,
    )
    topic_weight, stance_weight = _resolve_weight_pair(topic_weight, stance_weight)
    query_statement = str(statement or "").strip()
    if not query_statement:
        log_runtime_event("stance_rerank.no_statement")
        return matches

    claim_records = get_claim_records(matches)

    indexed_claims = []
    premises = []
    for idx, match in enumerate(matches):
        claim_record = claim_records.get(str(match.get("id") or "").strip())
        matches[idx].update(_claim_payload(claim_record))
        claim_summary = matches[idx].get("central_claim_summary")
        if claim_summary:
            indexed_claims.append(idx)
            premises.append(claim_summary)
    log_runtime_event(
        "stance_rerank.claims_ready",
        claim_premise_count=len(premises),
        match_count=len(matches),
    )

    nli_rows = score_nli_pairs(premises, query_statement) if premises else []
    log_runtime_event("stance_rerank.nli_done", nli_row_count=len(nli_rows))
    nli_by_match_idx = dict(zip(indexed_claims, nli_rows))
    topic_scores = _normalize_topic_scores(matches)

    reranked = []
    for idx, match in enumerate(matches):
        topic_score, topic_score_normalized = topic_scores[idx]
        nli_row = nli_by_match_idx.get(idx)
        if nli_row is None:
            stance_score = None
            stance_score_normalized = None
            entailment_prob = None
            neutral_prob = None
            contradiction_prob = None
            stance_label = None
        else:
            entailment_prob = nli_row["entailment_prob"]
            neutral_prob = nli_row["neutral_prob"]
            contradiction_prob = nli_row["contradiction_prob"]
            stance_score = nli_row["stance_score"]
            stance_score_normalized = normalize_stance_score(stance_score)
            stance_label = stance_label_from_probs(
                entailment_prob=entailment_prob,
                neutral_prob=neutral_prob,
                contradiction_prob=contradiction_prob,
            )

        match["query_statement"] = query_statement
        match["topic_statement"] = query_statement
        match["topic_score"] = topic_score
        match["topic_score_normalized"] = topic_score_normalized
        match["stance_entailment_prob"] = entailment_prob
        match["stance_neutral_prob"] = neutral_prob
        match["stance_contradiction_prob"] = contradiction_prob
        match["stance_score"] = stance_score
        match["stance_score_normalized"] = stance_score_normalized
        match["stance_label"] = stance_label
        match["combined_score"] = _combined_score(
            topic_score=topic_score_normalized,
            stance_score=stance_score_normalized,
            topic_weight=topic_weight,
            stance_weight=stance_weight,
        )
        match["topic_weight"] = topic_weight
        match["stance_weight"] = stance_weight
        reranked.append(match)

    reranked.sort(
        key=lambda match: (
            _safe_float(match.get("combined_score")),
            _safe_float(match.get("stance_score_normalized"), -1.0),
            _safe_float(match.get("topic_score_normalized")),
        ),
        reverse=True,
    )

    for rank_idx, match in enumerate(reranked, start=1):
        match["rerank_position"] = rank_idx

    log_runtime_event("stance_rerank.done", reranked_count=len(reranked))
    return reranked


def rerank_article_matches(
    article_matches,
    topic,
    opinion,
    topic_weight=DEFAULT_TOPIC_WEIGHT,
    stance_weight=DEFAULT_STANCE_WEIGHT,
    top_n=DEFAULT_RERANK_TOP_N,
):
    return rerank_article_matches_by_statement(
        article_matches=article_matches,
        statement=build_stance_statement(topic, opinion),
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        top_n=top_n,
    )
