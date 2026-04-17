from datetime import datetime, timezone

from backend.claims.claim_store import get_claim_records
from backend.runtime.runtime_debug import log_runtime_event
from backend.stance_processing.nli_processor import (
    normalize_stance_score,
    score_nli_pairs,
    stance_label_from_probs,
)


DEFAULT_TOPIC_WEIGHT = 0.4
DEFAULT_STANCE_WEIGHT = 0.4
DEFAULT_RECENCY_WEIGHT = 0.2
DEFAULT_RERANK_TOP_N = 20
MAX_RERANK_TOP_N = 100
RECENCY_HALF_LIFE_DAYS = 365.0 * 3.0
DEFAULT_NORMALIZE_TOPIC_SCORES = False
DEFAULT_STANCE_METHOD = "nli"
SUPPORTED_STANCE_METHODS = ("nli", "llm")


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


def _resolve_weight_triplet(topic_weight, stance_weight, recency_weight):
    resolved_topic = max(0.0, _safe_float(topic_weight, DEFAULT_TOPIC_WEIGHT))
    resolved_stance = max(0.0, _safe_float(stance_weight, DEFAULT_STANCE_WEIGHT))
    resolved_recency = max(0.0, _safe_float(recency_weight, DEFAULT_RECENCY_WEIGHT))
    if resolved_topic == 0.0 and resolved_stance == 0.0 and resolved_recency == 0.0:
        return DEFAULT_TOPIC_WEIGHT, DEFAULT_STANCE_WEIGHT, DEFAULT_RECENCY_WEIGHT
    return resolved_topic, resolved_stance, resolved_recency


def _resolve_top_n(top_n):
    try:
        resolved = int(top_n)
    except (TypeError, ValueError):
        resolved = DEFAULT_RERANK_TOP_N
    return max(1, min(MAX_RERANK_TOP_N, resolved))


def normalize_stance_method(value, default=DEFAULT_STANCE_METHOD):
    normalized = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"nli", "deberta", "deberta_nli"}:
        return "nli"
    if normalized in {"llm", "spark", "spark_llm", "rag", "llm_rag"}:
        return "llm"

    supported = ", ".join(SUPPORTED_STANCE_METHODS)
    raise ValueError(
        f"Unsupported stance_method {value!r}. Supported methods: {supported}."
    )


def _normalize_topic_scores(matches):
    max_score = max((_safe_float(match.get("score")) for match in matches), default=0.0)
    normalized = []
    for match in matches:
        topic_score = _safe_float(match.get("score"))
        normalized_score = topic_score / max_score if max_score > 0 else 0.0
        normalized.append((topic_score, normalized_score))
    return normalized


def _coerce_datetime(value):
    if isinstance(value, datetime):
        resolved = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            resolved = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


def _recency_score(date_value, reference_time=None):
    published_at = _coerce_datetime(date_value)
    if published_at is None:
        return None

    reference = reference_time or datetime.now(timezone.utc)
    age_seconds = max(0.0, (reference - published_at).total_seconds())
    age_days = age_seconds / 86400.0
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


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


def _combined_score(
    topic_score,
    stance_score,
    recency_score,
    topic_weight,
    stance_weight,
    recency_weight,
):
    effective_topic_weight = float(topic_weight)
    effective_stance_weight = float(stance_weight if stance_score is not None else 0.0)
    effective_recency_weight = float(recency_weight if recency_score is not None else 0.0)
    if (
        effective_topic_weight == 0.0
        and effective_stance_weight == 0.0
        and effective_recency_weight == 0.0
    ):
        return float(topic_score)
    total_weight = (
        effective_topic_weight
        + effective_stance_weight
        + effective_recency_weight
    )
    return (
        (float(topic_score) * effective_topic_weight)
        + (float(stance_score or 0.0) * effective_stance_weight)
        + (float(recency_score or 0.0) * effective_recency_weight)
    ) / total_weight


def rerank_article_matches_by_statement(
    article_matches,
    statement,
    topic_weight=DEFAULT_TOPIC_WEIGHT,
    stance_weight=DEFAULT_STANCE_WEIGHT,
    recency_weight=DEFAULT_RECENCY_WEIGHT,
    top_n=DEFAULT_RERANK_TOP_N,
    normalize_topic_scores=DEFAULT_NORMALIZE_TOPIC_SCORES,
    stance_method=DEFAULT_STANCE_METHOD,
):
    resolved_top_n = _resolve_top_n(top_n)
    resolved_stance_method = normalize_stance_method(stance_method)
    matches = [dict(match) for match in list(article_matches)[:resolved_top_n]]
    if not matches:
        return []

    log_runtime_event(
        "stance_rerank.start",
        match_count=len(matches),
        statement_chars=len(str(statement or "").strip()),
        top_n=resolved_top_n,
        normalize_topic_scores=bool(normalize_topic_scores),
        stance_method=resolved_stance_method,
    )
    topic_weight, stance_weight, recency_weight = _resolve_weight_triplet(
        topic_weight,
        stance_weight,
        recency_weight,
    )
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

    if resolved_stance_method == "llm":
        from backend.stance_processing.llm_processor import score_llm_article_agreement

        stance_rows = score_llm_article_agreement(matches, query_statement)
        log_runtime_event(
            "stance_rerank.llm_done",
            llm_row_count=len(stance_rows),
        )
        stance_by_match_idx = dict(enumerate(stance_rows))
    else:
        nli_rows = score_nli_pairs(premises, query_statement) if premises else []
        log_runtime_event("stance_rerank.nli_done", nli_row_count=len(nli_rows))
        stance_by_match_idx = dict(zip(indexed_claims, nli_rows))

    topic_scores = _normalize_topic_scores(matches)
    reference_time = datetime.now(timezone.utc)

    reranked = []
    for idx, match in enumerate(matches):
        topic_score, topic_score_normalized = topic_scores[idx]
        topic_score_display = (
            topic_score_normalized if normalize_topic_scores else topic_score
        )
        recency_score_normalized = _recency_score(
            match.get("date"),
            reference_time=reference_time,
        )
        stance_row = stance_by_match_idx.get(idx)
        if stance_row is None:
            stance_score = None
            stance_score_normalized = None
            entailment_prob = None
            neutral_prob = None
            contradiction_prob = None
            stance_label = None
        else:
            if resolved_stance_method == "llm":
                entailment_prob = None
                neutral_prob = None
                contradiction_prob = None
                stance_score = stance_row["stance_score"]
                stance_score_normalized = stance_row["agreement_score"]
                stance_label = None
            else:
                entailment_prob = stance_row["entailment_prob"]
                neutral_prob = stance_row["neutral_prob"]
                contradiction_prob = stance_row["contradiction_prob"]
                stance_score = stance_row["stance_score"]
                stance_score_normalized = normalize_stance_score(stance_score)
                stance_label = stance_label_from_probs(
                    entailment_prob=entailment_prob,
                    neutral_prob=neutral_prob,
                    contradiction_prob=contradiction_prob,
                )

        match["query_statement"] = query_statement
        match["topic_statement"] = query_statement
        match["stance_method"] = resolved_stance_method
        match["topic_score"] = topic_score
        match["topic_score_normalized"] = topic_score_normalized
        match["topic_score_display"] = topic_score_display
        match["topic_score_is_normalized"] = bool(normalize_topic_scores)
        match["recency_score_normalized"] = recency_score_normalized
        match["stance_entailment_prob"] = entailment_prob
        match["stance_neutral_prob"] = neutral_prob
        match["stance_contradiction_prob"] = contradiction_prob
        match["stance_score"] = stance_score
        match["stance_score_normalized"] = stance_score_normalized
        match["stance_label"] = stance_label
        match["llm_agreement_score"] = (
            stance_score_normalized if resolved_stance_method == "llm" else None
        )
        match["combined_score"] = _combined_score(
            topic_score=topic_score_display,
            stance_score=stance_score_normalized,
            recency_score=recency_score_normalized,
            topic_weight=topic_weight,
            stance_weight=stance_weight,
            recency_weight=recency_weight,
        )
        match["topic_weight"] = topic_weight
        match["stance_weight"] = stance_weight
        match["recency_weight"] = recency_weight
        reranked.append(match)

    reranked.sort(
        key=lambda match: (
            _safe_float(match.get("combined_score")),
            _safe_float(match.get("stance_score_normalized"), -1.0),
            _safe_float(match.get("topic_score_display")),
            _safe_float(match.get("recency_score_normalized"), -1.0),
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
    recency_weight=DEFAULT_RECENCY_WEIGHT,
    top_n=DEFAULT_RERANK_TOP_N,
    normalize_topic_scores=DEFAULT_NORMALIZE_TOPIC_SCORES,
    stance_method=DEFAULT_STANCE_METHOD,
):
    return rerank_article_matches_by_statement(
        article_matches=article_matches,
        statement=build_stance_statement(topic, opinion),
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        recency_weight=recency_weight,
        top_n=top_n,
        normalize_topic_scores=normalize_topic_scores,
        stance_method=stance_method,
    )
