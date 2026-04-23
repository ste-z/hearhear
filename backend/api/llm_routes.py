"""
LLM routes — only loaded when USE_LLM = True in routes.py.
Adds POST /api/chat for LLM-driven Guardian article RAG and
POST /api/llm/agreement-scores for direct agreement scoring.

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import re
import logging
from flask import request, jsonify, Response, stream_with_context

from backend.stance_processing.llm_processor import (
    _article_body_text_lookup as article_body_text_lookup,
    create_spark_client,
    score_llm_article_agreement,
    score_llm_article_agreement_by_paragraphs,
)
from backend.stance_processing.stance_rerank import (
    DEFAULT_CHUNKING_MODE,
    normalize_chunking_mode,
)
from backend.text_processing.search_helpers import (
    DEFAULT_RETRIEVAL_MODEL,
    normalize_retrieval_model,
)
from backend.text_processing.svd_dimension_labels import cached_svd_dimension_labels

logger = logging.getLogger(__name__)
QUERY_HELP_MAX_FIELD_CHARS = 220


def _coerce_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _clean_svd_text(value, max_chars=160):
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars - 3].rstrip()}..."


def _clean_svd_dimension_label(value):
    label = _clean_svd_text(value, 120)
    label = re.sub(r"^(concept|dimension)\s+\d+\s*[:\-]\s*", "", label, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"(concept|dimension)?\s*\d+", label, flags=re.IGNORECASE):
        return ""
    return label


def _normalize_svd_dimension_label_map(raw_labels):
    labels = {}

    if isinstance(raw_labels, dict):
        iterable = raw_labels.items()
    elif isinstance(raw_labels, list):
        iterable = (
            (item.get("dimension_index"), item.get("label"))
            for item in raw_labels
            if isinstance(item, dict)
        )
    else:
        return labels

    for raw_index, raw_label in iterable:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        label = _clean_svd_dimension_label(raw_label)
        if label:
            labels[index] = label

    return labels


def _coerce_svd_dimension_index(dimension):
    try:
        return int(dimension.get("dimension_index", -1))
    except (AttributeError, TypeError, ValueError):
        return -1


def _svd_dimension_terms(dimension):
    if not isinstance(dimension, dict):
        return []
    return [
        str(term).strip()
        for term in dimension.get("label_terms") or []
        if str(term).strip()
    ]


def _svd_dimension_display_name(dimension, dimension_labels=None):
    if not isinstance(dimension, dict):
        return "Unnamed latent concept"

    index = _coerce_svd_dimension_index(dimension)
    if isinstance(dimension_labels, dict) and index in dimension_labels:
        label = _clean_svd_dimension_label(dimension_labels.get(index))
        if label:
            return label

    for key in ("display_label", "dimension_name", "name", "label"):
        label = _clean_svd_dimension_label(dimension.get(key))
        if label:
            return label

    label_text = _clean_svd_dimension_label(dimension.get("label_text"))
    if label_text:
        return label_text

    label_terms = _svd_dimension_terms(dimension)
    if label_terms:
        return ", ".join(label_terms[:3])

    return "Unnamed latent concept"


def _format_svd_dimensions(dimensions, dimension_labels=None):
    if not isinstance(dimensions, list) or not dimensions:
        return "None"

    lines = []
    for dimension in dimensions:
        name = _svd_dimension_display_name(dimension, dimension_labels)
        value_raw = dimension.get("value")
        try:
            value = float(value_raw)
            value_text = f"{value:.3f}"
        except (TypeError, ValueError):
            value_text = str(value_raw)

        magnitude_raw = dimension.get("magnitude")
        magnitude_text = None
        try:
            magnitude = float(magnitude_raw)
            magnitude_text = f"{magnitude:.3f}"
        except (TypeError, ValueError):
            magnitude_text = str(magnitude_raw) if magnitude_raw is not None else None

        pole = dimension.get("pole")
        label_terms = _svd_dimension_terms(dimension)
        term_text = f"terms: {', '.join(label_terms)}" if label_terms else None

        parts = [f"Radar concept: {name}", f"value: {value_text}"]
        if magnitude_text:
            parts.append(f"magnitude: {magnitude_text}")
        if pole:
            parts.append(f"pole: {pole}")
        if term_text:
            parts.append(term_text)

        lines.append("; ".join(parts))

    return "\n".join(lines) if lines else "None"


def _clean_overview_text(value, max_chars=500):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars - 3].rstrip()}..."


def _clean_query_help_text(value, max_chars=QUERY_HELP_MAX_FIELD_CHARS):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.strip(" \t\r\n\"'")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _format_stance_query(topic, opinion):
    return f"Regarding {topic}, I believe {opinion}"


def _parse_stance_query(value):
    text = _clean_query_help_text(value, max_chars=600)
    match = re.match(
        r"^regarding\s+(.+?)\s*,?\s+i\s+believe\s+(.+?)\.?\s*$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None
    return (
        _clean_query_help_text(match.group(1)),
        _clean_query_help_text(match.group(2)),
    )


def _llm_json_object(raw_content, response_label):
    text = str(raw_content or "").strip()
    if not text:
        raise ValueError(f"Empty {response_label} response")

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text
    parsed = json.loads(candidate)
    if isinstance(parsed, list):
        return {"alternatives": parsed, "suggestions": parsed}
    if not isinstance(parsed, dict):
        raise ValueError(f"{response_label} response must be a JSON object")
    return parsed


def _query_help_key(value):
    return re.sub(r"[\W_]+", " ", str(value or "").lower()).strip()


def _normalize_query_rewrite_alternatives(raw_alternatives, fallback_topic, fallback_opinion):
    if isinstance(raw_alternatives, dict):
        raw_alternatives = list(raw_alternatives.values())
    if not isinstance(raw_alternatives, list):
        return []

    alternatives = []
    seen = set()
    for item in raw_alternatives:
        topic = ""
        opinion = ""
        rationale = ""

        if isinstance(item, dict):
            topic = _clean_query_help_text(
                item.get("topic")
                or item.get("regarding")
                or item.get("regarding_clause")
                or item.get("subject")
            )
            opinion = _clean_query_help_text(
                item.get("opinion")
                or item.get("belief")
                or item.get("stance")
                or item.get("i_believe")
                or item.get("i_believe_clause")
                or item.get("believe")
            )
            rationale = _clean_overview_text(
                item.get("rationale")
                or item.get("why")
                or item.get("reason")
                or item.get("explanation"),
                240,
            )
            if not topic or not opinion:
                parsed_topic, parsed_opinion = _parse_stance_query(
                    item.get("query") or item.get("formatted_query") or item.get("full_query")
                )
                topic = topic or parsed_topic or ""
                opinion = opinion or parsed_opinion or ""
        elif isinstance(item, str):
            topic, opinion = _parse_stance_query(item)

        if not topic and opinion:
            topic = fallback_topic
        if not opinion and topic:
            opinion = fallback_opinion
        topic = _clean_query_help_text(topic)
        opinion = _clean_query_help_text(opinion)

        if not topic or not opinion:
            continue

        if (
            _query_help_key(topic) == _query_help_key(fallback_topic)
            and _query_help_key(opinion) == _query_help_key(fallback_opinion)
        ):
            continue

        key = (topic.lower(), opinion.lower())
        if key in seen:
            continue

        alternatives.append({
            "topic": topic,
            "opinion": opinion,
            "query": _format_stance_query(topic, opinion),
            "rationale": rationale,
        })
        seen.add(key)

        if len(alternatives) >= 3:
            break

    return alternatives


def _normalize_query_improvement_suggestions(raw_suggestions):
    if not isinstance(raw_suggestions, list):
        return []

    suggestions = []
    seen = set()
    for item in raw_suggestions:
        if isinstance(item, dict):
            suggestion = _clean_overview_text(
                item.get("suggestion") or item.get("tip") or item.get("text"),
                360,
            )
        else:
            suggestion = _clean_overview_text(item, 360)
        key = suggestion.lower()
        if not suggestion or key in seen:
            continue
        suggestions.append(suggestion)
        seen.add(key)
        if len(suggestions) >= 6:
            break

    return suggestions


def _query_help_method_guidance(retrieval_model):
    if retrieval_model == "tfidf":
        return (
            "The first stage uses lexical TF-IDF cosine similarity over article text. "
            "The Regarding field should contain concrete words and phrases likely to appear in relevant Guardian opinion articles: "
            "policy nouns, actors, institutions, events, and high-signal synonyms. Avoid vague placeholders, pronouns, and overly broad one-word topics."
        )
    if retrieval_model == "minilm":
        return (
            "The first stage uses a dense MiniLM embedding model over semantic chunks or pooled article embeddings. "
            "The Regarding field should name the issue clearly and include a few concrete concepts, actors, or policy terms so the query lands near the intended semantic neighborhood. "
            "You do not need keyword stuffing, but you should avoid vague one-word topics."
        )

    return (
        "The first stage uses cosine similarity after projecting TF-IDF vectors into truncated-SVD latent semantic dimensions. "
        "The Regarding field should name the broad issue plus a few semantically related concepts so it lands near the intended latent theme. "
        "Avoid keyword stuffing, but include enough context to distinguish the issue from neighboring themes."
    )


def _result_index_for_article(article, fallback_index):
    fallback = int(fallback_index) + 1
    if not isinstance(article, dict):
        return fallback

    for key in ("result_index", "resultIndex", "source_index", "sourceIndex"):
        try:
            result_index = int(article.get(key))
        except (TypeError, ValueError):
            continue
        if result_index >= 1:
            return result_index
    return fallback


def _max_source_index(sources, fallback_count):
    indices = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        try:
            result_index = int(source.get("result_index"))
        except (TypeError, ValueError):
            continue
        if result_index >= 1:
            indices.append(result_index)
    return max(indices) if indices else int(fallback_count)


def _remap_result_indices(source_indices, articles):
    ordinal_to_result_index = {
        index + 1: _result_index_for_article(article, index)
        for index, article in enumerate(articles or [])
    }
    allowed_result_indices = set(ordinal_to_result_index.values())
    remapped = []
    seen = set()
    for source_index in source_indices or []:
        result_index = ordinal_to_result_index.get(source_index, source_index)
        if result_index not in allowed_result_indices:
            continue
        if result_index in seen:
            continue
        remapped.append(result_index)
        seen.add(result_index)
    return remapped


def _format_article_for_results_overview(article, index):
    result_index = _result_index_for_article(article, index)
    title = _clean_overview_text(article.get("title"), 240)
    summary = _clean_overview_text(article.get("summary"), 700)
    claim = _clean_overview_text(article.get("central_claim_summary"), 500)
    stance = _clean_overview_text(article.get("stance_label"), 80)
    agreement = article.get("llm_agreement_score")
    score = (
        article.get("combined_score")
        or article.get("topic_score_display")
        or article.get("topic_score")
        or article.get("score")
    )

    lines = [f"Result {result_index}: {title or 'Untitled'}"]
    if summary:
        lines.append(f"Summary: {summary}")
    if claim:
        lines.append(f"Central claim: {claim}")
    if stance:
        lines.append(f"Agreement label: {stance}")
    if agreement is not None:
        lines.append(f"LLM agreement score: {agreement}")
    if score is not None:
        lines.append(f"Ranking score: {score}")
    return "\n".join(lines)


def _clean_results_chat_list(value, max_items=6, max_chars=320):
    if not isinstance(value, list):
        return []

    items = []
    for raw_item in value[:max_items]:
        if isinstance(raw_item, dict):
            raw_text = raw_item.get("text") or raw_item.get("sentence") or raw_item.get("evidence")
        else:
            raw_text = raw_item
        text = _clean_overview_text(raw_text, max_chars)
        if text:
            items.append(text)
    return items


def _append_results_chat_list(lines, label, items):
    if not items:
        return
    lines.append(f"{label}:")
    lines.extend(f"- {item}" for item in items)


def _format_svd_context_group(label, dimensions, limit=5):
    if not isinstance(dimensions, list) or not dimensions:
        return ""
    formatted = _format_svd_dimensions(dimensions[:limit])
    if not formatted or formatted == "None":
        return ""
    return f"{label}:\n{formatted}"


def _ranking_value_text(value, decimals=3, max_chars=80):
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        text = _clean_overview_text(value, max_chars)
        return text or None


def _append_ranking_value(lines, label, value, decimals=3):
    text = _ranking_value_text(value, decimals=decimals)
    if text is not None:
        lines.append(f"{label}: {text}")


def _format_topic_chunks_for_ranking(article, limit=3, max_chars=360):
    chunks = article.get("topic_relevant_chunks")
    if not isinstance(chunks, list) or not chunks:
        return ""

    lines = ["Top retrieved topic chunks:"]
    count = 0
    for chunk in chunks[:limit]:
        if not isinstance(chunk, dict):
            continue
        text = _clean_overview_text(chunk.get("text"), max_chars)
        if not text:
            continue

        metadata = []
        try:
            chunk_index = int(chunk.get("chunk_index"))
            metadata.append(f"chunk {chunk_index + 1}")
        except (TypeError, ValueError):
            pass

        score_text = _ranking_value_text(chunk.get("topic_score"))
        if score_text is not None:
            metadata.append(f"topic_score={score_text}")

        source = _clean_overview_text(chunk.get("source"), 60)
        if source:
            metadata.append(f"source={source}")

        prefix = f"{'; '.join(metadata)}: " if metadata else ""
        lines.append(f"- {prefix}{text}")
        count += 1

    return "\n".join(lines) if count > 0 else ""


def _format_agreement_excerpts_for_ranking(article, limit=3, max_chars=360):
    excerpts = article.get("llm_relevant_paragraphs")
    if not isinstance(excerpts, list) or not excerpts:
        return ""

    lines = ["Agreement evidence excerpts:"]
    count = 0
    for excerpt in excerpts[:limit]:
        if not isinstance(excerpt, dict):
            continue
        text = _clean_overview_text(excerpt.get("text"), max_chars)
        if not text:
            continue

        metadata = []
        try:
            paragraph_index = int(excerpt.get("paragraph_index"))
            metadata.append(f"excerpt {paragraph_index + 1}")
        except (TypeError, ValueError):
            pass

        score_text = _ranking_value_text(excerpt.get("agreement_score"))
        if score_text is not None:
            metadata.append(f"agreement_score={score_text}")

        prefix = f"{'; '.join(metadata)}: " if metadata else ""
        lines.append(f"- {prefix}{text}")
        count += 1

    return "\n".join(lines) if count > 0 else ""


def _ranking_mode_system_prompt(retrieval_model):
    if retrieval_model == "tfidf":
        return (
            "You are an expert analyst explaining why a result ranked highly in a lexical "
            "retrieval-and-reranking system. Use plain language and focus on visible word or "
            "phrase overlap."
        )
    if retrieval_model == "minilm":
        return (
            "You are an expert analyst explaining why a result ranked highly in a dense semantic "
            "retrieval-and-reranking system. Use plain language and focus on thematic overlap, "
            "issue framing, actors, values, and arguments rather than model jargon."
        )
    return (
        "You are an expert analyst explaining why a result ranked highly in an SVD-based semantic "
        "retrieval-and-reranking system. Use plain language while accurately grounding the "
        "explanation in the provided latent concept labels."
    )


def _ranking_mode_label(retrieval_model):
    if retrieval_model == "tfidf":
        return "Lexical TF-IDF"
    if retrieval_model == "minilm":
        return "Enhanced Semantic (MiniLM)"
    return "Semantic SVD"


def _build_ranking_explanation_prompt(
    *,
    query,
    position_label,
    article,
    retrieval_model,
    chunking_mode,
    query_svd_dimensions=None,
    dimension_labels=None,
):
    resolved_chunking_mode = normalize_chunking_mode(chunking_mode, DEFAULT_CHUNKING_MODE)
    stage_one_granularity = (
        "semantic chunks"
        if resolved_chunking_mode != "none" or article.get("chunk_retrieval_enabled")
        else "whole article"
    )

    article_title = _clean_overview_text(article.get("title"), 240)
    article_summary = _clean_overview_text(article.get("summary"), 700)
    article_claim = _clean_overview_text(article.get("central_claim_summary"), 500)
    article_stance = _clean_overview_text(article.get("stance_label"), 80)

    score_lines = [
        f"Stage 1 retrieval mode: {_ranking_mode_label(retrieval_model)}",
        f"Stage 1 search granularity: {stage_one_granularity}",
    ]
    _append_ranking_value(score_lines, "Combined ranking score", article.get("combined_score"))
    _append_ranking_value(
        score_lines,
        "Topic score",
        article.get("topic_score_display")
        or article.get("topic_score")
        or article.get("score"),
    )
    _append_ranking_value(score_lines, "Agreement score", article.get("stance_score_normalized"))
    _append_ranking_value(score_lines, "Recency score", article.get("recency_score_normalized"))
    _append_ranking_value(score_lines, "Matched chunk count", article.get("chunk_retrieval_matched_chunk_count"), decimals=0)
    _append_ranking_value(score_lines, "Chunk topic score max", article.get("chunk_topic_score_max"))
    _append_ranking_value(score_lines, "Chunk topic score top-k mean", article.get("chunk_topic_score_top_k_mean"))
    _append_ranking_value(score_lines, "Chunk topic score coverage", article.get("chunk_topic_score_coverage"))
    _append_ranking_value(score_lines, "Agreement excerpt count", article.get("llm_related_chunk_count"), decimals=0)
    if article_stance:
        score_lines.append(f"Agreement label: {article_stance}")

    evidence_sections = [
        _format_topic_chunks_for_ranking(article),
        _format_agreement_excerpts_for_ranking(article),
    ]
    evidence_text = "\n\n".join(section for section in evidence_sections if section)
    if not evidence_text:
        evidence_text = "No chunk or excerpt evidence was provided."

    common_intro = (
        "Explain in plain language why this article ranked at the reported position. "
        "Keep the answer to at most 3 sentences. "
        "Use only the provided query, article metadata, score breakdown, and evidence snippets. "
        "If agreement or recency likely affected the final rank, mention that briefly. "
        "Do not invent article content beyond the provided metadata and evidence."
    )

    if retrieval_model == "tfidf":
        mode_instructions = (
            "The first stage used lexical TF-IDF matching. Focus on concrete overlap in words, "
            "phrases, named actors, policies, or issue terms visible in the query and the article "
            "metadata or evidence snippets. Do not mention latent concepts or embeddings."
        )
        extra_context = ""
    elif retrieval_model == "minilm":
        mode_instructions = (
            "The first stage used dense semantic matching. Focus on shared themes, issue framing, "
            "policy ideas, values, actors, or arguments that make the article semantically close to "
            "the query. If chunk evidence is present, mention the strongest matching chunk themes. "
            "Do not mention latent concepts, dimension IDs, or internal embedding mechanics."
        )
        extra_context = ""
    else:
        mode_instructions = (
            "The first stage used truncated-SVD latent semantic similarity. Focus on shared themes "
            "or concepts that connect the query and article. When referring to SVD concepts, use the "
            "provided concept names exactly and do not mention numeric dimension identifiers."
        )
        query_svd_text = _format_svd_dimensions(query_svd_dimensions or [], dimension_labels or {})
        article_query_chart_text = _format_svd_dimensions(
            article.get("svd_query_chart_dimensions"),
            dimension_labels or {},
        )
        article_chart_text = _format_svd_dimensions(
            article.get("svd_chart_dimensions"),
            dimension_labels or {},
        )
        article_dimensions_text = _format_svd_dimensions(
            article.get("svd_dimensions"),
            dimension_labels or {},
        )
        extra_context = (
            "\n\nQuery SVD dimensions:\n"
            f"{query_svd_text}\n\n"
            "Article SVD query chart dimensions:\n"
            f"{article_query_chart_text}\n\n"
            "Article shared corpus SVD dimensions:\n"
            f"{article_chart_text}\n\n"
            "Article top SVD dimensions:\n"
            f"{article_dimensions_text}\n"
        )

    prompt = (
        f"{common_intro} {mode_instructions}\n\n"
        f"Target rank position: {position_label}\n\n"
        f"Query:\n{query}\n\n"
        "Article:\n"
        f"Title: {article_title or 'N/A'}\n"
        f"Summary: {article_summary or 'N/A'}\n"
    )
    if article_claim:
        prompt += f"Claim: {article_claim}\n"

    prompt += (
        "\nScore context:\n"
        f"{chr(10).join(score_lines)}\n\n"
        "Evidence:\n"
        f"{evidence_text}"
    )

    if extra_context:
        prompt += extra_context

    return prompt


def _format_article_for_results_chat(article, index, body_text=None, max_body_chars=5000):
    result_index = _result_index_for_article(article, index)
    title = _clean_overview_text(article.get("title"), 240)
    article_id = _clean_overview_text(article.get("id"), 180)
    author = _clean_overview_text(
        article.get("author_display") or article.get("author_raw"),
        180,
    )
    date = _clean_overview_text(article.get("date"), 80)
    url = _clean_overview_text(article.get("url"), 320)
    summary = _clean_overview_text(article.get("summary"), 900)
    claim = _clean_overview_text(article.get("central_claim_summary"), 700)
    thesis = _clean_overview_text(article.get("thesis_sentence"), 520)
    keywords = [
        _clean_overview_text(keyword, 80)
        for keyword in (article.get("keywords") or [])[:12]
        if _clean_overview_text(keyword, 80)
    ]

    lines = [f"Result {result_index}: {title or 'Untitled'}"]
    if article_id:
        lines.append(f"Article ID: {article_id}")
    if author:
        lines.append(f"Author: {author}")
    if date:
        lines.append(f"Date: {date}")
    if url:
        lines.append(f"URL: {url}")
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")
    if article.get("character_count") is not None:
        lines.append(f"Character count: {article.get('character_count')}")
    if article.get("word_count") is not None:
        lines.append(f"Word count: {article.get('word_count')}")
    if summary:
        lines.append(f"Summary: {summary}")
    if claim:
        lines.append(f"Central claim: {claim}")
    if thesis:
        lines.append(f"Thesis sentence: {thesis}")

    _append_results_chat_list(
        lines,
        "Support sentences",
        _clean_results_chat_list(article.get("support_sentences"), max_items=8, max_chars=360),
    )

    resolved_body_text = _clean_overview_text(
        body_text or article.get("body_text"),
        max_body_chars,
    )
    if resolved_body_text:
        lines.append(f"Body text excerpt: {resolved_body_text}")

    stance = _clean_overview_text(article.get("stance_label"), 80)
    if stance:
        lines.append(f"Agreement label: {stance}")
    if article.get("llm_agreement_score") is not None:
        lines.append(f"LLM agreement score: {article.get('llm_agreement_score')}")
    if article.get("combined_score") is not None:
        lines.append(f"Combined ranking score: {article.get('combined_score')}")
    if article.get("topic_score_display") is not None:
        lines.append(f"Topic score: {article.get('topic_score_display')}")
    if article.get("stance_score_normalized") is not None:
        lines.append(f"Agreement score: {article.get('stance_score_normalized')}")
    if article.get("recency_score_normalized") is not None:
        lines.append(f"Recency score: {article.get('recency_score_normalized')}")

    stance_probs = [
        ("support", article.get("stance_entailment_prob")),
        ("neutral", article.get("stance_neutral_prob")),
        ("contradict", article.get("stance_contradiction_prob")),
    ]
    if any(value is not None for _, value in stance_probs):
        lines.append(
            "Stance probabilities: "
            + ", ".join(f"{label}={value}" for label, value in stance_probs if value is not None)
        )

    relevant_paragraphs = article.get("llm_relevant_paragraphs")
    if isinstance(relevant_paragraphs, list) and relevant_paragraphs:
        lines.append("Relevant article excerpts:")
        for paragraph in relevant_paragraphs[:8]:
            if not isinstance(paragraph, dict):
                continue
            text = _clean_overview_text(paragraph.get("text"), 520)
            if not text:
                continue
            score = paragraph.get("agreement_score")
            score_text = f" agreement_score={score};" if score is not None else ""
            lines.append(f"-{score_text} {text}")

    sentiment = article.get("vader_sentiment")
    if isinstance(sentiment, dict):
        sentiment_parts = []
        for key in ("label", "tone_strength", "compound", "negative", "neutral", "positive"):
            value = sentiment.get(key)
            if value is not None:
                sentiment_parts.append(f"{key}={value}")
        if sentiment_parts:
            lines.append(f"Sentiment: {', '.join(sentiment_parts)}")
        snippets = sentiment.get("snippets")
        if isinstance(snippets, dict):
            for label in ("negative", "positive"):
                _append_results_chat_list(
                    lines,
                    f"Most {label} sentiment sentences",
                    _clean_results_chat_list(snippets.get(label), max_items=3, max_chars=260),
                )

    svd_groups = [
        _format_svd_context_group("Query-anchored latent concepts", article.get("svd_query_chart_dimensions")),
        _format_svd_context_group("Shared corpus latent concepts", article.get("svd_chart_dimensions")),
        _format_svd_context_group("Top article latent concepts", article.get("svd_dimensions")),
    ]
    lines.extend(group for group in svd_groups if group)
    return "\n".join(lines)


def _source_for_results_overview(article, index):
    return {
        "result_index": _result_index_for_article(article, index),
        "title": _clean_overview_text(article.get("title"), 240) or "Untitled",
        "url": article.get("url"),
        "article_id": article.get("id"),
    }


def _coerce_result_indices(value, max_index):
    if max_index <= 0:
        return []

    raw_values = value if isinstance(value, list) else [value]
    indices = []
    seen = set()
    for raw_value in raw_values:
        try:
            index = int(raw_value)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > max_index or index in seen:
            continue
        indices.append(index)
        seen.add(index)
    return indices


def _parse_results_overview_evidence_items(raw_items, max_index, max_items=5):
    evidence_items = []
    if not isinstance(raw_items, list):
        return evidence_items

    for item in raw_items[:max_items]:
        if isinstance(item, str):
            evidence = _clean_overview_text(item, 320)
            source_indices = []
        elif isinstance(item, dict):
            evidence = _clean_overview_text(
                item.get("evidence") or item.get("text") or item.get("claim"),
                320,
            )
            source_indices = _coerce_result_indices(
                item.get("source_indices") or item.get("sources") or item.get("result_indices"),
                max_index,
            )
        else:
            continue

        if evidence:
            evidence_items.append({
                "evidence": evidence,
                "source_indices": source_indices,
            })

    return evidence_items


def _parse_results_overview_arguments(raw_items, max_index, max_items=4):
    arguments = []
    if not isinstance(raw_items, list):
        return arguments

    for item in raw_items[:max_items]:
        if isinstance(item, str):
            argument = _clean_overview_text(item, 360)
            source_indices = []
            evidence = []
        elif isinstance(item, dict):
            argument = _clean_overview_text(
                item.get("argument") or item.get("claim") or item.get("point"),
                360,
            )
            source_indices = _coerce_result_indices(
                item.get("source_indices") or item.get("sources") or item.get("result_indices"),
                max_index,
            )
            evidence = _parse_results_overview_evidence_items(
                item.get("evidence") or item.get("key_evidence"),
                max_index,
                max_items=3,
            )
        else:
            continue

        if argument:
            arguments.append({
                "argument": argument,
                "source_indices": source_indices,
                "evidence": evidence,
            })

    return arguments


def _parse_results_chat_response(raw_content, max_source_index=10):
    text = str(raw_content or "").strip()
    if not text:
        raise ValueError("Empty results chat response")

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {
            "answer": _clean_overview_text(text, 1600),
            "source_indices": [],
            "follow_up_questions": [],
        }

    if not isinstance(parsed, dict):
        raise ValueError("Results chat response must be a JSON object")

    answer = _clean_overview_text(parsed.get("answer"), 1800)
    if not answer:
        raise ValueError("Results chat response is missing an answer")

    follow_ups = []
    raw_follow_ups = parsed.get("follow_up_questions")
    if isinstance(raw_follow_ups, list):
        for item in raw_follow_ups[:3]:
            question = _clean_overview_text(item, 180)
            if question:
                follow_ups.append(question)

    return {
        "answer": answer,
        "source_indices": _coerce_result_indices(
            parsed.get("source_indices") or parsed.get("sources") or parsed.get("result_indices"),
            max_source_index,
        ),
        "follow_up_questions": follow_ups,
    }


def _parse_results_overview_response(raw_content, max_source_index=10):
    text = str(raw_content or "").strip()
    if not text:
        raise ValueError("Empty results overview response")

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {
            "overview": text,
            "key_points": [],
            "supporting_arguments": [],
            "opposing_arguments": [],
            "key_evidence": [],
            "caveat": "",
        }

    if not isinstance(parsed, dict):
        raise ValueError("Results overview response must be a JSON object")

    overview = _clean_overview_text(parsed.get("overview"), 1200)
    key_points = []
    raw_key_points = parsed.get("key_points")
    if isinstance(raw_key_points, list):
        for item in raw_key_points[:4]:
            point = _clean_overview_text(item, 320)
            if point:
                key_points.append(point)

    caveat = _clean_overview_text(parsed.get("caveat"), 400)
    if not overview and key_points:
        overview = key_points[0]
        key_points = key_points[1:]
    if not overview:
        raise ValueError("Results overview response is missing an overview")

    return {
        "overview": overview,
        "key_points": key_points,
        "supporting_arguments": _parse_results_overview_arguments(
            parsed.get("supporting_arguments"),
            max_index=max_source_index,
        ),
        "opposing_arguments": _parse_results_overview_arguments(
            parsed.get("opposing_arguments"),
            max_index=max_source_index,
        ),
        "key_evidence": _parse_results_overview_evidence_items(
            parsed.get("key_evidence"),
            max_index=max_source_index,
        ),
        "caveat": caveat,
    }


def _format_dimension_for_labeling(dimension):
    try:
        index = int(dimension.get("dimension_index"))
    except (AttributeError, TypeError, ValueError):
        return None

    terms = [
        str(term).strip()
        for term in list(dimension.get("label_terms") or [])[:8]
        if str(term).strip()
    ]
    if not terms:
        label_text = _clean_overview_text(dimension.get("label_text"), 240)
        terms = [term.strip() for term in label_text.split(",") if term.strip()][:8]

    value = dimension.get("value")
    pole = _clean_overview_text(dimension.get("pole"), 40)
    return {
        "dimension_index": index,
        "terms": terms,
        "value": value,
        "pole": pole,
    }


def _dimension_indices(dimensions):
    indices = []
    seen = set()
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            continue
        try:
            index = int(dimension.get("dimension_index"))
        except (TypeError, ValueError):
            continue
        if index in seen:
            continue
        indices.append(index)
        seen.add(index)
    return indices


def _cached_svd_dimension_label_map(*dimension_groups):
    indices = []
    seen = set()
    for dimensions in dimension_groups:
        if not isinstance(dimensions, list):
            continue
        for index in _dimension_indices(dimensions):
            if index in seen:
                continue
            indices.append(index)
            seen.add(index)

    if not indices:
        return {}

    try:
        return _normalize_svd_dimension_label_map(cached_svd_dimension_labels(indices))
    except Exception:
        logger.exception("Failed to load cached SVD dimension labels")
        return {}


def _parse_svd_dimension_labels(raw_content, requested_indices):
    text = str(raw_content or "").strip()
    if not text:
        raise ValueError("Empty SVD label response")

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text
    parsed = json.loads(candidate)

    raw_labels = parsed.get("labels") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_labels, list):
        raise ValueError("SVD label response must contain a labels array")

    requested = {int(index) for index in requested_indices}
    labels = []
    seen = set()
    for item in raw_labels:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("dimension_index"))
        except (TypeError, ValueError):
            continue
        if index not in requested or index in seen:
            continue
        label = _clean_overview_text(item.get("label"), 80)
        label = re.sub(r"^(concept|dimension)\s+\d+\s*[:\-]\s*", "", label, flags=re.IGNORECASE).strip()
        if not label:
            continue
        labels.append({
            "dimension_index": index,
            "label": label,
        })
        seen.add(index)

    if not labels:
        raise ValueError("SVD label response did not include usable labels")
    return labels


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Guardian opinion article titles, "
                "summaries, and metadata. Decide whether the user's message needs "
                "article retrieval. Reply with exactly: YES followed by one concise "
                "search query, or NO if article data is not needed."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(.+)", content)
    if yes_match:
        return True, yes_match.group(1).strip().lower()
    if re.search(r"\bYES\b", content):
        return True, user_message
    return False, None


def register_chat_route(app, json_search):
    """Register the /api/chat SSE endpoint. Called from backend/api/routes.py."""

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY or API_KEY not set — add it to your .env file"}), 500

        try:
            client = create_spark_client(api_key=api_key)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        use_search, search_term = llm_search_decision(client, user_message)

        if use_search:
            articles = json_search(search_term or user_message, top_n=8)
            context_text = "\n\n---\n\n".join(
                (
                    f"Title: {article.get('title')}\n"
                    f"Summary: {article.get('summary')}\n"
                    f"Date: {article.get('date')}\n"
                    f"Author: {article.get('author_display') or article.get('author_raw')}\n"
                    f"URL: {article.get('url')}"
                )
                for article in articles
            ) or "No matching Guardian articles found."
            messages = [
                {"role": "system", "content": "Answer questions about Guardian opinion articles using only the article information provided."},
                {"role": "user", "content": f"Article information:\n\n{context_text}\n\nUser question: {user_message}"},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for Guardian opinion article research."},
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and search_term:
                yield f"data: {json.dumps({'search_term': search_term})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/llm/agreement-scores", methods=["POST"])
    def agreement_scores():
        data = request.get_json() or {}
        thesis = (
            data.get("thesis")
            or data.get("statement")
            or data.get("selected_thesis_sentence")
            or ""
        )
        articles = data.get("articles") or data.get("results") or []
        if not str(thesis or "").strip():
            return jsonify({"error": "Thesis is required"}), 400
        if not isinstance(articles, list) or not articles:
            return jsonify({"error": "A non-empty articles list is required"}), 400

        raw_chunking_mode = data.get("chunking_mode") or data.get("chunking")
        legacy_use_chunking = _coerce_bool(
            data.get("use_chunking")
            if "use_chunking" in data
            else data.get("paragraph_chunking"),
            False,
        )
        chunking_mode = normalize_chunking_mode(raw_chunking_mode, DEFAULT_CHUNKING_MODE)
        if raw_chunking_mode is None and legacy_use_chunking and chunking_mode == "none":
            chunking_mode = "paragraph"

        try:
            if chunking_mode != "none":
                scores = score_llm_article_agreement_by_paragraphs(
                    articles=articles,
                    thesis=thesis,
                    chunking_mode=chunking_mode,
                )
            else:
                scores = score_llm_article_agreement(
                    articles=articles,
                    thesis=thesis,
                )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception:
            logger.exception("Agreement scoring request failed")
            return jsonify({"error": "LLM agreement scoring failed"}), 500

        return jsonify({"scores": scores})

    @app.route("/api/llm/query-help", methods=["POST"])
    def query_help():
        payload = request.get_json(silent=True) or {}
        topic = _clean_query_help_text(payload.get("topic"))
        opinion = _clean_query_help_text(payload.get("opinion"))
        action = str(payload.get("action") or "rewrite").strip().lower()

        if not topic and not opinion:
            return jsonify({"error": "Add a topic or stance before asking for AI query help."}), 400
        if action not in {"rewrite", "suggest"}:
            return jsonify({"error": "Unsupported query help action."}), 400

        try:
            retrieval_model = normalize_retrieval_model(
                payload.get("retrieval_model") or DEFAULT_RETRIEVAL_MODEL
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY or API_KEY not set — add it to your .env file"}), 500

        try:
            client = create_spark_client(api_key=api_key)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        method_guidance = _query_help_method_guidance(retrieval_model)
        existing_query = _format_stance_query(topic or "[missing topic]", opinion or "[missing stance]")
        base_context = (
            f"Retrieval method: {retrieval_model}\n"
            f"{method_guidance}\n\n"
            "The interface has exactly two user-editable fields:\n"
            f"- Regarding: {topic or '[blank]'}\n"
            f"- I believe: {opinion or '[blank]'}\n\n"
            f"Current formatted query: {existing_query}\n"
            "The Regarding field drives topic retrieval. The I believe field drives stance/agreement scoring after retrieval. "
            "You may improve both fields, but the I believe clause must preserve the user's original position."
        )

        if action == "rewrite":
            system_prompt = (
                "You improve two-part search queries for a Guardian opinion article retrieval system. "
                "Return valid JSON only. Return exactly three alternatives under an alternatives array. "
                "Each alternative must include topic, opinion, query, and rationale. "
                "The query string must follow this exact template with no trailing punctuation: "
                "\"Regarding <topic>, I believe <opinion>\". "
                "Every alternative must be meaningfully different from the current formatted query. "
                "Rewrite the Regarding clause to improve retrieval with concrete topical language. "
                "Rewrite the I believe clause only to make the same stance clearer, more explicit, and easier to score for agreement. "
                "Do not flip, soften, intensify, broaden, narrow, or add a materially new belief beyond what the user implied. "
                "Keep both clauses concise, specific, and natural. Do not add facts not implied by the user's text; "
                "if one field is blank, infer only a minimal compatible completion from the other field."
            )
            user_prompt = (
                f"{base_context}\n\n"
                "Generate three stronger alternatives that should retrieve better first-stage topic matches for the selected retrieval method, "
                "while preserving the user's intended stance exactly. Do not return the current query unchanged."
            )
        else:
            system_prompt = (
                "You coach users on improving two-part search queries for a Guardian opinion article retrieval system. "
                "Return valid JSON only with a suggestions array of 4 to 6 short strings. "
                "Suggestions must be specific to the selected retrieval method and the user's current Regarding / I believe fields. "
                "Tell users they can clarify the I believe clause for agreement scoring, but should not change the underlying stance."
            )
            user_prompt = (
                f"{base_context}\n\n"
                "Tell the user how to improve this query for the selected retrieval method. "
                "Make the advice actionable and specific to these fields."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = client.chat(messages)
            parsed = _llm_json_object(response.get("content"), "query help")
            if action == "rewrite":
                alternatives = _normalize_query_rewrite_alternatives(
                    parsed.get("alternatives") or parsed.get("queries") or parsed.get("options"),
                    fallback_topic=topic,
                    fallback_opinion=opinion,
                )
                if len(alternatives) < 3:
                    retry_messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"{base_context}\n\n"
                                "Your previous answer did not produce three usable changed alternatives. "
                                "Return JSON only in this exact shape: "
                                "{\"alternatives\":["
                                "{\"topic\":\"...\",\"opinion\":\"...\",\"query\":\"Regarding ..., I believe ...\",\"rationale\":\"...\"},"
                                "{\"topic\":\"...\",\"opinion\":\"...\",\"query\":\"Regarding ..., I believe ...\",\"rationale\":\"...\"},"
                                "{\"topic\":\"...\",\"opinion\":\"...\",\"query\":\"Regarding ..., I believe ...\",\"rationale\":\"...\"}"
                                "]}. "
                                "All three alternatives must be different from the current query, and each must preserve the user's stance."
                            ),
                        },
                    ]
                    retry_response = client.chat(retry_messages)
                    retry_parsed = _llm_json_object(retry_response.get("content"), "query help retry")
                    retry_alternatives = _normalize_query_rewrite_alternatives(
                        retry_parsed.get("alternatives")
                        or retry_parsed.get("queries")
                        or retry_parsed.get("options"),
                        fallback_topic=topic,
                        fallback_opinion=opinion,
                    )
                    if len(retry_alternatives) > len(alternatives):
                        alternatives = retry_alternatives
                if len(alternatives) < 1:
                    raise ValueError("No usable query alternatives returned")
                return jsonify({
                    "alternatives": alternatives[:3],
                    "retrieval_model": retrieval_model,
                })

            suggestions = _normalize_query_improvement_suggestions(
                parsed.get("suggestions") or parsed.get("tips") or parsed.get("advice"),
            )
            if not suggestions:
                raise ValueError("No usable query suggestions returned")
            return jsonify({
                "suggestions": suggestions,
                "retrieval_model": retrieval_model,
            })
        except Exception:
            logger.exception("Query help request failed")
            return jsonify({"error": "LLM query help failed."}), 500

    @app.route("/api/llm/explain-ranking", methods=["POST"])
    def explain_ranking():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "").strip()
        position = payload.get("position")
        article = payload.get("article") or {}
        retrieval_model = normalize_retrieval_model(
            payload.get("retrieval_model")
            or article.get("chunk_retrieval_model")
            or DEFAULT_RETRIEVAL_MODEL
        )
        chunking_mode = normalize_chunking_mode(
            payload.get("chunking_mode")
            or article.get("chunk_retrieval_chunking_mode")
            or article.get("llm_chunking_mode")
            or DEFAULT_CHUNKING_MODE,
            DEFAULT_CHUNKING_MODE,
        )
        query_svd_dimensions = payload.get("query_svd_dimensions") or []
        provided_dimension_labels = _normalize_svd_dimension_label_map(
            payload.get("svd_dimension_labels") or payload.get("dimension_labels")
        )

        if not query:
            return jsonify({"error": "Query text is required."}), 400
        if not isinstance(article, dict) or not article:
            return jsonify({"error": "Article payload is required."}), 400

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY or API_KEY not set — add it to your .env file"}), 500

        try:
            client = create_spark_client(api_key=api_key)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        dimension_labels = {}
        if retrieval_model == "svd":
            cached_dimension_labels = _cached_svd_dimension_label_map(
                query_svd_dimensions,
                article.get("svd_query_chart_dimensions"),
                article.get("svd_chart_dimensions"),
                article.get("svd_dimensions"),
            )
            dimension_labels = {
                **cached_dimension_labels,
                **provided_dimension_labels,
            }

        position_label = str(position) if position is not None else "unknown"
        prompt = _build_ranking_explanation_prompt(
            query=query,
            position_label=position_label,
            article=article,
            retrieval_model=retrieval_model,
            chunking_mode=chunking_mode,
            query_svd_dimensions=query_svd_dimensions,
            dimension_labels=dimension_labels,
        )

        messages = [
            {"role": "system", "content": _ranking_mode_system_prompt(retrieval_model)},
            {"role": "user", "content": prompt},
        ]

        try:
            response = client.chat(messages)
            explanation = (response.get("content") or "").strip()
            if not explanation:
                raise RuntimeError("Received empty explanation from the LLM.")
        except Exception as exc:
            logger.exception("Ranking explanation request failed")
            return jsonify({"error": "LLM ranking explanation failed."}), 500

        return jsonify({"explanation": explanation})

    @app.route("/api/llm/svd-dimension-labels", methods=["POST"])
    def svd_dimension_labels():
        payload = request.get_json(silent=True) or {}
        dimensions = payload.get("dimensions") or []
        if not isinstance(dimensions, list) or not dimensions:
            return jsonify({"error": "A non-empty dimensions list is required."}), 400

        requested_indices = _dimension_indices(dimensions)
        cached_labels = cached_svd_dimension_labels(requested_indices)
        cached_indices = {
            int(item["dimension_index"])
            for item in cached_labels
            if isinstance(item, dict) and "dimension_index" in item
        }
        if cached_labels and len(cached_indices) == len(set(requested_indices)):
            return jsonify({"labels": cached_labels, "source": "precomputed"})

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            if cached_labels:
                return jsonify({"labels": cached_labels, "source": "precomputed_partial"})
            return jsonify({"error": "SPARK_API_KEY or API_KEY not set — add it to your .env file"}), 500

        label_inputs = []
        for dimension in dimensions[:24]:
            if not isinstance(dimension, dict):
                continue
            try:
                dimension_index = int(dimension.get("dimension_index"))
            except (TypeError, ValueError):
                continue
            if dimension_index in cached_indices:
                continue
            formatted = _format_dimension_for_labeling(dimension)
            if formatted is not None:
                label_inputs.append(formatted)

        if not label_inputs:
            if cached_labels:
                return jsonify({"labels": cached_labels, "source": "precomputed"})
            return jsonify({"error": "No usable SVD dimensions were provided."}), 400

        try:
            client = create_spark_client(api_key=api_key)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        system_prompt = (
            "You label latent SVD dimensions for a news opinion search interface. "
            "Each dimension is represented by its top terms and may include a signed loading value. "
            "Write a short human-readable topic label for each dimension, 2 to 6 words long. "
            "Use broad topic language such as 'Immigration and asylum policy' or 'Climate and energy politics'. "
            "Do not include the words Concept or Dimension, do not include numbers, and do not explain. "
            "Return valid JSON only with this shape: "
            "{\"labels\":[{\"dimension_index\":0,\"label\":\"Example topic\"}]}"
        )
        user_prompt = json.dumps({"dimensions": label_inputs}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = client.chat(messages)
            live_labels = _parse_svd_dimension_labels(
                response.get("content"),
                requested_indices=[item["dimension_index"] for item in label_inputs],
            )
        except Exception:
            logger.exception("SVD dimension labeling request failed")
            return jsonify({"error": "LLM SVD dimension labeling failed"}), 500

        labels = cached_labels + live_labels
        return jsonify({"labels": labels, "source": "mixed" if cached_labels else "llm"})

    @app.route("/api/llm/results-overview", methods=["POST"])
    def results_overview():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "").strip()
        articles = payload.get("articles") or payload.get("results") or []
        mode = str(payload.get("mode") or "search").strip().lower()

        if not query:
            return jsonify({"error": "Query text is required."}), 400
        if not isinstance(articles, list) or not articles:
            return jsonify({"error": "A non-empty articles list is required."}), 400

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY or API_KEY not set — add it to your .env file"}), 500

        try:
            client = create_spark_client(api_key=api_key)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        usable_articles = [
            article for article in articles
            if isinstance(article, dict) and not bool(article.get("llm_irrelevant"))
        ][:10]
        if not usable_articles:
            return jsonify({"error": "No relevant articles are available to summarize."}), 400

        context_text = "\n\n---\n\n".join(
            _format_article_for_results_overview(article, index)
            for index, article in enumerate(usable_articles)
        )
        sources = [
            _source_for_results_overview(article, index)
            for index, article in enumerate(usable_articles)
        ]

        system_prompt = (
            "You write a concise AI overview for a Guardian opinion article results page. "
            "Summarize the retrieved results as a collection, not one article at a time. "
            "Use only the supplied search query and article result snippets. Do not add outside facts. "
            "Focus on the overall pattern: whether the results mostly support, oppose, complicate, or split on the user's view; "
            "what central claims repeat across articles; and where the retrieved articles differ from each other. "
            "Include key arguments that support the user's view and key arguments that challenge or oppose it. "
            "Attach source_indices to every argument and evidence item using the 1-based Result numbers from the snippets. "
            "Only cite a Result number when that supplied snippet directly supports the argument or evidence. "
            "Do not write a bullet per article, do not list titles, and do not make claims about articles not shown. "
            "Be careful about uncertainty: describe patterns in the retrieved results, not the whole news landscape. "
            "Return valid JSON only with keys: overview, key_points, supporting_arguments, opposing_arguments, key_evidence, caveat. "
            "overview must be 2-3 short sentences about the overall result set. "
            "key_points must be an array of 2-4 short strings covering shared themes, differences, or agreement patterns. "
            "supporting_arguments and opposing_arguments must each be arrays of 1-3 objects shaped "
            "{\"argument\":\"...\",\"source_indices\":[1],\"evidence\":[{\"evidence\":\"...\",\"source_indices\":[1]}]}. "
            "key_evidence must be an array of 2-5 objects shaped {\"evidence\":\"...\",\"source_indices\":[1,2]}. "
            "caveat must be one short sentence about limits of the retrieved results."
        )
        user_prompt = (
            f"Search mode: {mode}\n"
            f"Search query:\n{query}\n\n"
            f"Top retrieved article snippets:\n\n{context_text}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = client.chat(messages)
            overview = _parse_results_overview_response(
                response.get("content"),
                max_source_index=_max_source_index(sources, len(usable_articles)),
            )
        except Exception:
            logger.exception("Results overview request failed")
            return jsonify({"error": "LLM results overview failed"}), 500

        overview["sources"] = sources
        return jsonify(overview)

    @app.route("/api/llm/results-chat", methods=["POST"])
    def results_chat():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question") or payload.get("message") or "").strip()
        query = str(payload.get("query") or "").strip()
        articles = payload.get("articles") or payload.get("results") or []
        mode = str(payload.get("mode") or "search").strip().lower()
        article_scope = str(payload.get("article_scope") or payload.get("scope") or "").strip().lower()
        history = payload.get("history") or []

        if not question:
            return jsonify({"error": "Question text is required."}), 400
        if not isinstance(articles, list) or not articles:
            return jsonify({"error": "A non-empty articles list is required."}), 400

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY or API_KEY not set — add it to your .env file"}), 500

        try:
            client = create_spark_client(api_key=api_key)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        usable_articles = [
            article for article in articles
            if isinstance(article, dict) and not bool(article.get("llm_irrelevant"))
        ][:10]
        if not usable_articles:
            return jsonify({"error": "No relevant articles are available for chat."}), 400

        sources = [
            _source_for_results_overview(article, index)
            for index, article in enumerate(usable_articles)
        ]
        is_selected_article_scope = article_scope in {
            "selected",
            "selected_articles",
            "attached",
            "attachments",
        }
        body_lookup = {}
        if is_selected_article_scope:
            try:
                body_lookup = article_body_text_lookup(usable_articles)
            except Exception:
                logger.exception("Selected article body lookup failed")
                body_lookup = {}

        max_body_chars = 7000 if len(usable_articles) <= 2 else 4200 if len(usable_articles) <= 5 else 2600
        if is_selected_article_scope:
            context_text = "\n\n---\n\n".join(
                _format_article_for_results_chat(
                    article,
                    index,
                    body_text=body_lookup.get(str(article.get("id") or "").strip()),
                    max_body_chars=max_body_chars,
                )
                for index, article in enumerate(usable_articles)
            )
        else:
            context_text = "\n\n---\n\n".join(
                _format_article_for_results_overview(article, index)
                for index, article in enumerate(usable_articles)
            )

        history_lines = []
        if isinstance(history, list):
            for item in history[-6:]:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip().lower()
                content = _clean_overview_text(item.get("content"), 500)
                if role in {"user", "assistant"} and content:
                    history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "None"

        system_prompt = (
            "You answer follow-up questions about a Guardian opinion article results page. "
            "Use only the supplied search query, prior chat, and article context. "
            "Do not add outside facts or claim you know article details beyond the supplied context. "
            "If article attachments are supplied, focus on those selected articles. "
            "When the answer depends on specific results, cite them with source_indices using the 1-based Result numbers. "
            "If the snippets do not contain enough information, say what is missing and answer only what can be inferred. "
            "Return valid JSON only with keys: answer, source_indices, follow_up_questions. "
            "answer must be concise but useful. source_indices must list every Result number that directly supports the answer. "
            "follow_up_questions must be an array of 0-3 short suggested questions."
        )
        user_prompt = (
            f"Search mode: {mode}\n"
            f"Article scope: {'selected article attachments' if is_selected_article_scope else 'top retrieved results'}\n"
            f"Original search query:\n{query or 'Not provided'}\n\n"
            f"Prior chat:\n{history_text}\n\n"
            f"Article context:\n\n{context_text}\n\n"
            f"User question:\n{question}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = client.chat(messages)
            answer = _parse_results_chat_response(
                response.get("content"),
                max_source_index=_max_source_index(sources, len(usable_articles)),
            )
            if is_selected_article_scope:
                answer["source_indices"] = _remap_result_indices(
                    answer.get("source_indices"),
                    usable_articles,
                )
        except Exception:
            logger.exception("Results chat request failed")
            return jsonify({"error": "LLM results chat failed"}), 500

        answer["sources"] = sources
        return jsonify(answer)
