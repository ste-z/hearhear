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
    create_spark_client,
    score_llm_article_agreement,
    score_llm_article_agreement_by_paragraphs,
)
from backend.stance_processing.stance_rerank import (
    DEFAULT_CHUNKING_MODE,
    normalize_chunking_mode,
)
from backend.text_processing.svd_dimension_labels import cached_svd_dimension_labels

logger = logging.getLogger(__name__)


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


def _format_svd_dimensions(dimensions):
    if not isinstance(dimensions, list) or not dimensions:
        return "None"

    lines = []
    for dimension in dimensions:
        try:
            index = int(dimension.get("dimension_index", -1))
        except (TypeError, ValueError):
            index = -1

        label = str(dimension.get("label_text") or dimension.get("dimension_label") or index)
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
        label_terms = [str(term).strip() for term in dimension.get("label_terms") or [] if str(term).strip()]
        term_text = f"terms: {', '.join(label_terms)}" if label_terms else None

        parts = [f"Dimension {index}", f"label: {label}", f"value: {value_text}"]
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


def _format_article_for_results_overview(article, index):
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

    lines = [f"Result {index + 1}: {title or 'Untitled'}"]
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


def _parse_results_overview_response(raw_content):
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

    @app.route("/api/llm/explain-ranking", methods=["POST"])
    def explain_ranking():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "").strip()
        position = payload.get("position")
        article = payload.get("article") or {}
        query_svd_dimensions = payload.get("query_svd_dimensions") or []

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

        query_svd_text = _format_svd_dimensions(query_svd_dimensions)
        article_query_chart_text = _format_svd_dimensions(article.get("svd_query_chart_dimensions"))
        article_chart_text = _format_svd_dimensions(article.get("svd_chart_dimensions"))
        article_dimensions_text = _format_svd_dimensions(article.get("svd_dimensions"))

        article_title = str(article.get("title") or "").strip()
        article_summary = str(article.get("summary") or "").strip()
        article_claim = str(article.get("central_claim_summary") or "").strip()
        article_score = (
            article.get("topic_score_display")
            or article.get("topic_score")
            or article.get("combined_score")
            or article.get("score")
            or None
        )
        if article_score is not None:
            article_score = str(article_score)

        position_label = str(position) if position is not None else "unknown"

        prompt = (
            "This retrieval system represents queries and articles in a shared latent semantic space via SVD. "
            "Articles are ranked based on similarity to the query in this space. "
            "Given the query, the article metadata, and the available SVD representations, explain why this article is ranked at position "
            f"{position_label}. Focus on shared themes, concepts, or terminology that likely contributed to high similarity. "
            "Please keep your response concise, with a maximum of 3 sentences to make it easy for users to get a quick insight into the ranking."
            "Do not invent unobserved article content beyond the title, summary, claim, and provided SVD dimension metadata. \n\n"
            f"Query:\n{query}\n\n"
            "Article:\n"
            f"Title: {article_title or 'N/A'}\n"
            f"Summary: {article_summary or 'N/A'}\n"
        )
        if article_claim:
            prompt += f"Claim: {article_claim}\n"
        if article_score is not None:
            prompt += f"Similarity score: {article_score}\n"

        prompt += (
            "\nQuery SVD dimensions:\n"
            f"{query_svd_text}\n\n"
            "Article SVD query chart dimensions:\n"
            f"{article_query_chart_text}\n\n"
            "Article shared corpus SVD dimensions:\n"
            f"{article_chart_text}\n\n"
            "Article top SVD dimensions:\n"
            f"{article_dimensions_text}\n"
        )

        messages = [
            {"role": "system", "content": "You are an expert data analyst explaining search ranking behavior in an SVD-based retrieval system."},
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

        system_prompt = (
            "You write a concise AI overview for a Guardian opinion article results page. "
            "Summarize the retrieved results as a collection, not one article at a time. "
            "Use only the supplied search query and article result snippets. Do not add outside facts. "
            "Focus on the overall pattern: whether the results mostly support, oppose, complicate, or split on the user's view; "
            "what central claims repeat across articles; and where the retrieved articles differ from each other. "
            "Do not write a bullet per article, do not list titles, and do not make claims about articles not shown. "
            "Be careful about uncertainty: describe patterns in the retrieved results, not the whole news landscape. "
            "Return valid JSON only with keys: overview, key_points, caveat. "
            "overview must be 2-3 short sentences about the overall result set. "
            "key_points must be an array of 2-4 short strings covering shared themes, differences, or agreement patterns. "
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
            overview = _parse_results_overview_response(response.get("content"))
        except Exception:
            logger.exception("Results overview request failed")
            return jsonify({"error": "LLM results overview failed"}), 500

        return jsonify(overview)
